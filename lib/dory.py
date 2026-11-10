"""Critical Module: Dory-4B architecture, loop controller, logit lens per loop, and untied token initialization.

Implements the exact architecture and weights layout of `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16`
(nGPT hyperspherical normalization + Looped Recurrent Middle Block) as a native HuggingFace
`PreTrainedModel` (`DoryConfig`, `DoryModel`, `DoryForCausalLM`) registered with `AutoConfig`
and `AutoModelForCausalLM`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        PretrainedConfig,
        PreTrainedModel,
    )
    from transformers.generation import GenerationMixin
    from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _HAS_TORCH = False

    class PretrainedConfig:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

    class PreTrainedModel:  # type: ignore[no-redef]
        pass

    class GenerationMixin:  # type: ignore[no-redef]
        pass

    class _DummyNN:
        Module = object
        Embedding = object
        Linear = object
        Parameter = object
        ModuleList = list

    nn = _DummyNN()  # type: ignore[assignment]

DORY_HF_REPO = "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"
NANBEIGE_HF_REPO = "Nanbeige/Nanbeige4.2-3B"


def justnorm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Hyperspherical L2 normalization in float32, cast back to `x.dtype` (nGPT / Dory)."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).sum(dim=dim, keepdim=True) + eps)
    return y.to(x.dtype)


def slerp_norm(a: torch.Tensor, b: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
    """Normalized linear interpolation on the hypersphere (nGPT residual update)."""
    ya = justnorm(a).float()
    yb = justnorm(b).float()
    return justnorm(ya + lr.float() * (yb - ya)).to(a.dtype)


def slerp_alpha(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: torch.Tensor,
    alpha_scale: float,
) -> torch.Tensor:
    """Eigenlearning rate residual update using `|alpha * alpha_scale|`."""
    return slerp_norm(a, b, (alpha.float() * alpha_scale).abs())


def _default_dory_layer_types(num_hidden_layers: int = 72) -> list[str]:
    """Generate the repeating 8-sublayer pattern from Dory's config.json:
    3 sliding_attention+mlp pairs followed by 1 full_attention+mlp pair.
    """
    pattern = [
        "sliding_attention",
        "mlp",
        "sliding_attention",
        "mlp",
        "sliding_attention",
        "mlp",
        "full_attention",
        "mlp",
    ]
    types: list[str] = []
    for i in range(num_hidden_layers):
        types.append(pattern[i % len(pattern)])
    return types


class DoryConfig(PretrainedConfig):
    """Configuration class for NVIDIA Nemotron-Labs Dory1.0-4B-BF16."""

    model_type = "dory"

    def __init__(
        self,
        vocab_size: int = 131072,
        hidden_size: int = 2560,
        num_hidden_layers: int = 72,
        intermediate_size: int = 7168,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        head_dim: int = 256,
        max_position_embeddings: int = 262144,
        tie_word_embeddings: bool = False,
        ngpt_alpha_init: float = 0.05,
        attention_output_gate_logit_scale_init: float | None = None,
        mlp_bias: bool = False,
        attention_bias: bool = False,
        n_input_layers: int = 16,
        n_recurrent_layers: int = 40,
        n_output_layers: int = 16,
        n_recurrent_loops: int = 2,
        hidden_act: str = "silu",
        initializer_range: float = 0.0177,
        sliding_window: int = 1024,
        layer_types: list[str] | None = None,
        rope_parameters: dict[str, Any] | None = None,
        kv_cache_mode: Literal["per_loop", "last_loop"] = "per_loop",
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.ngpt_alpha_init = ngpt_alpha_init
        self.attention_output_gate_logit_scale_init = attention_output_gate_logit_scale_init
        self.mlp_bias = mlp_bias
        self.attention_bias = attention_bias
        self.n_input_layers = n_input_layers
        self.n_recurrent_layers = n_recurrent_layers
        self.n_output_layers = n_output_layers
        self.n_recurrent_loops = n_recurrent_loops
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.sliding_window = sliding_window
        self.layer_types = (
            list(layer_types)
            if layer_types is not None
            else _default_dory_layer_types(num_hidden_layers)
        )
        self.rope_parameters = rope_parameters or {
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
            },
            "full_attention": {
                "rope_type": "default",
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 1.0,
            },
        }
        self.kv_cache_mode: Literal["per_loop", "last_loop"] = kv_cache_mode


class DoryLoopCache:
    """KV Cache supporting Dory's recurrent middle block in both 'per_loop' and 'last_loop' modes.

    In 'per_loop' mode, each `(layer_idx, loop_idx)` pair maintains its own KV history across
    generated tokens.
    In 'last_loop' mode, recurrent layers share a single cache slot `(layer_idx, 0)` per token
    position: during the `n_loops` passes of the current step, earlier loops read the cached
    past keys/values from previous tokens and overwrite the current token slice so that after
    the final loop completes, only the last loop's KV representation is retained.
    """

    def __init__(self, mode: Literal["per_loop", "last_loop"] = "per_loop") -> None:
        self.mode: Literal["per_loop", "last_loop"] = mode
        self._store: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self.seen_tokens: int = 0

    def get_seq_length(self, layer_idx: int = 0) -> int:
        for (l_idx, _), (k, _) in self._store.items():
            if l_idx == layer_idx:
                return int(k.shape[2])
        if self._store:
            first_k = next(iter(self._store.values()))[0]
            return int(first_k.shape[2])
        return 0

    def get_max_length(self) -> int | None:
        return None

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        loop_idx: int = 0,
        is_last_loop: bool = True,
        past_seen_tokens: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update and return `(k_full, v_full)` of shape `[batch, kv_heads, seq_total, head_dim]`."""
        slot_loop = loop_idx if self.mode == "per_loop" else 0
        key = (layer_idx, slot_loop)

        if key not in self._store:
            self._store[key] = (key_states, value_states)
            return key_states, value_states

        prev_k, prev_v = self._store[key]
        if self.mode == "last_loop" and prev_k.shape[2] > past_seen_tokens:
            # Earlier loop in the SAME generation step already appended new tokens; replace tail
            prev_k = prev_k[:, :, :past_seen_tokens, :]
            prev_v = prev_v[:, :, :past_seen_tokens, :]

        full_k = torch.cat([prev_k, key_states], dim=2)
        full_v = torch.cat([prev_v, value_states], dim=2)
        self._store[key] = (full_k, full_v)
        return full_k, full_v


class DoryRotaryEmbedding(nn.Module):
    """Standard half-rotation RoPE with configurable theta."""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = float(theta)
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # q, k: [batch, seq, heads, head_dim]
        # position_ids: [batch, seq]
        freqs = torch.einsum(
            "bs,d->bsd",
            position_ids.float(),
            self.inv_freq.to(device=q.device, dtype=torch.float32),
        )
        cos = freqs.cos().unsqueeze(2)  # [batch, seq, 1, head_dim // 2]
        sin = freqs.sin().unsqueeze(2)

        def _apply_rotary(x: torch.Tensor) -> torch.Tensor:
            xf = x.float()
            d2 = xf.shape[-1] // 2
            x1, x2 = xf[..., :d2], xf[..., d2:]
            out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
            return out.to(x.dtype)

        return _apply_rotary(q), _apply_rotary(k)


class DoryAttention(nn.Module):
    """Dory nGPT Attention mixer with QK hyperspherical normalization and sigmoid output gate."""

    def __init__(
        self,
        config: DoryConfig,
        layer_idx: int,
        is_sliding: bool,
        rope: DoryRotaryEmbedding,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_sliding = is_sliding
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.sliding_window = config.sliding_window if is_sliding else None
        self.rope = rope

        self.q_proj = nn.Linear(config.hidden_size, self.q_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.kv_size, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.kv_size, bias=config.attention_bias)
        self.g_proj = nn.Linear(config.hidden_size, self.q_size, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.q_size, config.hidden_size, bias=config.attention_bias)

        self.sqk = nn.Parameter(torch.full((self.q_size,), config.initializer_range))
        self._sqk_factor = (self.head_dim**0.5) / config.initializer_range

        self.s_gate = nn.Parameter(torch.full((self.q_size,), config.initializer_range))
        g0 = (
            config.attention_output_gate_logit_scale_init
            if config.attention_output_gate_logit_scale_init is not None
            else (config.hidden_size**0.5)
        )
        self._s_gate_factor = (g0**0.5) / config.initializer_range

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: DoryLoopCache | None = None,
        loop_idx: int = 0,
        is_last_loop: bool = True,
        past_seen_tokens: int = 0,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        gate = self.g_proj(hidden_states)

        q, k = self.rope(q, k, position_ids)

        sqk = (self.sqk.float() * self._sqk_factor).view(1, 1, self.num_heads, self.head_dim)
        q = (sqk.square() * justnorm(q).float()).to(q.dtype)
        k = justnorm(k)

        # Transpose to [bsz, heads, seq, head_dim]
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        if past_key_values is not None:
            k_t, v_t = past_key_values.update(
                k_t,
                v_t,
                layer_idx=self.layer_idx,
                loop_idx=loop_idx,
                is_last_loop=is_last_loop,
                past_seen_tokens=past_seen_tokens,
            )

        kv_len = k_t.shape[2]
        if self.num_kv_heads != self.num_heads:
            k_rep = k_t.repeat_interleave(self.num_queries_per_kv, dim=1)
            v_rep = v_t.repeat_interleave(self.num_queries_per_kv, dim=1)
        else:
            k_rep = k_t
            v_rep = v_t

        # Build causal + optional sliding-window + optional padding mask
        # Query positions correspond to [kv_len - q_len .. kv_len - 1]
        q_pos = torch.arange(kv_len - q_len, kv_len, device=hidden_states.device).unsqueeze(1)
        k_pos = torch.arange(kv_len, device=hidden_states.device).unsqueeze(0)
        allowed = k_pos <= q_pos  # [q_len, kv_len]
        if self.sliding_window is not None and self.sliding_window > 0:
            allowed = allowed & ((q_pos - k_pos) < self.sliding_window)

        attn_bias = torch.zeros(
            (1, 1, q_len, kv_len), dtype=q_t.dtype, device=hidden_states.device
        )
        attn_bias = attn_bias.masked_fill(~allowed.unsqueeze(0).unsqueeze(0), float("-inf"))

        if attention_mask is not None and attention_mask.ndim == 2:
            # attention_mask: [bsz, total_seq] or [bsz, q_len] where 1 = valid, 0 = pad
            mask_len = attention_mask.shape[1]
            if mask_len == kv_len:
                pad_mask = attention_mask[:, None, None, :].to(torch.bool)
                attn_bias = attn_bias.expand(bsz, 1, q_len, kv_len).clone()
                attn_bias = attn_bias.masked_fill(~pad_mask, float("-inf"))
                # Prevent all-masked rows from producing NaNs in softmax
                all_masked = ~ (allowed.unsqueeze(0) & pad_mask.squeeze(1))
                attn_bias = torch.where(
                    all_masked.all(dim=-1, keepdim=True).unsqueeze(1),
                    torch.zeros_like(attn_bias),
                    attn_bias,
                )

        attn_out = F.scaled_dot_product_attention(
            q_t,
            k_rep,
            v_rep,
            attn_mask=attn_bias,
            scale=self.scaling,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, self.q_size)

        sg = (self.s_gate.float() * self._s_gate_factor).square()
        attn_out = (attn_out.float() * torch.sigmoid(gate.float() * sg)).to(attn_out.dtype)
        return self.o_proj(justnorm(attn_out))


class DoryMLP(nn.Module):
    """Dory nGPT SwiGLU MLP sublayer with `suv_gate` / `suv_up` scaling and `justnorm`."""

    def __init__(self, config: DoryConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.suv_gate = nn.Parameter(torch.ones(config.intermediate_size))
        self.suv_up = nn.Parameter(torch.ones(config.intermediate_size))
        self._suv_factor = config.hidden_size**0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = (self.gate_proj(x).float() * (self.suv_gate.float() * self._suv_factor)).to(x.dtype)
        u = (self.up_proj(x).float() * (self.suv_up.float() * self._suv_factor)).to(x.dtype)
        m = (F.silu(g.float()) * u.float()).to(x.dtype)
        return self.down_proj(justnorm(m))


class DoryLayer(nn.Module):
    """Single Dory sublayer: either an attention sublayer (`attn_alpha`) or MLP sublayer (`mlp_alpha`)."""

    def __init__(
        self,
        config: DoryConfig,
        layer_idx: int,
        layer_type: str,
        rope_sliding: DoryRotaryEmbedding,
        rope_full: DoryRotaryEmbedding,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        self.is_mlp = layer_type == "mlp"
        self._alpha_scale = config.ngpt_alpha_init / config.initializer_range

        if self.is_mlp:
            self.mlp_alpha = nn.Parameter(
                torch.full((config.hidden_size,), config.initializer_range)
            )
            self.mixer: nn.Module = DoryMLP(config)
        else:
            self.attn_alpha = nn.Parameter(
                torch.full((config.hidden_size,), config.initializer_range)
            )
            is_sliding = layer_type == "sliding_attention"
            self.mixer = DoryAttention(
                config=config,
                layer_idx=layer_idx,
                is_sliding=is_sliding,
                rope=rope_sliding if is_sliding else rope_full,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: DoryLoopCache | None = None,
        loop_idx: int = 0,
        is_last_loop: bool = True,
        past_seen_tokens: int = 0,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.is_mlp:
            out = self.mixer(hidden_states)
            return slerp_alpha(hidden_states, out, self.mlp_alpha, self._alpha_scale)
        out = self.mixer(
            hidden_states,
            position_ids=position_ids,
            past_key_values=past_key_values,
            loop_idx=loop_idx,
            is_last_loop=is_last_loop,
            past_seen_tokens=past_seen_tokens,
            attention_mask=attention_mask,
        )
        return slerp_alpha(hidden_states, out, self.attn_alpha, self._alpha_scale)


class DoryLMHead(nn.Module):
    """Untied output projection head with per-token `logit_scale` matching `model.safetensors`."""

    def __init__(self, config: DoryConfig) -> None:
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.empty(config.vocab_size, config.hidden_size))
        self.logit_scale = nn.Parameter(
            torch.full((config.vocab_size,), config.initializer_range)
        )
        nn.init.normal_(self.weight, mean=0.0, std=config.initializer_range)

    @property
    def in_features(self) -> int:
        return int(self.weight.shape[1])

    @property
    def out_features(self) -> int:
        return int(self.weight.shape[0])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = F.linear(hidden_states, self.weight).float()
        scale = self.logit_scale.float() / self.config.initializer_range
        return logits * scale


class DoryPreTrainedModel(PreTrainedModel):
    """Base class for Dory models."""

    config_class = DoryConfig
    base_model_prefix = "backbone"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DoryLayer"]

    def _init_weights(self, module: nn.Module) -> None:
        std = getattr(self.config, "initializer_range", 0.0177)
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)


class DoryModel(DoryPreTrainedModel):
    """Dory backbone (`embeddings` + `layers` split into input, recurrent, and output blocks)."""

    def __init__(self, config: DoryConfig) -> None:
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        rp = config.rope_parameters or {}
        sliding_theta = float(rp.get("sliding_attention", {}).get("rope_theta", 10000.0))
        full_theta = float(rp.get("full_attention", {}).get("rope_theta", 10000000.0))
        self.rope_sliding = DoryRotaryEmbedding(config.head_dim, theta=sliding_theta)
        self.rope_full = DoryRotaryEmbedding(config.head_dim, theta=full_theta)

        self.layers = nn.ModuleList(
            [
                DoryLayer(
                    config=config,
                    layer_idx=i,
                    layer_type=config.layer_types[i],
                    rope_sliding=self.rope_sliding,
                    rope_full=self.rope_full,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.gradient_checkpointing = False
        self.post_init()

    @property
    def embed_tokens(self) -> nn.Embedding:
        """Alias `embed_tokens` -> `embeddings` for compatibility with PEFT/TRL utilities."""
        return self.embeddings

    @embed_tokens.setter
    def embed_tokens(self, value: nn.Embedding) -> None:
        self.embeddings = value

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embeddings = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: DoryLoopCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        n_loops: int | None = None,
        collect_loop_states: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, DoryLoopCache | None, list[torch.Tensor]]:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either `input_ids` or `inputs_embeds` must be provided.")
            h = self.embeddings(input_ids)
        else:
            h = inputs_embeds

        bsz, seq_len, _ = h.shape
        use_cache = bool(use_cache) if use_cache is not None else False
        if use_cache and not isinstance(past_key_values, DoryLoopCache):
            past_key_values = DoryLoopCache(mode=getattr(self.config, "kv_cache_mode", "per_loop"))

        past_seen_tokens = past_key_values.seen_tokens if past_key_values is not None else 0
        if position_ids is None:
            if attention_mask is not None and attention_mask.ndim == 2:
                # Account for left-padding if present
                pos_full = attention_mask.long().cumsum(-1) - 1
                pos_full.masked_fill_(attention_mask == 0, 0)
                position_ids = pos_full[:, -seq_len:]
            else:
                position_ids = (
                    torch.arange(
                        past_seen_tokens,
                        past_seen_tokens + seq_len,
                        device=h.device,
                        dtype=torch.long,
                    )
                    .unsqueeze(0)
                    .expand(bsz, -1)
                )

        n_in = self.config.n_input_layers
        n_rec = self.config.n_recurrent_layers
        active_loops = int(n_loops if n_loops is not None else self.config.n_recurrent_loops)
        if active_loops < 1:
            raise ValueError(f"n_recurrent_loops must be >= 1, got {active_loops}")

        # 1. Input block (executed once)
        for layer in self.layers[:n_in]:
            h = layer(
                h,
                position_ids=position_ids,
                past_key_values=past_key_values,
                loop_idx=0,
                is_last_loop=True,
                past_seen_tokens=past_seen_tokens,
                attention_mask=attention_mask,
            )

        # 2. Recurrent middle block (executed `active_loops` times)
        input_h = h
        state = input_h
        loop_states: list[torch.Tensor] = []

        for loop_idx in range(active_loops):
            is_last = loop_idx == (active_loops - 1)
            h = justnorm(state.float() + input_h.float()).to(state.dtype)
            for layer in self.layers[n_in : n_in + n_rec]:
                h = layer(
                    h,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    loop_idx=loop_idx,
                    is_last_loop=is_last,
                    past_seen_tokens=past_seen_tokens,
                    attention_mask=attention_mask,
                )
            state = justnorm(h)
            h = state
            if collect_loop_states:
                loop_states.append(state)

        # 3. Output block (executed once)
        for layer in self.layers[n_in + n_rec :]:
            h = layer(
                h,
                position_ids=position_ids,
                past_key_values=past_key_values,
                loop_idx=0,
                is_last_loop=True,
                past_seen_tokens=past_seen_tokens,
                attention_mask=attention_mask,
            )

        if past_key_values is not None:
            past_key_values.seen_tokens = past_seen_tokens + seq_len

        return h, past_key_values, loop_states


class DoryForCausalLM(DoryPreTrainedModel, GenerationMixin):
    """HuggingFace-compatible CausalLM for Dory1.0-4B-BF16.

    Weight keys match `model.safetensors` 1-to-1:
    - `backbone.embeddings.weight`
    - `backbone.layers.{0..71}.*`
    - `lm_head.weight`
    - `lm_head.logit_scale`
    """

    _tied_weights_keys: list[str] = []

    def __init__(self, config: DoryConfig) -> None:
        super().__init__(config)
        self.backbone = DoryModel(config)
        self.lm_head = DoryLMHead(config)
        self.post_init()

    @property
    def model(self) -> DoryModel:
        return self.backbone

    def get_input_embeddings(self) -> nn.Embedding:
        return self.backbone.embeddings

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.backbone.embeddings = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.lm_head = new_embeddings  # type: ignore[assignment]

    def set_loops(
        self,
        n_loops: int,
        kv_cache_mode: Literal["per_loop", "last_loop"] | None = None,
    ) -> None:
        """Set `n_recurrent_loops` (e.g. 1, 2, or 4) and optional `kv_cache_mode` at runtime."""
        if n_loops < 1:
            raise ValueError(f"n_loops must be >= 1, got {n_loops}")
        self.config.n_recurrent_loops = int(n_loops)
        self.backbone.config.n_recurrent_loops = int(n_loops)
        if kv_cache_mode is not None:
            if kv_cache_mode not in ("per_loop", "last_loop"):
                raise ValueError(f"Invalid kv_cache_mode: {kv_cache_mode}")
            self.config.kv_cache_mode = kv_cache_mode
            self.backbone.config.kv_cache_mode = kv_cache_mode

    def resize_token_embeddings(
        self,
        new_num_tokens: int | None = None,
        pad_to_multiple_of: int | None = None,
        mean_resizing: bool = False,
    ) -> nn.Embedding:
        """Resize both `backbone.embeddings` and untied `lm_head` (`weight` + `logit_scale`)."""
        if new_num_tokens is None:
            return self.get_input_embeddings()

        if pad_to_multiple_of is not None and new_num_tokens % pad_to_multiple_of != 0:
            new_num_tokens = ((new_num_tokens // pad_to_multiple_of) + 1) * pad_to_multiple_of

        old_emb = self.backbone.embeddings
        old_num_tokens, hidden_dim = old_emb.weight.shape
        if new_num_tokens == old_num_tokens:
            return old_emb

        new_emb = nn.Embedding(
            new_num_tokens,
            hidden_dim,
            device=old_emb.weight.device,
            dtype=old_emb.weight.dtype,
        )
        nn.init.normal_(new_emb.weight, mean=0.0, std=self.config.initializer_range)
        copy_count = min(old_num_tokens, new_num_tokens)
        with torch.no_grad():
            new_emb.weight[:copy_count].copy_(old_emb.weight[:copy_count])
            if new_num_tokens > old_num_tokens:
                mean_vec = old_emb.weight.float().mean(dim=0).to(old_emb.weight.dtype)
                new_emb.weight[old_num_tokens:].copy_(mean_vec.unsqueeze(0).expand(new_num_tokens - old_num_tokens, -1))
        self.backbone.embeddings = new_emb

        # Resize untied DoryLMHead (`weight` and `logit_scale`)
        old_w = self.lm_head.weight
        old_scale = self.lm_head.logit_scale
        new_w = nn.Parameter(
            torch.empty(
                (new_num_tokens, hidden_dim),
                device=old_w.device,
                dtype=old_w.dtype,
            )
        )
        new_scale = nn.Parameter(
            torch.full(
                (new_num_tokens,),
                self.config.initializer_range,
                device=old_scale.device,
                dtype=old_scale.dtype,
            )
        )
        with torch.no_grad():
            new_w[:copy_count].copy_(old_w[:copy_count])
            new_scale[:copy_count].copy_(old_scale[:copy_count])
            if new_num_tokens > old_num_tokens:
                mean_w = old_w.float().mean(dim=0).to(old_w.dtype)
                mean_s = old_scale.float().mean().to(old_scale.dtype)
                new_w[old_num_tokens:].copy_(mean_w.unsqueeze(0).expand(new_num_tokens - old_num_tokens, -1))
                new_scale[old_num_tokens:].fill_(mean_s)

        self.lm_head.weight = new_w
        self.lm_head.logit_scale = new_scale
        self.config.vocab_size = new_num_tokens
        self.backbone.config.vocab_size = new_num_tokens
        self.lm_head.config.vocab_size = new_num_tokens
        return self.backbone.embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: DoryLoopCache | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if isinstance(past_key_values, DoryLoopCache) and past_key_values.seen_tokens > 0:
            input_ids = input_ids[:, -1:]
            inputs_embeds = None

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            if isinstance(past_key_values, DoryLoopCache) and past_key_values.seen_tokens > 0:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        model_inputs: dict[str, Any] = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            " position_ids".strip(): position_ids,
            "attention_mask": attention_mask,
        }
        if inputs_embeds is not None:
            model_inputs["inputs_embeds"] = inputs_embeds
        if "n_loops" in kwargs:
            model_inputs["n_loops"] = kwargs["n_loops"]
        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: DoryLoopCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        n_loops: int | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        hidden_states, present_kv, loop_states = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            n_loops=n_loops,
            collect_loop_states=bool(output_hidden_states),
        )
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        all_hidden = tuple(loop_states + [hidden_states]) if output_hidden_states else None
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=present_kv,  # type: ignore[arg-type]
            hidden_states=all_hidden,
            attentions=None,
        )


def register_dory_auto() -> None:
    """Register `DoryConfig` and `DoryForCausalLM` with HuggingFace `AutoConfig` and `AutoModelForCausalLM`."""
    if not _HAS_TORCH:
        return
    try:
        AutoConfig.register("dory", DoryConfig, exist_ok=True)
    except TypeError:
        try:
            AutoConfig.register("dory", DoryConfig)
        except ValueError:
            pass
    try:
        AutoModelForCausalLM.register(DoryConfig, DoryForCausalLM, exist_ok=True)
    except TypeError:
        try:
            AutoModelForCausalLM.register(DoryConfig, DoryForCausalLM)
        except ValueError:
            pass


if _HAS_TORCH:
    register_dory_auto()


def set_dory_loops(
    model: Any,
    n_loops: int,
    kv_cache_mode: Literal["per_loop", "last_loop"] | None = None,
) -> None:
    """Configure `n_recurrent_loops` (1, 2, or 4) on a Dory model (or PEFT-wrapped Dory model)."""
    if n_loops < 1:
        raise ValueError(f"n_loops must be >= 1, got {n_loops}")
    # Unwrap PEFT or DataParallel if needed
    target = getattr(model, "base_model", model)
    if _HAS_TORCH and hasattr(target, "model") and isinstance(getattr(target, "model"), PreTrainedModel):
        target = target.model
    if hasattr(target, "set_loops"):
        target.set_loops(n_loops, kv_cache_mode=kv_cache_mode)
    if hasattr(model, "config") and model.config is not None:
        model.config.n_recurrent_loops = int(n_loops)
        if kv_cache_mode is not None:
            model.config.kv_cache_mode = kv_cache_mode


def load_dory(
    model_name_or_path: str = DORY_HF_REPO,
    *,
    n_loops: int = 2,
    kv_cache_mode: Literal["per_loop", "last_loop"] = "per_loop",
    torch_dtype: Any = "bfloat16",
    device_map: str | dict[str, Any] | None = "auto",
    load_in_4bit: bool = False,
    **kwargs: Any,
) -> tuple[DoryForCausalLM, Any]:
    """Load `DoryForCausalLM` and its `AutoTokenizer` with runtime loop control."""
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch and transformers are required to load pretrained Dory weights.")
    if torch_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    register_dory_auto()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = dict(kwargs)
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    config = DoryConfig.from_pretrained(model_name_or_path)
    config.n_recurrent_loops = int(n_loops)
    config.kv_cache_mode = kv_cache_mode

    model = DoryForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        torch_dtype=torch_dtype,
        device_map=device_map,
        **load_kwargs,
    )
    model.eval()
    return model, tokenizer


@dataclass
class LoopLensEntry:
    loop_idx: int
    position_idx: int
    input_token: str
    top_tokens: list[str]
    top_token_ids: list[int]
    top_probs: list[float]


def logit_lens_per_loop(
    model: DoryForCausalLM,
    tokenizer: Any,
    prompt: str | torch.LongTensor,
    *,
    n_loops: int = 4,
    top_k: int = 5,
    last_n_positions: int = 5,
    apply_output_block: bool = False,
) -> list[dict[str, Any]]:
    """Project intermediate hidden states at the end of each recurrent loop (`1..n_loops`)
    onto `lm_head` to inspect how top-k token predictions evolve across loops for the last
    `last_n_positions` tokens of `prompt` (Section 2.2 of ESPEC_ANTIGRAVITY.md).

    Args:
        model: `DoryForCausalLM` instance.
        tokenizer: Tokenizer (or object with `.encode` / `.decode`).
        prompt: Input string or `[1, seq_len]` token IDs tensor.
        n_loops: Number of recurrent loops to run (default 4).
        top_k: Number of top tokens to extract per position and loop (default 5).
        last_n_positions: Number of trailing sequence positions to inspect (default 5).
        apply_output_block: If True, pass each loop's hidden state through `output_layers`
            before projecting onto `lm_head`; if False (classic Logit Lens), project the
            loop's output hidden state directly onto `lm_head`.

    Returns:
        A list of dicts (one per loop `1..n_loops`), each containing `loop_idx` and `positions`
        (list of per-position top-k token predictions and probabilities).
    """
    device = next(model.parameters()).device
    if isinstance(prompt, str):
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
    else:
        input_ids = prompt.to(device)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

    seq_len = int(input_ids.shape[1])
    n_pos = min(last_n_positions, seq_len)
    pos_indices = list(range(seq_len - n_pos, seq_len))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        _, _, loop_states = model.backbone(
            input_ids=input_ids,
            use_cache=False,
            n_loops=n_loops,
            collect_loop_states=True,
        )

        n_in = model.config.n_input_layers
        n_rec = model.config.n_recurrent_layers
        pos_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)

        results_per_loop: list[dict[str, Any]] = []
        for idx, state in enumerate(loop_states):
            h_proj = state
            if apply_output_block:
                for out_layer in model.backbone.layers[n_in + n_rec :]:
                    h_proj = out_layer(h_proj, position_ids=pos_ids, past_key_values=None)

            logits = model.lm_head(h_proj[:, pos_indices, :])  # [1, n_pos, vocab]
            probs = F.softmax(logits.float(), dim=-1)
            top_vals, top_ids = torch.topk(probs[0], k=min(top_k, probs.shape[-1]), dim=-1)

            positions_info: list[dict[str, Any]] = []
            for p_rel, p_abs in enumerate(pos_indices):
                in_tok_id = int(input_ids[0, p_abs].item())
                in_tok_str = (
                    tokenizer.decode([in_tok_id])
                    if hasattr(tokenizer, "decode")
                    else str(in_tok_id)
                )
                t_ids = [int(x) for x in top_ids[p_rel].tolist()]
                t_probs = [round(float(x), 5) for x in top_vals[p_rel].tolist()]
                t_strs = [
                    tokenizer.decode([tid]) if hasattr(tokenizer, "decode") else str(tid)
                    for tid in t_ids
                ]
                positions_info.append(
                    {
                        "position_idx": p_abs,
                        "input_token_id": in_tok_id,
                        "input_token": in_tok_str,
                        "top_tokens": t_strs,
                        "top_token_ids": t_ids,
                        "top_probs": t_probs,
                    }
                )

            results_per_loop.append(
                {
                    "loop_idx": idx + 1,
                    "positions": positions_info,
                }
            )

    if was_training:
        model.train()
    return results_per_loop


def init_new_tokens_embeddings(
    model: PreTrainedModel,
    tokenizer: Any,
    new_tokens: list[str] | dict[str, str],
) -> dict[str, int]:
    """Add custom tokens (e.g., DSL primitives or `<latent>`) to `tokenizer`, resize `model`
    embeddings with `tie_word_embeddings=False`, and initialize BOTH the input embedding
    (`embed_tokens` / `embeddings`) and the untied output head (`lm_head`) of each new token
    as the mean of the embeddings of its constituent subtokens (Section 4 F3 & Section 6).

    Args:
        model: A `PreTrainedModel` (such as `DoryForCausalLM`) with `tie_word_embeddings=False`.
        tokenizer: HuggingFace tokenizer (or compatible tokenizer with `add_tokens` & `encode`).
        new_tokens: Either a list of token strings (e.g. `["<FOR>", "<IF>"]`) where each token's
            semantic seed text is the token stripped of `<`/`>` (or the token string itself),
            or a dict mapping `{new_token_str: seed_description_text}`.

    Returns:
        Mapping `{new_token_str: assigned_token_id}`.
    """
    if isinstance(new_tokens, dict):
        token_to_seed = dict(new_tokens)
        token_list = list(token_to_seed.keys())
    else:
        token_list = list(new_tokens)
        token_to_seed = {}
        for tok in token_list:
            stripped = tok.strip("<>").replace("_", " ").strip().lower()
            token_to_seed[tok] = stripped if stripped else tok

    old_vocab_size = len(tokenizer)

    # Pre-tokenize seeds BEFORE adding new tokens so they decompose into existing subtokens
    seed_subtoken_ids: dict[str, list[int]] = {}
    for tok, seed_text in token_to_seed.items():
        ids = tokenizer.encode(seed_text, add_special_tokens=False)
        valid_ids = [i for i in ids if 0 <= i < old_vocab_size]
        seed_subtoken_ids[tok] = valid_ids

    tokenizer.add_tokens(token_list, special_tokens=False)
    new_vocab_size = len(tokenizer)
    model.resize_token_embeddings(new_vocab_size)

    input_emb = model.get_input_embeddings()
    output_emb = model.get_output_embeddings()

    assigned_ids: dict[str, int] = {}
    with torch.no_grad():
        in_w = input_emb.weight
        out_w = output_emb.weight if output_emb is not None and hasattr(output_emb, "weight") else None
        out_scale = (
            output_emb.logit_scale
            if output_emb is not None and hasattr(output_emb, "logit_scale")
            else None
        )

        # Fallback mean across old vocabulary
        global_in_mean = in_w[:old_vocab_size].float().mean(dim=0).to(in_w.dtype)
        global_out_mean = (
            out_w[:old_vocab_size].float().mean(dim=0).to(out_w.dtype)
            if out_w is not None
            else None
        )
        global_scale_mean = (
            out_scale[:old_vocab_size].float().mean().to(out_scale.dtype)
            if out_scale is not None
            else None
        )

        for tok in token_list:
            tok_id = tokenizer.convert_tokens_to_ids(tok)
            assigned_ids[tok] = int(tok_id)
            sub_ids = seed_subtoken_ids.get(tok, [])
            if sub_ids:
                idx_tensor = torch.tensor(sub_ids, dtype=torch.long, device=in_w.device)
                in_vec = in_w.index_select(0, idx_tensor).float().mean(dim=0).to(in_w.dtype)
                in_w[tok_id].copy_(in_vec)
                if out_w is not None:
                    idx_out = idx_tensor.to(out_w.device)
                    out_vec = out_w.index_select(0, idx_out).float().mean(dim=0).to(out_w.dtype)
                    out_w[tok_id].copy_(out_vec)
                if out_scale is not None:
                    idx_s = idx_tensor.to(out_scale.device)
                    s_val = out_scale.index_select(0, idx_s).float().mean().to(out_scale.dtype)
                    out_scale[tok_id].copy_(s_val)
            else:
                in_w[tok_id].copy_(global_in_mean)
                if out_w is not None and global_out_mean is not None:
                    out_w[tok_id].copy_(global_out_mean)
                if out_scale is not None and global_scale_mean is not None:
                    out_scale[tok_id].copy_(global_scale_mean)

    return assigned_ids


# ==============================================================================
# Pure-Python fallback when PyTorch is not installed in a bare local environment
# ==============================================================================
if not _HAS_TORCH:  # pragma: no cover
    import random

    def _vec_norm(v: list[float], eps: float = 1e-6) -> list[float]:
        s = math.sqrt(sum(x * x for x in v) + eps)
        return [x / s for x in v]

    def _vec_slerp(a: list[float], b: list[float], lr: float) -> list[float]:
        na = _vec_norm(a)
        nb = _vec_norm(b)
        return _vec_norm([x + abs(lr) * (y - x) for x, y in zip(na, nb)])

    class _PureMatrix:
        def __init__(self, rows: list[list[float]]) -> None:
            self.rows = [list(r) for r in rows]

        @property
        def shape(self) -> tuple[int, int]:
            return (len(self.rows), len(self.rows[0]) if self.rows else 0)

        def data_ptr(self) -> int:
            return id(self.rows)

        def clone(self) -> "_PureMatrix":
            return _PureMatrix([list(r) for r in self.rows])

        def detach(self) -> "_PureMatrix":
            return self

        def float(self) -> "_PureMatrix":
            return self

        def __getitem__(self, idx: int | list[int]) -> Any:
            if isinstance(idx, list):
                return _PureMatrix([self.rows[i] for i in idx])
            return self.rows[idx]

        def __setitem__(self, idx: int, val: list[float]) -> None:
            self.rows[idx] = list(val)

        def mean_axis0(self) -> list[float]:
            if not self.rows:
                return []
            cols = len(self.rows[0])
            n = float(len(self.rows))
            return [sum(r[c] for r in self.rows) / n for c in range(cols)]

    class _PureEmbedding:
        def __init__(self, vocab_size: int, hidden_size: int, seed: int = 42) -> None:
            rng = random.Random(seed)
            self.weight = _PureMatrix(
                [
                    _vec_norm([rng.gauss(0.0, 0.02) for _ in range(hidden_size)])
                    for _ in range(vocab_size)
                ]
            )

    class _PureOutput:
        def __init__(self, logits: list[list[list[float]]]) -> None:
            self.logits = logits

    class DoryModel:  # type: ignore[no-redef]
        def __init__(self, config: DoryConfig) -> None:
            self.config = config
            self.embeddings = _PureEmbedding(config.vocab_size, config.hidden_size, seed=101)
            rng = random.Random(202)
            self._layer_Mix: list[list[float]] = [
                [rng.uniform(-0.3, 0.3) for _ in range(config.hidden_size)]
                for _ in range(config.num_hidden_layers)
            ]

        @property
        def embed_tokens(self) -> _PureEmbedding:
            return self.embeddings

        def forward(
            self,
            input_ids: list[list[int]],
            n_loops: int | None = None,
            collect_loop_states: bool = False,
            **kwargs: Any,
        ) -> tuple[list[list[list[float]]], None, list[list[list[list[float]]]]]:
            active_loops = int(n_loops if n_loops is not None else self.config.n_recurrent_loops)
            n_in = self.config.n_input_layers
            n_rec = self.config.n_recurrent_layers

            batch_h: list[list[list[float]]] = [
                [list(self.embeddings.weight[tok]) for tok in seq] for seq in input_ids
            ]

            def _run_sublayer(states: list[list[list[float]]], l_idx: int, loop_i: int) -> list[list[list[float]]]:
                w = self._layer_Mix[l_idx]
                out_b: list[list[list[float]]] = []
                for seq in states:
                    out_s: list[list[float]] = []
                    for pos_i, vec in enumerate(seq):
                        mixed = [
                            math.sin(v * (1.0 + w[d]) + 0.1 * (loop_i + 1) + 0.03 * pos_i)
                            for d, v in enumerate(vec)
                        ]
                        out_s.append(_vec_slerp(vec, mixed, 0.25))
                    out_b.append(out_s)
                return out_b

            for l_idx in range(n_in):
                batch_h = _run_sublayer(batch_h, l_idx, 0)

            input_h = [[list(v) for v in seq] for seq in batch_h]
            state = [[list(v) for v in seq] for seq in batch_h]
            loop_states: list[list[list[list[float]]]] = []

            for loop_i in range(active_loops):
                batch_h = [
                    [_vec_norm([a + b for a, b in zip(sv, iv)]) for sv, iv in zip(s_seq, i_seq)]
                    for s_seq, i_seq in zip(state, input_h)
                ]
                for l_idx in range(n_in, n_in + n_rec):
                    batch_h = _run_sublayer(batch_h, l_idx, loop_i)
                state = [[_vec_norm(v) for v in seq] for seq in batch_h]
                batch_h = state
                if collect_loop_states:
                    loop_states.append([[list(v) for v in seq] for seq in state])

            for l_idx in range(n_in + n_rec, self.config.num_hidden_layers):
                batch_h = _run_sublayer(batch_h, l_idx, 0)

            return batch_h, None, loop_states

    class DoryForCausalLM:  # type: ignore[no-redef]
        def __init__(self, config: DoryConfig) -> None:
            self.config = config
            self.backbone = DoryModel(config)
            self.lm_head = _PureEmbedding(config.vocab_size, config.hidden_size, seed=303)
            self.training = False

        def eval(self) -> "DoryForCausalLM":
            self.training = False
            return self

        def train(self, mode: bool = True) -> "DoryForCausalLM":
            self.training = mode
            return self

        def get_input_embeddings(self) -> _PureEmbedding:
            return self.backbone.embeddings

        def get_output_embeddings(self) -> _PureEmbedding:
            return self.lm_head

        def set_loops(
            self,
            n_loops: int,
            kv_cache_mode: Literal["per_loop", "last_loop"] | None = None,
        ) -> None:
            if n_loops < 1:
                raise ValueError(f"n_loops must be >= 1, got {n_loops}")
            self.config.n_recurrent_loops = int(n_loops)
            self.backbone.config.n_recurrent_loops = int(n_loops)
            if kv_cache_mode is not None:
                self.config.kv_cache_mode = kv_cache_mode
                self.backbone.config.kv_cache_mode = kv_cache_mode

        def resize_token_embeddings(self, new_num_tokens: int) -> _PureEmbedding:
            old_n, dim = self.backbone.embeddings.weight.shape
            if new_num_tokens > old_n:
                in_mean = self.backbone.embeddings.weight.mean_axis0()
                out_mean = self.lm_head.weight.mean_axis0()
                for _ in range(new_num_tokens - old_n):
                    self.backbone.embeddings.weight.rows.append(list(in_mean))
                    self.lm_head.weight.rows.append(list(out_mean))
            self.config.vocab_size = new_num_tokens
            self.backbone.config.vocab_size = new_num_tokens
            return self.backbone.embeddings

        def _project_logits(self, hidden: list[list[list[float]]]) -> list[list[list[float]]]:
            head_rows = self.lm_head.weight.rows
            return [
                [
                    [sum(h_d * w_d for h_d, w_d in zip(vec, w_row)) for w_row in head_rows]
                    for vec in seq
                ]
                for seq in hidden
            ]

        def __call__(self, input_ids: Any, n_loops: int | None = None, **kwargs: Any) -> _PureOutput:
            h, _, _ = self.backbone.forward(input_ids=input_ids, n_loops=n_loops)
            return _PureOutput(self._project_logits(h))

    def logit_lens_per_loop(  # type: ignore[no-redef]
        model: DoryForCausalLM,
        tokenizer: Any,
        prompt: str | list[list[int]],
        *,
        n_loops: int = 4,
        top_k: int = 5,
        last_n_positions: int = 5,
        apply_output_block: bool = False,
    ) -> list[dict[str, Any]]:
        if isinstance(prompt, str):
            enc = tokenizer(prompt)
            input_ids = enc["input_ids"]
            if hasattr(input_ids, "tolist"):
                input_ids = input_ids.tolist()
        else:
            input_ids = prompt

        seq = input_ids[0]
        seq_len = len(seq)
        n_pos = min(last_n_positions, seq_len)
        pos_indices = list(range(seq_len - n_pos, seq_len))

        _, _, loop_states = model.backbone.forward(
            input_ids=input_ids, n_loops=n_loops, collect_loop_states=True
        )
        results_per_loop: list[dict[str, Any]] = []
        for idx, state in enumerate(loop_states):
            logits_b = model._project_logits(state)[0]
            positions_info: list[dict[str, Any]] = []
            for p_abs in pos_indices:
                row_logits = logits_b[p_abs]
                max_l = max(row_logits)
                exps = [math.exp(x - max_l) for x in row_logits]
                denom = sum(exps)
                probs = [x / denom for x in exps]
                ranked = sorted(enumerate(probs), key=lambda item: item[1], reverse=True)[:top_k]
                in_id = int(seq[p_abs])
                positions_info.append(
                    {
                        "position_idx": p_abs,
                        "input_token_id": in_id,
                        "input_token": tokenizer.decode([in_id]),
                        "top_tokens": [tokenizer.decode([tid]) for tid, _ in ranked],
                        "top_token_ids": [tid for tid, _ in ranked],
                        "top_probs": [round(p, 5) for _, p in ranked],
                    }
                )
            results_per_loop.append({"loop_idx": idx + 1, "positions": positions_info})
        return results_per_loop

    def init_new_tokens_embeddings(  # type: ignore[no-redef]
        model: Any,
        tokenizer: Any,
        new_tokens: list[str] | dict[str, str],
    ) -> dict[str, int]:
        if isinstance(new_tokens, dict):
            token_to_seed = dict(new_tokens)
            token_list = list(token_to_seed.keys())
        else:
            token_list = list(new_tokens)
            token_to_seed = {
                t: (t.strip("<>").replace("_", " ").strip().lower() or t) for t in token_list
            }

        old_vocab_size = len(tokenizer)
        seed_subtoken_ids: dict[str, list[int]] = {}
        for tok, seed_text in token_to_seed.items():
            ids = tokenizer.encode(seed_text, add_special_tokens=False)
            seed_subtoken_ids[tok] = [i for i in ids if 0 <= i < old_vocab_size]

        tokenizer.add_tokens(token_list, special_tokens=False)
        model.resize_token_embeddings(len(tokenizer))

        in_w = model.get_input_embeddings().weight
        out_w = model.get_output_embeddings().weight
        assigned: dict[str, int] = {}
        for tok in token_list:
            tok_id = tokenizer.convert_tokens_to_ids(tok)
            assigned[tok] = int(tok_id)
            sub_ids = seed_subtoken_ids.get(tok, [])
            if sub_ids:
                in_w[tok_id] = in_w[sub_ids].mean_axis0()
                out_w[tok_id] = out_w[sub_ids].mean_axis0()
        return assigned


DEFAULT_EQUIVALENCE_PROMPTS: tuple[str, ...] = (
    "Write a Python function `fibonacci(n)` that returns the n-th Fibonacci number (0-indexed).",
    "Write a Python function `two_sum(nums, target)` that returns the indices of two numbers adding up to `target`.",
    "Write a Python function `is_palindrome(s)` that checks whether string `s` is a palindrome ignoring non-alphanumeric characters.",
    "Write a Python function `binary_search(arr, x)` that returns the index of `x` in sorted list `arr` or -1 if not present.",
    "Write a Python function `merge_intervals(intervals)` that merges all overlapping intervals in a list of `[start, end]` pairs.",
)


def compare_logprob_sequences(
    vllm_steps: list[dict[str, Any]],
    hf_steps: list[dict[str, Any]],
    *,
    prompt_idx: int,
    n_loops: int,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Compare per-position top-1 token IDs and logprobs between vLLM and our HF reimplementation.

    Each element in `vllm_steps` and `hf_steps` is a dict:
        `{"top1_id": int, "top1_logprob": float, "logprobs_map": dict[int, float] (optional)}`.
    """
    n_positions = min(len(vllm_steps), len(hf_steps))
    divergences: list[dict[str, Any]] = []
    abs_diffs: list[float] = []

    for step_idx in range(n_positions):
        v_step = vllm_steps[step_idx]
        h_step = hf_steps[step_idx]
        v_id = int(v_step["top1_id"])
        h_id = int(h_step["top1_id"])
        v_lp = float(v_step["top1_logprob"])
        h_lp = float(h_step["top1_logprob"])

        # Compare logprob on the same (vLLM top-1) token if available in hf_step['logprobs_map']
        h_lp_on_v = float(h_step.get("logprobs_map", {}).get(v_id, h_lp))
        diff = abs(v_lp - h_lp_on_v)
        abs_diffs.append(diff)

        if v_id != h_id:
            v_tok = tokenizer.decode([v_id]) if (tokenizer and hasattr(tokenizer, "decode")) else str(v_id)
            h_tok = tokenizer.decode([h_id]) if (tokenizer and hasattr(tokenizer, "decode")) else str(h_id)
            divergences.append(
                {
                    "n_loops": n_loops,
                    "prompt_idx": prompt_idx,
                    "step_idx": step_idx,
                    "vllm_top1_id": v_id,
                    "vllm_top1_token": v_tok,
                    "vllm_logprob": round(v_lp, 6),
                    "hf_top1_id": h_id,
                    "hf_top1_token": h_tok,
                    "hf_logprob": round(h_lp, 6),
                    "hf_logprob_on_vllm_top1": round(h_lp_on_v, 6),
                    "abs_logprob_diff": round(diff, 6),
                }
            )

    return {
        "n_loops": n_loops,
        "prompt_idx": prompt_idx,
        "n_positions": n_positions,
        "n_divergences": len(divergences),
        "divergence_rate": (len(divergences) / float(n_positions)) if n_positions > 0 else 0.0,
        "mean_abs_logprob_diff": (sum(abs_diffs) / len(abs_diffs)) if abs_diffs else 0.0,
        "max_abs_logprob_diff": max(abs_diffs) if abs_diffs else 0.0,
        "divergences": divergences,
    }


def verify_vllm_hf_equivalence(
    prompts: list[str] | tuple[str, ...] = DEFAULT_EQUIVALENCE_PROMPTS,
    *,
    loops: tuple[int, ...] = (1, 2, 4),
    max_new_tokens: int = 32,
    max_divergence_rate: float = 0.02,
    model_name_or_path: str = DORY_HF_REPO,
    hf_model: Any | None = None,
    tokenizer: Any | None = None,
    vllm_runner: Any | None = None,
    hf_runner: Any | None = None,
    output_json_path: str | None = None,
    halt_on_failure: bool = True,
) -> dict[str, Any]:
    """Mandatory Equivalence Test between official vLLM PR #60149 and our HF `DoryForCausalLM`.

    Runs `len(prompts)` (default 5) prompts with `loops` (default 1, 2, and 4) for `max_new_tokens`
    (default 32) greedy tokens, comparing the logprobs and top-1 predicted tokens at every position
    under the exact same input prefix (teacher-forced along the vLLM greedy trajectory) as well as
    free-running rollout agreement.

    If top-1 diverges in more than `max_divergence_rate` (~2%) of positions, prints the exact
    divergent positions and raises `RuntimeError` (when `halt_on_failure=True`).
    """
    import gc
    import json
    from pathlib import Path

    prompt_list = list(prompts)
    if tokenizer is None and _HAS_TORCH and vllm_runner is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Format prompts with chat template if tokenizer is available
    formatted_prompts: list[str] = []
    for p in prompt_list:
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            try:
                formatted_prompts.append(
                    str(
                        tokenizer.apply_chat_template(
                            [{"role": "user", "content": p}],
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=True,
                        )
                    )
                )
                continue
            except Exception:
                pass
        formatted_prompts.append(p)

    # Step 1: Collect vLLM trajectories & logprobs for each loop count k in `loops`
    # vllm_data[k] -> list (per prompt) of {"prompt_token_ids": [...], "steps": [{"top1_id", "top1_logprob"}, ...]}
    vllm_data: dict[int, list[dict[str, Any]]] = {}

    if vllm_runner is not None:
        for k in loops:
            vllm_data[int(k)] = vllm_runner(formatted_prompts, int(k), int(max_new_tokens))
    else:
        from vllm import LLM, SamplingParams  # type: ignore[import-not-found]

        # If hf_model is currently on GPU, move it to CPU temporarily so vLLM can allocate KV cache
        hf_orig_device = None
        if hf_model is not None and _HAS_TORCH and hasattr(hf_model, "parameters"):
            try:
                hf_orig_device = next(hf_model.parameters()).device
                if hf_orig_device.type == "cuda":
                    hf_model.to("cpu")
                    torch.cuda.empty_cache()
            except Exception:
                pass

        for k in loops:
            print(f"[Equivalence] Rodando vLLM oficial com n_recurrent_loops={k} em {len(formatted_prompts)} prompts...")
            llm = LLM(
                model=model_name_or_path,
                trust_remote_code=True,
                dtype="bfloat16",
                hf_overrides={"n_recurrent_loops": int(k)},
                gpu_memory_utilization=0.80,
                max_model_len=4096,
                seed=42,
            )
            sp = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=int(max_new_tokens),
                logprobs=5,
                seed=42,
            )
            outs = llm.generate(formatted_prompts, sp)
            loop_items: list[dict[str, Any]] = []
            for req_out in outs:
                p_ids = list(req_out.prompt_token_ids)
                comp = req_out.outputs[0]
                gen_ids = list(comp.token_ids)
                step_list: list[dict[str, Any]] = []
                for s_idx, tok_id in enumerate(gen_ids):
                    lp_dict = comp.logprobs[s_idx] if (comp.logprobs and s_idx < len(comp.logprobs)) else {}
                    if lp_dict and tok_id in lp_dict:
                        raw_lp = lp_dict[tok_id]
                        lp_val = float(raw_lp.logprob if hasattr(raw_lp, "logprob") else raw_lp)
                    else:
                        lp_val = 0.0
                    step_list.append(
                        {
                            "top1_id": int(tok_id),
                            "top1_logprob": lp_val,
                        }
                    )
                loop_items.append(
                    {
                        "prompt_token_ids": p_ids,
                        "generated_token_ids": gen_ids,
                        "steps": step_list,
                    }
                )
            vllm_data[int(k)] = loop_items

            # Clean up vLLM instance before next loop count
            del llm
            gc.collect()
            if _HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()

        if hf_model is not None and hf_orig_device is not None and hf_orig_device.type == "cuda":
            hf_model.to(hf_orig_device)

    # Step 2: Collect HF reimplementation logprobs on the exact same input sequence for each loop k
    hf_data: dict[int, list[list[dict[str, Any]]]] = {}
    if hf_runner is not None:
        for k in loops:
            hf_data[int(k)] = hf_runner(formatted_prompts, vllm_data[int(k)], int(k), int(max_new_tokens))
    else:
        own_hf_model = False
        if hf_model is None:
            hf_model, tokenizer = load_dory(
                model_name_or_path,
                n_loops=2,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            own_hf_model = True

        device = next(hf_model.parameters()).device
        hf_model.eval()

        for k in loops:
            set_dory_loops(hf_model, int(k))
            loop_hf_steps: list[list[dict[str, Any]]] = []
            for p_idx, v_item in enumerate(vllm_data[int(k)]):
                p_ids = v_item["prompt_token_ids"]
                gen_ids = v_item["generated_token_ids"]
                n_gen = len(gen_ids)
                # Teacher-forced full sequence: prompt_ids + gen_ids[:-1] predicts gen_ids[0..n_gen-1]
                full_ids = p_ids + gen_ids[:-1]
                inp = torch.tensor([full_ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    out = hf_model(input_ids=inp, use_cache=False, n_loops=int(k))
                    # Slice logits corresponding to the last token of prompt through the (n_gen-1)-th generated token
                    start_pos = len(p_ids) - 1
                    step_logits = out.logits[0, start_pos : start_pos + n_gen, :].float()
                    step_logprobs = F.log_softmax(step_logits, dim=-1)
                    top1_vals, top1_ids = torch.max(step_logprobs, dim=-1)

                p_steps: list[dict[str, Any]] = []
                for s_idx in range(n_gen):
                    v_tok_id = int(gen_ids[s_idx])
                    h_id = int(top1_ids[s_idx].item())
                    h_lp = float(top1_vals[s_idx].item())
                    h_lp_on_v = float(step_logprobs[s_idx, v_tok_id].item())
                    p_steps.append(
                        {
                            "top1_id": h_id,
                            "top1_logprob": h_lp,
                            "logprobs_map": {v_tok_id: h_lp_on_v, h_id: h_lp},
                        }
                    )
                loop_hf_steps.append(p_steps)
            hf_data[int(k)] = loop_hf_steps

        if own_hf_model:
            del hf_model
            gc.collect()
            if _HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Step 3: Aggregate comparisons across all loops and prompts
    per_loop_summary: dict[int, dict[str, Any]] = {}
    all_divergences: list[dict[str, Any]] = []
    total_positions = 0
    total_divergences = 0
    all_abs_diffs: list[float] = []

    for k in loops:
        k_int = int(k)
        k_pos = 0
        k_div = 0
        k_diffs: list[float] = []
        k_divergences: list[dict[str, Any]] = []

        for p_idx in range(len(formatted_prompts)):
            v_steps = vllm_data[k_int][p_idx]["steps"]
            h_steps = hf_data[k_int][p_idx]
            cmp_res = compare_logprob_sequences(
                v_steps,
                h_steps,
                prompt_idx=p_idx,
                n_loops=k_int,
                tokenizer=tokenizer,
            )
            k_pos += cmp_res["n_positions"]
            k_div += cmp_res["n_divergences"]
            if cmp_res["n_positions"] > 0:
                k_diffs.append(cmp_res["mean_abs_logprob_diff"])
            k_divergences.extend(cmp_res["divergences"])
            all_divergences.extend(cmp_res["divergences"])

        k_rate = (k_div / float(k_pos)) if k_pos > 0 else 0.0
        k_mean_diff = (sum(k_diffs) / len(k_diffs)) if k_diffs else 0.0
        per_loop_summary[k_int] = {
            "n_loops": k_int,
            "n_positions": k_pos,
            "n_divergences": k_div,
            "divergence_rate": round(k_rate, 6),
            "mean_abs_logprob_diff": round(k_mean_diff, 6),
            "divergences": k_divergences,
        }
        total_positions += k_pos
        total_divergences += k_div
        all_abs_diffs.extend(k_diffs)

    overall_rate = (total_divergences / float(total_positions)) if total_positions > 0 else 0.0
    overall_mean_diff = (sum(all_abs_diffs) / len(all_abs_diffs)) if all_abs_diffs else 0.0
    # Fails if overall rate > max_divergence_rate OR any individual loop count exceeds max_divergence_rate
    passed = (overall_rate <= max_divergence_rate) and all(
        info["divergence_rate"] <= max_divergence_rate for info in per_loop_summary.values()
    )

    report = {
        "passed": passed,
        "max_divergence_rate_threshold": max_divergence_rate,
        "total_positions": total_positions,
        "total_divergences": total_divergences,
        "overall_divergence_rate": round(overall_rate, 6),
        "overall_mean_abs_logprob_diff": round(overall_mean_diff, 6),
        "per_loop": per_loop_summary,
        "all_divergences": all_divergences,
    }

    if output_json_path is not None:
        out_p = Path(output_json_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    status_tag = "APROVADO" if passed else "FALHOU"
    print(
        f"[Equivalence {status_tag}] Divergência Top-1 global: {total_divergences}/{total_positions} "
        f"({overall_rate * 100:.2f}%, limite <= {max_divergence_rate * 100:.2f}%) | "
        f"|Δ logprob| médio: {overall_mean_diff:.5f}"
    )
    for k_int, info in per_loop_summary.items():
        print(
            f"  - n_loops={k_int}: {info['n_divergences']}/{info['n_positions']} "
            f"({info['divergence_rate'] * 100:.2f}%) | |Δ logprob| médio={info['mean_abs_logprob_diff']:.5f}"
        )

    if all_divergences:
        print("\nPosições com divergência de Top-1 (primeiras 20):")
        for d in all_divergences[:20]:
            print(
                f"  [loop={d['n_loops']} | prompt={d['prompt_idx']} | pos={d['step_idx']:2d}] "
                f"vLLM={d['vllm_top1_id']} ({repr(d['vllm_top1_token'])}, lp={d['vllm_logprob']:.4f}) vs "
                f"HF={d['hf_top1_id']} ({repr(d['hf_top1_token'])}, lp={d['hf_logprob']:.4f}) | "
                f"|Δlp|={d['abs_logprob_diff']:.4f}"
            )

    if not passed and halt_on_failure:
        raise RuntimeError(
            f"Teste de equivalência vLLM vs HF reprovado! Divergência Top-1 = {overall_rate * 100:.2f}% "
            f"(limite = {max_divergence_rate * 100:.2f}%). Verifique as posições listadas acima."
        )

    return report


