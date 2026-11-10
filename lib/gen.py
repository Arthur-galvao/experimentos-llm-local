"""Critical Module: Unified generation wrapper, CoT/answer parsing, and token counting (Section 2.3)."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Literal

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from .dory import DORY_HF_REPO, QWEN_CONTROL_HF_REPO, set_dory_loops

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
DORY_VLLM_OVERRIDE_KEYS: tuple[str, str] = ("n_recurrent_loops", "recurrent_kv_cache_mode")


def build_dory_hf_overrides(
    n_loops: int = 2,
    recurrent_kv_cache_mode: Literal["per_loop", "last_loop"] = "per_loop",
) -> dict[str, Any]:
    """Build the exact `hf_overrides` dict matching Dory's `config.json` fields for vLLM PR #60149."""
    if n_loops < 1:
        raise ValueError(f"n_recurrent_loops must be >= 1, got {n_loops}")
    if recurrent_kv_cache_mode not in ("per_loop", "last_loop"):
        raise ValueError(f"Invalid recurrent_kv_cache_mode: {recurrent_kv_cache_mode}")
    return {
        "n_recurrent_loops": int(n_loops),
        "recurrent_kv_cache_mode": str(recurrent_kv_cache_mode),
    }


@dataclass
class Output:
    """Standardized generation output for all phases and evaluations."""

    text: str
    cot_text: str
    answer_text: str
    n_cot_tokens: int
    n_total_tokens: int
    latency_s: float
    backend: Literal["hf", "vllm"] = "hf"


def split_cot_and_answer(
    generated_text: str,
    *,
    prompt_ended_with_think: bool = False,
    open_tag: str = THINK_OPEN,
    close_tag: str = THINK_CLOSE,
) -> tuple[str, str]:
    """Split `generated_text` into `(cot_text, answer_text)` using `<think>` / `</think>` delimiters.

    Handles all three common cases:
    1. `<think>...reasoning...</think>...answer...` present in `generated_text`.
    2. `...reasoning...</think>...answer...` when `<think>` was already appended by the chat template
       in the prompt (`prompt_ended_with_think=True` or `</think>` appears without `<think>`).
    3. Unclosed `<think>...reasoning...` (generation truncated at `max_new_tokens` before `</think>`):
       all text after `<think>` is treated as `cot_text` and `answer_text` is `""`.
    4. Neither `<think>` nor `</think>` present and `prompt_ended_with_think=False`:
       `cot_text` is `""` and `answer_text` is `generated_text.strip()`.
    """
    text = generated_text or ""
    open_idx = text.find(open_tag)
    close_idx = text.find(close_tag)

    if open_idx != -1 and close_idx != -1 and open_idx < close_idx:
        cot = text[open_idx + len(open_tag) : close_idx].strip()
        answer = text[close_idx + len(close_tag) :].strip()
        return cot, answer

    if close_idx != -1 and (open_idx == -1 or close_idx < open_idx):
        # `<think>` was part of the prompt template before generation started
        cot = text[:close_idx].strip()
        answer = text[close_idx + len(close_tag) :].strip()
        return cot, answer

    if open_idx != -1 and close_idx == -1:
        # Unclosed `<think>` tag (hit max_new_tokens during CoT)
        cot = text[open_idx + len(open_tag) :].strip()
        return cot, ""

    if prompt_ended_with_think:
        # Prompt ended with `<think>` and model never closed `</think>`
        return text.strip(), ""

    return "", text.strip()


def _count_tokens(text: str, tokenizer: Any | None) -> int:
    """Count tokens using `tokenizer.encode(text, add_special_tokens=False)` or whitespace fallback."""
    if not text:
        return 0
    if tokenizer is not None and hasattr(tokenizer, "encode"):
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return len(tokenizer.encode(text))
    # Deterministic fallback if no tokenizer is passed: split into words/punctuation
    return len(re.findall(r"\w+|[^\w\s]", text))


def _find_subsequence(seq: list[int], subseq: list[int], start: int = 0) -> int:
    """Return the first index `>= start` where `subseq` appears in `seq`, or -1."""
    if not subseq or len(subseq) > len(seq) - start:
        return -1
    limit = len(seq) - len(subseq) + 1
    for i in range(start, limit):
        if seq[i : i + len(subseq)] == subseq:
            return i
    return -1


def _resolve_tag_token_ids(
    tag: str,
    tokenizer: Any | None,
    explicit_id: int | None = None,
) -> list[int]:
    """Resolve the token ID sequence for a delimiter tag (`<think>` or `</think>`)."""
    if explicit_id is not None:
        return [int(explicit_id)]
    if tokenizer is None:
        return []
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        try:
            tid = tokenizer.convert_tokens_to_ids(tag)
            unk_id = getattr(tokenizer, "unk_token_id", None)
            if isinstance(tid, int) and tid >= 0 and (unk_id is None or tid != unk_id):
                return [tid]
        except Exception:
            pass
    if hasattr(tokenizer, "encode"):
        try:
            ids = tokenizer.encode(tag, add_special_tokens=False)
        except TypeError:
            ids = tokenizer.encode(tag)
        if isinstance(ids, list) and ids:
            return [int(x) for x in ids]
    return []


def _count_cot_from_token_ids(
    token_ids: list[int],
    *,
    tokenizer: Any | None = None,
    generated_text: str = "",
    prompt_ended_with_think: bool = False,
    think_open_token_id: int | None = None,
    think_close_token_id: int | None = None,
) -> int:
    """Count CoT tokens strictly from generated `token_ids` up to the `</think>` token,
    never by character ratio.
    """
    ids = [int(t) for t in token_ids]
    if not ids:
        return 0

    open_ids = _resolve_tag_token_ids(THINK_OPEN, tokenizer, think_open_token_id)
    close_ids = _resolve_tag_token_ids(THINK_CLOSE, tokenizer, think_close_token_id)

    open_pos = _find_subsequence(ids, open_ids, start=0) if open_ids else -1
    search_start = (open_pos + len(open_ids)) if open_pos != -1 else 0
    close_pos = _find_subsequence(ids, close_ids, start=search_start) if close_ids else -1

    if open_pos != -1 and close_pos != -1:
        return max(0, close_pos - (open_pos + len(open_ids)))
    if close_pos != -1:
        return max(0, close_pos)
    if open_pos != -1 and close_pos == -1:
        return max(0, len(ids) - (open_pos + len(open_ids)))

    # If BPE fused `</think>` with whitespace/newlines, locate the exact token index via prefix decode
    if tokenizer is not None and hasattr(tokenizer, "decode") and (
        THINK_CLOSE in generated_text or THINK_OPEN in generated_text
    ):
        decoded_open_idx = -1
        decoded_close_idx = -1
        for idx in range(len(ids)):
            prefix_str = tokenizer.decode(ids[: idx + 1], skip_special_tokens=False)
            if decoded_open_idx == -1 and THINK_OPEN in prefix_str:
                decoded_open_idx = idx
            if decoded_close_idx == -1 and THINK_CLOSE in prefix_str:
                decoded_close_idx = idx
                break
        if decoded_open_idx != -1 and decoded_close_idx != -1 and decoded_open_idx < decoded_close_idx:
            return max(0, decoded_close_idx - (decoded_open_idx + 1))
        if decoded_close_idx != -1:
            return max(0, decoded_close_idx)
        if decoded_open_idx != -1:
            return max(0, len(ids) - (decoded_open_idx + 1))

    if prompt_ended_with_think and THINK_CLOSE not in generated_text:
        return len(ids)
    return 0


def count_cot_tokens(
    text_or_cot: str | list[int] = "",
    tokenizer: Any | None = None,
    *,
    token_ids: list[int] | None = None,
    think_open_token_id: int | None = None,
    think_close_token_id: int | None = None,
    is_full_generation: bool = False,
    prompt_ended_with_think: bool = False,
) -> int:
    """Count the number of tokens in the Chain-of-Thought (CoT) segment.

    When `token_ids` are provided (or `text_or_cot` is a `list[int]`), counts CoT tokens
    strictly from the generated `token_ids` up to the `</think>` token (excluding `<think>`
    and `</think>` delimiters), never by character proportion.
    """
    if isinstance(text_or_cot, list):
        return _count_cot_from_token_ids(
            text_or_cot,
            tokenizer=tokenizer,
            generated_text="",
            prompt_ended_with_think=prompt_ended_with_think,
            think_open_token_id=think_open_token_id,
            think_close_token_id=think_close_token_id,
        )

    if token_ids is not None:
        return _count_cot_from_token_ids(
            token_ids,
            tokenizer=tokenizer,
            generated_text=text_or_cot,
            prompt_ended_with_think=prompt_ended_with_think,
            think_open_token_id=think_open_token_id,
            think_close_token_id=think_close_token_id,
        )

    if is_full_generation or (THINK_OPEN in text_or_cot) or (THINK_CLOSE in text_or_cot):
        cot_text, _ = split_cot_and_answer(
            text_or_cot, prompt_ended_with_think=prompt_ended_with_think
        )
    else:
        cot_text = text_or_cot
    return _count_tokens(cot_text, tokenizer)


def format_chat_prompt(
    tokenizer: Any,
    user_prompt: str,
    *,
    system_prompt: str | None = None,
    enable_thinking: bool = True,
) -> str:
    """Format a user prompt using the model's `apply_chat_template` if available."""
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        try:
            return str(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            )
        except TypeError:
            return str(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
    if system_prompt:
        return f"{system_prompt}\n\n{user_prompt}\n{THINK_OPEN}\n" if enable_thinking else f"{system_prompt}\n\n{user_prompt}\n"
    return f"{user_prompt}\n{THINK_OPEN}\n" if enable_thinking else f"{user_prompt}\n"


def _generate_hf(
    model: Any,
    prompts: list[str],
    *,
    tokenizer: Any,
    max_new_tokens: int,
    temperature: float,
    n_loops: int | None,
    n_samples: int,
    batch_size: int,
    top_p: float = 0.95,
) -> list[Output]:
    """HuggingFace batch generation backend (default across all notebooks)."""
    if n_loops is not None:
        set_dory_loops(model, n_loops)

    # Expand prompts by n_samples
    expanded_prompts: list[str] = []
    for p in prompts:
        for _ in range(n_samples):
            expanded_prompts.append(p)

    # Support callable/mock models in unit tests or custom wrappers
    if tokenizer is None or not hasattr(model, "generate"):
        outputs: list[Output] = []
        for p in expanded_prompts:
            t0 = time.perf_counter()
            if callable(model):
                raw_out = model(p, max_new_tokens=max_new_tokens, temperature=temperature, n_loops=n_loops)
                gen_text = raw_out if isinstance(raw_out, str) else str(raw_out)
            else:
                raise TypeError("Model must implement `.generate()` or be callable when tokenizer is None.")
            elapsed = time.perf_counter() - t0
            ended_with_think = p.rstrip().endswith(THINK_OPEN)
            cot_text, answer_text = split_cot_and_answer(
                gen_text, prompt_ended_with_think=ended_with_think
            )
            n_cot = _count_tokens(cot_text, tokenizer)
            n_tot = _count_tokens(gen_text, tokenizer)
            outputs.append(
                Output(
                    text=gen_text,
                    cot_text=cot_text,
                    answer_text=answer_text,
                    n_cot_tokens=n_cot,
                    n_total_tokens=n_tot,
                    latency_s=elapsed,
                    backend="hf",
                )
            )
        return outputs

    # Ensure left-padding for batched causal LM generation
    orig_padding_side = getattr(tokenizer, "padding_side", "left")
    tokenizer.padding_side = "left"
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = next(model.parameters()).device
    results: list[Output] = []

    try:
        for start in range(0, len(expanded_prompts), batch_size):
            batch_prompts = expanded_prompts[start : start + batch_size]
            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            prompt_len = input_ids.shape[1]
            do_sample = temperature > 1e-5
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": int(max_new_tokens),
                "do_sample": do_sample,
                "use_cache": True,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                gen_kwargs["temperature"] = float(temperature)
                gen_kwargs["top_p"] = float(top_p)
            if n_loops is not None:
                gen_kwargs["n_loops"] = int(n_loops)

            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                out_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_elapsed = time.perf_counter() - t0
            per_item_latency = batch_elapsed / max(1, len(batch_prompts))

            for idx, prompt_str in enumerate(batch_prompts):
                gen_seq = out_ids[idx, prompt_len:]
                # Strip trailing pad/eos tokens when counting total generated tokens
                eos_id = tokenizer.eos_token_id
                pad_id = tokenizer.pad_token_id
                token_list = gen_seq.tolist()
                while token_list and token_list[-1] in (eos_id, pad_id):
                    token_list.pop()
                n_total_tokens = len(token_list)
                gen_text = tokenizer.decode(token_list, skip_special_tokens=False)
                ended_with_think = prompt_str.rstrip().endswith(THINK_OPEN)
                cot_text, answer_text = split_cot_and_answer(
                    gen_text, prompt_ended_with_think=ended_with_think
                )
                n_cot_tokens = count_cot_tokens(
                    gen_text,
                    tokenizer=tokenizer,
                    token_ids=token_list,
                    is_full_generation=True,
                    prompt_ended_with_think=ended_with_think,
                )
                results.append(
                    Output(
                        text=gen_text,
                        cot_text=cot_text,
                        answer_text=answer_text,
                        n_cot_tokens=n_cot_tokens,
                        n_total_tokens=n_total_tokens,
                        latency_s=per_item_latency,
                        backend="hf",
                    )
                )
    finally:
        tokenizer.padding_side = orig_padding_side

    return results


def evict_hf_models_from_gpu(model: Any | None = None) -> None:
    """Move any HuggingFace / PyTorch model off the GPU and clear CUDA cache before creating a vLLM engine."""
    import gc

    if model is not None and hasattr(model, "to") and hasattr(model, "parameters"):
        try:
            model.to("cpu")
        except Exception:
            pass

    if torch is not None and hasattr(torch, "nn"):
        for obj in gc.get_objects():
            try:
                if isinstance(obj, torch.nn.Module):
                    for p in obj.parameters(recurse=False):
                        if getattr(p, "is_cuda", False):
                            obj.to("cpu")
                            break
            except Exception:
                continue

    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


_ACTIVE_VLLM_CACHE: dict[str, Any] = {
    "key": None,
    "engine": None,
}


def _shutdown_cached_vllm_engine() -> None:
    """Release any cached vLLM engine and free GPU memory."""
    import gc

    if _ACTIVE_VLLM_CACHE.get("engine") is not None:
        _ACTIVE_VLLM_CACHE["engine"] = None
        _ACTIVE_VLLM_CACHE["key"] = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def _is_vllm_engine_instance(obj: Any) -> bool:
    """Return True if `obj` is an already-initialized vLLM `LLM` instance (or test mock engine),
    as opposed to a model repo string or a HuggingFace `PreTrainedModel`.
    """
    if isinstance(obj, str) or obj is None:
        return False
    if hasattr(obj, "parameters") and hasattr(obj, "prepare_inputs_for_generation"):
        return False
    if torch is not None and hasattr(torch, "nn") and isinstance(obj, torch.nn.Module):
        return False
    return hasattr(obj, "generate")


def _generate_vllm(
    vllm_engine_or_model_name: Any,
    prompts: list[str],
    *,
    tokenizer: Any | None,
    max_new_tokens: int,
    temperature: float,
    n_loops: int | None,
    n_samples: int,
    top_p: float = 0.95,
    recurrent_kv_cache_mode: Literal["per_loop", "last_loop"] = "per_loop",
    seed: int = 42,
) -> list[Output]:
    """vLLM generation backend (PR #60149) with `hf_overrides` for Dory and text-only init for Qwen3.5-4B."""
    from vllm import LLM, SamplingParams  # type: ignore[import-not-found]

    if _is_vllm_engine_instance(vllm_engine_or_model_name):
        llm = vllm_engine_or_model_name
    else:
        evict_hf_models_from_gpu(vllm_engine_or_model_name)
        if isinstance(vllm_engine_or_model_name, str):
            model_name = vllm_engine_or_model_name
        else:
            model_name = (
                getattr(getattr(vllm_engine_or_model_name, "config", None), "_name_or_path", None)
                or getattr(vllm_engine_or_model_name, "name_or_path", None)
                or DORY_HF_REPO
            )
            if n_loops is None:
                n_loops = getattr(
                    getattr(vllm_engine_or_model_name, "config", None), "n_recurrent_loops", None
                )

        is_qwen = ("qwen" in model_name.lower()) or (model_name == QWEN_CONTROL_HF_REPO)
        cache_key = (model_name, None if is_qwen else n_loops, recurrent_kv_cache_mode)

        if _ACTIVE_VLLM_CACHE.get("engine") is not None and _ACTIVE_VLLM_CACHE.get("key") == cache_key:
            llm = _ACTIVE_VLLM_CACHE["engine"]
        else:
            _shutdown_cached_vllm_engine()
            evict_hf_models_from_gpu()
            llm_kwargs: dict[str, Any] = {
                "model": model_name,
                "trust_remote_code": True,
                "dtype": "bfloat16",
                "seed": int(seed),
            }
            if is_qwen:
                # Initialize Qwen/Qwen3.5-4B in text-only mode (disable multimodal vision/video profiling)
                llm_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
            elif n_loops is not None:
                llm_kwargs["hf_overrides"] = build_dory_hf_overrides(
                    n_loops=int(n_loops),
                    recurrent_kv_cache_mode=recurrent_kv_cache_mode,
                )
            llm = LLM(**llm_kwargs)
            _ACTIVE_VLLM_CACHE["key"] = cache_key
            _ACTIVE_VLLM_CACHE["engine"] = llm

    if tokenizer is None and hasattr(llm, "get_tokenizer"):
        try:
            tokenizer = llm.get_tokenizer()
        except Exception:
            pass

    sampling_params = SamplingParams(
        n=int(n_samples),
        temperature=float(temperature),
        top_p=float(top_p) if temperature > 1e-5 else 1.0,
        max_tokens=int(max_new_tokens),
        seed=int(seed),
    )

    t0 = time.perf_counter()
    request_outputs = llm.generate(prompts, sampling_params)
    total_elapsed = time.perf_counter() - t0
    total_samples = max(1, len(prompts) * n_samples)
    per_sample_latency = total_elapsed / total_samples

    results: list[Output] = []
    for p_idx, req_out in enumerate(request_outputs):
        ended_with_think = prompts[p_idx].rstrip().endswith(THINK_OPEN)
        for sample in req_out.outputs:
            gen_text = sample.text
            sample_token_ids = (
                [int(t) for t in sample.token_ids]
                if hasattr(sample, "token_ids") and sample.token_ids is not None
                else None
            )
            n_total = (
                len(sample_token_ids)
                if sample_token_ids is not None
                else _count_tokens(gen_text, tokenizer)
            )
            cot_text, answer_text = split_cot_and_answer(
                gen_text, prompt_ended_with_think=ended_with_think
            )
            n_cot = count_cot_tokens(
                gen_text,
                tokenizer=tokenizer,
                token_ids=sample_token_ids,
                is_full_generation=True,
                prompt_ended_with_think=ended_with_think,
            )
            results.append(
                Output(
                    text=gen_text,
                    cot_text=cot_text,
                    answer_text=answer_text,
                    n_cot_tokens=n_cot,
                    n_total_tokens=n_total,
                    latency_s=per_sample_latency,
                    backend="vllm",
                )
            )
    return results


def generate(
    model: Any,
    prompts: list[str] | str,
    max_new_tokens: int = 4096,
    temperature: float = 0.0,
    n_loops: int | None = None,
    n_samples: int = 1,
    *,
    tokenizer: Any | None = None,
    backend: Literal["hf", "vllm"] = "hf",
    allow_hf_fallback: bool = False,
    recurrent_kv_cache_mode: Literal["per_loop", "last_loop"] = "per_loop",
    batch_size: int = 4,
    top_p: float = 0.95,
    seed: int = 42,
) -> list[Output]:
    """Unified generation wrapper across all notebooks and benchmarks (Section 1.3).

    Args:
        model: `DoryForCausalLM`, HuggingFace `PreTrainedModel`, `(model, tokenizer)` tuple,
            model repo string, vLLM `LLM` instance, or callable mock.
        prompts: Single prompt string or list of prompt strings.
        max_new_tokens: Maximum number of new tokens to generate (default 4096).
        temperature: Sampling temperature (`0.0` = greedy decoding).
        n_loops: Optional override for Dory's `n_recurrent_loops` (1, 2, or 4).
        n_samples: Number of completions to sample per prompt (default 1).
        tokenizer: Tokenizer (automatically unpacked if `model` is a `(model, tokenizer)` tuple).
        backend: `'hf'` (default) or `'vllm'`.
        allow_hf_fallback: When `False` (default) and `backend='vllm'` fails, raises an error.
            Fallback to HF is only permitted when `allow_hf_fallback=True` (e.g. smoke tests).
        recurrent_kv_cache_mode: `'per_loop'` (default) or `'last_loop'` for Dory's `hf_overrides`.
        batch_size: Batch size for HF generation.
        top_p: Nucleus sampling top-p when `temperature > 0`.
        seed: Random seed for reproducible sampling (default 42).

    Returns:
        List of `Output(text, cot_text, answer_text, n_cot_tokens, n_total_tokens, latency_s, backend)`
        of length `len(prompts) * n_samples`.
    """
    if isinstance(model, tuple) and len(model) == 2 and tokenizer is None:
        model, tokenizer = model

    prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
    if not prompt_list:
        return []

    if backend == "vllm":
        try:
            return _generate_vllm(
                model,
                prompt_list,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                n_loops=n_loops,
                n_samples=n_samples,
                top_p=top_p,
                recurrent_kv_cache_mode=recurrent_kv_cache_mode,
                seed=seed,
            )
        except Exception as exc:
            if not allow_hf_fallback:
                raise RuntimeError(
                    f"Falha no backend vLLM com allow_hf_fallback=False: {exc}"
                ) from exc
            return _generate_hf(
                model,
                prompt_list,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                n_loops=n_loops,
                n_samples=n_samples,
                batch_size=batch_size,
                top_p=top_p,
            )

    return _generate_hf(
        model,
        prompt_list,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        n_loops=n_loops,
        n_samples=n_samples,
        batch_size=batch_size,
        top_p=top_p,
    )
