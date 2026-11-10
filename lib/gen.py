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

from .dory import set_dory_loops

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


@dataclass
class Output:
    """Standardized generation output for all phases and evaluations."""

    text: str
    cot_text: str
    answer_text: str
    n_cot_tokens: int
    n_total_tokens: int
    latency_s: float


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


def count_cot_tokens(
    text_or_cot: str,
    tokenizer: Any | None = None,
    *,
    is_full_generation: bool = False,
    prompt_ended_with_think: bool = False,
) -> int:
    """Count the number of tokens in the Chain-of-Thought (CoT) segment.

    If `is_full_generation=True` (or if `<think>` / `</think>` tags are detected in `text_or_cot`),
    first extracts `cot_text` via `split_cot_and_answer` and then counts its tokens with
    `add_special_tokens=False` (excluding `<think>` and `</think>` delimiter tags themselves).
    """
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
                n_cot_tokens = _count_tokens(cot_text, tokenizer)
                results.append(
                    Output(
                        text=gen_text,
                        cot_text=cot_text,
                        answer_text=answer_text,
                        n_cot_tokens=n_cot_tokens,
                        n_total_tokens=n_total_tokens,
                        latency_s=per_item_latency,
                    )
                )
    finally:
        tokenizer.padding_side = orig_padding_side

    return results


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
) -> list[Output]:
    """Optional vLLM generation backend (used only if vLLM with Dory support is installed)."""
    from vllm import LLM, SamplingParams  # type: ignore[import-not-found]

    if isinstance(vllm_engine_or_model_name, str):
        hf_overrides = {"n_recurrent_loops": int(n_loops)} if n_loops is not None else {}
        llm = LLM(
            model=vllm_engine_or_model_name,
            trust_remote_code=True,
            dtype="bfloat16",
            hf_overrides=hf_overrides,
        )
    else:
        llm = vllm_engine_or_model_name

    sampling_params = SamplingParams(
        n=int(n_samples),
        temperature=float(temperature),
        top_p=float(top_p) if temperature > 1e-5 else 1.0,
        max_tokens=int(max_new_tokens),
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
            n_total = len(sample.token_ids) if hasattr(sample, "token_ids") else _count_tokens(gen_text, tokenizer)
            cot_text, answer_text = split_cot_and_answer(
                gen_text, prompt_ended_with_think=ended_with_think
            )
            n_cot = _count_tokens(cot_text, tokenizer)
            results.append(
                Output(
                    text=gen_text,
                    cot_text=cot_text,
                    answer_text=answer_text,
                    n_cot_tokens=n_cot,
                    n_total_tokens=n_total,
                    latency_s=per_sample_latency,
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
    batch_size: int = 4,
    top_p: float = 0.95,
) -> list[Output]:
    """Unified generation wrapper across all notebooks and benchmarks (Section 2.3).

    Args:
        model: `DoryForCausalLM`, HuggingFace `PreTrainedModel`, `(model, tokenizer)` tuple,
            vLLM `LLM` instance, or callable mock.
        prompts: Single prompt string or list of prompt strings.
        max_new_tokens: Maximum number of new tokens to generate (default 4096).
        temperature: Sampling temperature (`0.0` = greedy decoding).
        n_loops: Optional override for Dory's `n_recurrent_loops` (1, 2, or 4).
        n_samples: Number of completions to sample per prompt (default 1).
        tokenizer: Tokenizer (automatically unpacked if `model` is a `(model, tokenizer)` tuple).
        backend: `'hf'` (default) or `'vllm'`.
        batch_size: Batch size for HF generation.
        top_p: Nucleus sampling top-p when `temperature > 0`.

    Returns:
        List of `Output(text, cot_text, answer_text, n_cot_tokens, n_total_tokens, latency_s)`
        of length `len(prompts) * n_samples`.
    """
    if isinstance(model, tuple) and len(model) == 2 and tokenizer is None:
        model, tokenizer = model

    prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
    if not prompt_list:
        return []

    if backend == "vllm":
        return _generate_vllm(
            model,
            prompt_list,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            n_loops=n_loops,
            n_samples=n_samples,
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
