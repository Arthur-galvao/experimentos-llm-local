"""Critical Module: GRPO reward calculation with accuracy, CoT length penalty, and format bonus (Section 4 F1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .gen import THINK_CLOSE, THINK_OPEN, split_cot_and_answer, _count_tokens
from .sandbox import check_solution


@dataclass
class RewardBreakdown:
    """Detailed breakdown of the GRPO reward for a single completion."""

    total_reward: float
    pass_score: float
    length_penalty: float
    format_bonus: float
    n_cot_tokens: int
    passed_tests: bool


def has_valid_think_format(
    completion_text: str,
    *,
    prompt_ended_with_think: bool = False,
) -> bool:
    """Return True if the completion properly closes `</think>` and provides a non-empty final answer."""
    text = completion_text or ""
    if prompt_ended_with_think:
        if THINK_CLOSE not in text:
            return False
        _, answer = split_cot_and_answer(text, prompt_ended_with_think=True)
        return bool(answer.strip())

    open_idx = text.find(THINK_OPEN)
    close_idx = text.find(THINK_CLOSE)
    if open_idx == -1 or close_idx == -1 or open_idx >= close_idx:
        return False
    _, answer = split_cot_and_answer(text, prompt_ended_with_think=False)
    return bool(answer.strip())


def compute_reward(
    completion_text: str,
    *,
    passed_tests: bool | None = None,
    test_code: str | None = None,
    entry_point: str | None = None,
    io_examples: list[dict[str, str]] | None = None,
    tokenizer: Any | None = None,
    n_cot_tokens: int | None = None,
    lam: float = 0.1,
    l_ref: float = 512.0,
    format_bonus_val: float = 0.05,
    prompt_ended_with_think: bool = False,
    timeout_s: float = 4.0,
    max_memory_mb: int = 512,
) -> RewardBreakdown:
    """Compute the GRPO reward defined in Section 4 (F1) of `ESPEC_ANTIGRAVITY.md`:

        r = pass_tests(answer) - lam * (n_cot_tokens / L_ref) + format_bonus

    where:
      - `pass_tests(answer)` is 1.0 if all tests pass in the sandbox, 0.0 otherwise.
      - `lam` (λ) is the CoT compression weight (e.g., 0.05, 0.1, 0.2).
      - `l_ref` (L_ref) is the reference median CoT length of the baseline (> 0).
      - `format_bonus` is `format_bonus_val` (default 0.05) if `<think>...</think>` is properly
        formatted and followed by a non-empty answer, else 0.0.
    """
    if l_ref <= 0:
        raise ValueError(f"l_ref must be > 0, got {l_ref}")

    cot_text, answer_text = split_cot_and_answer(
        completion_text, prompt_ended_with_think=prompt_ended_with_think
    )
    if n_cot_tokens is None:
        n_cot_tokens = _count_tokens(cot_text, tokenizer)

    if passed_tests is None:
        candidate_for_sandbox = answer_text if answer_text.strip() else completion_text
        sandbox_res = check_solution(
            candidate_for_sandbox,
            test_code=test_code,
            entry_point=entry_point,
            io_examples=io_examples,
            timeout_s=timeout_s,
            max_memory_mb=max_memory_mb,
        )
        passed_tests = sandbox_res.passed

    pass_score = 1.0 if passed_tests else 0.0
    length_penalty = float(lam) * (float(n_cot_tokens) / float(l_ref))
    fmt_ok = has_valid_think_format(
        completion_text, prompt_ended_with_think=prompt_ended_with_think
    )
    fmt_bonus = float(format_bonus_val) if fmt_ok else 0.0

    total = pass_score - length_penalty + fmt_bonus
    return RewardBreakdown(
        total_reward=total,
        pass_score=pass_score,
        length_penalty=length_penalty,
        format_bonus=fmt_bonus,
        n_cot_tokens=int(n_cot_tokens),
        passed_tests=bool(passed_tests),
    )


def make_grpo_reward_fn(
    *,
    tokenizer: Any | None = None,
    lam: float = 0.1,
    l_ref: float = 512.0,
    format_bonus_val: float = 0.05,
    timeout_s: float = 4.0,
    max_memory_mb: int = 512,
) -> Callable[..., list[float]]:
    """Create a TRL `GRPOTrainer`-compatible reward function `reward_fn(prompts, completions, **kwargs)`."""

    def _reward_fn(
        prompts: list[Any],
        completions: list[Any],
        **kwargs: Any,
    ) -> list[float]:
        test_codes = kwargs.get("test_code") or kwargs.get("test") or [None] * len(completions)
        entry_points = kwargs.get("entry_point") or [None] * len(completions)
        io_list = kwargs.get("io_examples") or [None] * len(completions)

        rewards: list[float] = []
        for idx, comp in enumerate(completions):
            if isinstance(comp, list) and comp and isinstance(comp[0], dict):
                comp_str = str(comp[-1].get("content", ""))
            else:
                comp_str = str(comp)

            prompt_item = prompts[idx] if idx < len(prompts) else ""
            if isinstance(prompt_item, list) and prompt_item and isinstance(prompt_item[-1], dict):
                prompt_str = str(prompt_item[-1].get("content", ""))
            else:
                prompt_str = str(prompt_item)

            ended_with_think = prompt_str.rstrip().endswith(THINK_OPEN)
            breakdown = compute_reward(
                comp_str,
                test_code=test_codes[idx] if idx < len(test_codes) else None,
                entry_point=entry_points[idx] if idx < len(entry_points) else None,
                io_examples=io_list[idx] if idx < len(io_list) else None,
                tokenizer=tokenizer,
                lam=lam,
                l_ref=l_ref,
                format_bonus_val=format_bonus_val,
                prompt_ended_with_think=ended_with_think,
                timeout_s=timeout_s,
                max_memory_mb=max_memory_mb,
            )
            rewards.append(float(breakdown.total_reward))
        return rewards

    return _reward_fn
