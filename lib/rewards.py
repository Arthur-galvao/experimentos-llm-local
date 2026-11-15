"""Critical Module: GRPO reward = passed * exp(-lam * max(0, n_cot - free_tokens) / l_ref).

Sem bônus de formato: reasoning que nunca fecha `</think>` tem resposta vazia, então não passa e vale 0.
Só a CoT acima de `free_tokens` é penalizada, e a penalidade multiplica o acerto (errado vale sempre 0).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from .gen import split_cot_and_answer, _count_tokens
from .sandbox import check_solution


@dataclass
class RewardBreakdown:
    """Detailed breakdown of the GRPO reward for a single completion."""

    total_reward: float
    pass_score: float
    length_factor: float
    n_cot_tokens: int
    passed_tests: bool


def compute_reward(
    completion_text: str,
    *,
    l_ref: float,
    free_tokens: int,
    passed_tests: bool | None = None,
    test_code: str | None = None,
    entry_point: str | None = None,
    io_examples: list[dict[str, str]] | None = None,
    tokenizer: Any | None = None,
    n_cot_tokens: int | None = None,
    lam: float = 0.2,
    prompt_ended_with_think: bool = True,
    timeout_s: float = 6.0,
    max_memory_mb: int = 4096,
) -> RewardBreakdown:
    """Compute `passed * exp(-lam * max(0, n_cot_tokens - free_tokens) / l_ref)`.

    `passed` is 1.0 if the answer (text after `</think>`) passes all tests in the sandbox. An empty answer
    (reasoning never closed) is a fail and is not sent to the sandbox.
    """
    if l_ref <= 0:
        raise ValueError(f"l_ref must be > 0, got {l_ref}")

    cot_text, answer_text = split_cot_and_answer(
        completion_text, prompt_ended_with_think=prompt_ended_with_think
    )
    if n_cot_tokens is None:
        n_cot_tokens = _count_tokens(cot_text, tokenizer)

    if passed_tests is None:
        if not answer_text.strip():
            passed_tests = False
        else:
            passed_tests = check_solution(
                answer_text,
                test_code=test_code,
                entry_point=entry_point,
                io_examples=io_examples,
                timeout_s=timeout_s,
                max_memory_mb=max_memory_mb,
            ).passed

    pass_score = 1.0 if passed_tests else 0.0
    length_factor = math.exp(-float(lam) * max(0, int(n_cot_tokens) - int(free_tokens)) / float(l_ref))
    return RewardBreakdown(
        total_reward=pass_score * length_factor,
        pass_score=pass_score,
        length_factor=length_factor,
        n_cot_tokens=int(n_cot_tokens),
        passed_tests=bool(passed_tests),
    )


def make_grpo_reward_fn(
    *,
    l_ref: float,
    free_tokens: int,
    tokenizer: Any | None = None,
    lam: float = 0.2,
    prompt_ended_with_think: bool = True,
    timeout_s: float = 6.0,
    max_memory_mb: int = 4096,
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

            breakdown = compute_reward(
                comp_str,
                test_code=test_codes[idx] if idx < len(test_codes) else None,
                entry_point=entry_points[idx] if idx < len(entry_points) else None,
                io_examples=io_list[idx] if idx < len(io_list) else None,
                tokenizer=tokenizer,
                lam=lam,
                l_ref=l_ref,
                free_tokens=free_tokens,
                prompt_ended_with_think=prompt_ended_with_think,
                timeout_s=timeout_s,
                max_memory_mb=max_memory_mb,
            )
            rewards.append(float(breakdown.total_reward))
        return rewards

    return _reward_fn
