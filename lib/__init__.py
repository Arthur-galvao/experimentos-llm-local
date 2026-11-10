"""Shared library for the Coder Comprimido experimental pipeline (F0-F5)."""

from .hw import PROFILE, detect_hardware_profile, get_hw_config, print_hardware_summary
from .io import (
    DEFAULT_DRIVE_ROOT,
    CheckpointManager,
    ResumableProgress,
    setup_drive_and_cache,
)
from .registry import (
    RESULT_FIELDS,
    append_result,
    get_git_sha,
    has_completed_run,
    load_results,
)
from .metrics import (
    acerto_por_1k_tokens,
    check_phase_gate,
    compute_pareto_frontier,
    estimate_compute_cost,
    plot_pareto,
)
from .dory import (
    DoryConfig,
    DoryForCausalLM,
    DoryModel,
    init_new_tokens_embeddings,
    load_dory,
    logit_lens_per_loop,
    register_dory_auto,
    set_dory_loops,
)
from .gen import (
    Output,
    count_cot_tokens,
    format_chat_prompt,
    generate,
    split_cot_and_answer,
)
from .sandbox import (
    SandboxResult,
    check_solution,
    extract_python_code,
    run_code_in_sandbox,
)
from .rewards import (
    RewardBreakdown,
    compute_reward,
    make_grpo_reward_fn,
)
from .evals import (
    build_lcb_quick_split,
    eval_gate,
    eval_humaneval_plus,
    eval_lcb_pro,
    eval_lcb_v6,
    eval_mbpp_plus,
    eval_quick,
    eval_scicode,
)

__all__ = [
    "PROFILE",
    "detect_hardware_profile",
    "get_hw_config",
    "print_hardware_summary",
    "DEFAULT_DRIVE_ROOT",
    "CheckpointManager",
    "ResumableProgress",
    "setup_drive_and_cache",
    "RESULT_FIELDS",
    "append_result",
    "get_git_sha",
    "has_completed_run",
    "load_results",
    "acerto_por_1k_tokens",
    "check_phase_gate",
    "compute_pareto_frontier",
    "estimate_compute_cost",
    "plot_pareto",
    "DoryConfig",
    "DoryForCausalLM",
    "DoryModel",
    "init_new_tokens_embeddings",
    "load_dory",
    "logit_lens_per_loop",
    "register_dory_auto",
    "set_dory_loops",
    "Output",
    "count_cot_tokens",
    "format_chat_prompt",
    "generate",
    "split_cot_and_answer",
    "SandboxResult",
    "check_solution",
    "extract_python_code",
    "run_code_in_sandbox",
    "RewardBreakdown",
    "compute_reward",
    "make_grpo_reward_fn",
    "build_lcb_quick_split",
    "eval_gate",
    "eval_humaneval_plus",
    "eval_lcb_pro",
    "eval_lcb_v6",
    "eval_mbpp_plus",
    "eval_quick",
    "eval_scicode",
]
