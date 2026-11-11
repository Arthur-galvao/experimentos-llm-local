"""Unit tests for all critical modules in `lib/` (Section 6 of ESPEC_ANTIGRAVITY.md).

Compatible with both `pytest tests/` (Colab) and `python -m unittest -v tests/test_critical.py` (local).
"""

from __future__ import annotations

import math
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from lib.dory import (
    DORY_HF_REPO,
    QWEN_CONTROL_HF_REPO,
    DoryConfig,
    DoryForCausalLM,
    init_new_tokens_embeddings,
    logit_lens_per_loop,
    set_dory_loops,
    verify_vllm_hf_equivalence,
)
from lib.evals import build_lcb_quick_split
from lib.gen import (
    DORY_VLLM_OVERRIDE_KEYS,
    build_dory_hf_overrides,
    count_cot_tokens,
    generate,
    run_vllm_in_subprocess,
    split_cot_and_answer,
)
from lib.hw import detect_hardware_profile, get_hw_config
from lib.io import CheckpointManager, ResumableProgress, setup_drive_and_cache
from lib.metrics import check_phase_gate, compute_pareto_frontier, plot_pareto
from lib.registry import RESULT_FIELDS, append_result, load_results
from lib.rewards import compute_reward, make_grpo_reward_fn
from lib.sandbox import check_solution, run_code_in_sandbox


class DeterministicWordTokenizer:
    """Deterministic word/symbol tokenizer for self-contained offline unit tests."""

    def __init__(self) -> None:
        base_words = [
            "<pad>",
            "<bos>",
            "<eos>",
            "<think>",
            "</think>",
            "step",
            "1",
            "2",
            "3",
            "we",
            "need",
            "to",
            "add",
            "two",
            "numbers",
            "a",
            "and",
            "b",
            "return",
            "sum",
            "for",
            "loop",
            "range",
            "if",
            "else",
            "def",
            "solve",
            "x",
            "y",
        ]
        self.tok_to_id: dict[str, int] = {w: i for i, w in enumerate(base_words)}
        self.id_to_tok: dict[int, str] = {i: w for i, w in enumerate(base_words)}
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.padding_side = "left"

    def __len__(self) -> int:
        return len(self.tok_to_id)

    def add_tokens(self, new_tokens: list[str], special_tokens: bool = False) -> int:
        added = 0
        for tok in new_tokens:
            if tok not in self.tok_to_id:
                idx = len(self.tok_to_id)
                self.tok_to_id[tok] = idx
                self.id_to_tok[idx] = tok
                added += 1
        return added

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.tok_to_id[token]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        tokens = text.strip().split()
        ids: list[int] = []
        if add_special_tokens:
            ids.append(self.bos_token_id)
        for t in tokens:
            ids.append(self.tok_to_id.get(t, 5))
        return ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        words: list[str] = []
        for tid in token_ids:
            t_int = int(tid)
            if skip_special_tokens and t_int in (self.pad_token_id, self.bos_token_id, self.eos_token_id):
                continue
            words.append(self.id_to_tok.get(t_int, f"tok_{t_int}"))
        return " ".join(words)

    def __call__(
        self,
        prompts: str | list[str],
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = False,
    ) -> dict[str, Any]:
        batch = [prompts] if isinstance(prompts, str) else list(prompts)
        encoded = [self.encode(p, add_special_tokens=True) for p in batch]
        max_len = max(len(seq) for seq in encoded)
        padded: list[list[int]] = []
        masks: list[list[int]] = []
        for seq in encoded:
            pad_len = max_len - len(seq)
            if self.padding_side == "left":
                padded.append([self.pad_token_id] * pad_len + seq)
                masks.append([0] * pad_len + [1] * len(seq))
            else:
                padded.append(seq + [self.pad_token_id] * pad_len)
                masks.append([1] * len(seq) + [0] * pad_len)
        if torch is not None:
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        return {"input_ids": padded, "attention_mask": masks}


def _make_tiny_dory(n_loops: int = 2) -> tuple[DoryForCausalLM, DeterministicWordTokenizer]:
    """Build a small deterministic DoryForCausalLM model for fast CPU/GPU testing."""
    if torch is not None:
        torch.manual_seed(42)
    tokenizer = DeterministicWordTokenizer()
    cfg = DoryConfig(
        vocab_size=len(tokenizer),
        hidden_size=64,
        num_hidden_layers=8,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        n_input_layers=2,
        n_recurrent_layers=4,
        n_output_layers=2,
        n_recurrent_loops=n_loops,
        tie_word_embeddings=False,
    )
    model = DoryForCausalLM(cfg)
    model.eval()
    return model, tokenizer


def _flatten_floats(x: Any) -> list[float]:
    if hasattr(x, "detach"):
        return [float(v) for v in x.detach().float().view(-1).tolist()]
    if isinstance(x, list):
        out: list[float] = []
        for item in x:
            out.extend(_flatten_floats(item))
        return out
    return [float(x)]


def _vectors_close(a: Any, b: Any, atol: float = 1e-5) -> bool:
    fa = _flatten_floats(a)
    fb = _flatten_floats(b)
    if len(fa) != len(fb):
        return False
    return all(abs(x - y) <= atol for x, y in zip(fa, fb))


# ==============================================================================
# 1. Mandatory Test: CoT token counting on an example with a known result
# ==============================================================================
def test_cot_token_counting_known_example() -> None:
    tokenizer = DeterministicWordTokenizer()
    sample = "<think> step 1 we need to add two numbers a and b </think> def solve ( a , b ) : return a + b"

    cot_text, answer_text = split_cot_and_answer(sample)
    assert cot_text == "step 1 we need to add two numbers a and b"
    assert answer_text == "def solve ( a , b ) : return a + b"

    # Known exact CoT token count: 11 tokens ("step", "1", "we", "need", "to", "add", "two", "numbers", "a", "and", "b")
    n_cot = count_cot_tokens(sample, tokenizer=tokenizer, is_full_generation=True)
    assert n_cot == 11

    # Case where <think> was already part of the prompt template
    sample_no_open = "step 1 step 2 return sum </think> return a + b"
    cot2, ans2 = split_cot_and_answer(sample_no_open, prompt_ended_with_think=True)
    assert cot2 == "step 1 step 2 return sum"
    assert ans2 == "return a + b"
    assert count_cot_tokens(sample_no_open, tokenizer=tokenizer, prompt_ended_with_think=True) == 6

    # Case where generation truncated before </think>
    truncated = "<think> step 1 we need to"
    cot3, ans3 = split_cot_and_answer(truncated)
    assert cot3 == "step 1 we need to"
    assert ans3 == ""
    assert count_cot_tokens(truncated, tokenizer=tokenizer) == 5


# ==============================================================================
# 2. Mandatory Test: Reward calculation in synthetic cases (correct/short, correct/long, wrong)
# ==============================================================================
def test_reward_synthetic_cases() -> None:
    tokenizer = DeterministicWordTokenizer()
    test_suite = "assert add(2, 3) == 5\nassert add(-1, 1) == 0"

    short_correct = (
        "<think> add a and b </think>\n"
        "```python\ndef add(a, b):\n    return a + b\n```"
    )
    long_cot_words = " ".join(["step 1 we need to add two numbers a and b"] * 10)  # 110 tokens
    long_correct = (
        f"<think> {long_cot_words} </think>\n"
        "```python\ndef add(a, b):\n    return a + b\n```"
    )
    wrong_completion = (
        "<think> add a and b </think>\n"
        "```python\ndef add(a, b):\n    return a - b\n```"
    )

    r_short = compute_reward(
        short_correct,
        test_code=test_suite,
        tokenizer=tokenizer,
        lam=0.1,
        l_ref=100.0,
        format_bonus_val=0.05,
    )
    r_long = compute_reward(
        long_correct,
        test_code=test_suite,
        tokenizer=tokenizer,
        lam=0.1,
        l_ref=100.0,
        format_bonus_val=0.05,
    )
    r_wrong = compute_reward(
        wrong_completion,
        test_code=test_suite,
        tokenizer=tokenizer,
        lam=0.1,
        l_ref=100.0,
        format_bonus_val=0.05,
    )

    assert r_short.passed_tests is True
    assert r_long.passed_tests is True
    assert r_wrong.passed_tests is False

    assert r_short.n_cot_tokens == 4
    assert r_long.n_cot_tokens == 110

    # Exact formula checks: r = pass_score - lam * (n_cot / l_ref) + format_bonus
    assert abs(r_short.total_reward - (1.0 - 0.1 * (4 / 100.0) + 0.05)) < 1e-6
    assert abs(r_long.total_reward - (1.0 - 0.1 * (110 / 100.0) + 0.05)) < 1e-6
    assert abs(r_wrong.total_reward - (0.0 - 0.1 * (4 / 100.0) + 0.05)) < 1e-6

    # Ordering: correct/short > correct/long > wrong
    assert r_short.total_reward > r_long.total_reward > r_wrong.total_reward

    # Also verify GRPOTrainer wrapper function
    grpo_fn = make_grpo_reward_fn(tokenizer=tokenizer, lam=0.1, l_ref=100.0, format_bonus_val=0.05)
    batch_rewards = grpo_fn(
        prompts=["Solve add", "Solve add", "Solve add"],
        completions=[short_correct, long_correct, wrong_completion],
        test_code=[test_suite, test_suite, test_suite],
    )
    assert len(batch_rewards) == 3
    assert batch_rewards[0] > batch_rewards[1] > batch_rewards[2]


# ==============================================================================
# 3. Mandatory Test: Sandbox respecting timeout and memory limit
# ==============================================================================
def test_sandbox_timeout_and_memory_limit() -> None:
    # 3a. Valid code passes quickly
    ok_res = check_solution(
        "```python\ndef mul(a, b):\n    return a * b\n```",
        test_code="assert mul(6, 7) == 42",
        timeout_s=3.0,
        max_memory_mb=256,
    )
    assert ok_res.passed is True
    assert ok_res.timed_out is False
    assert ok_res.memory_exceeded is False

    # 3b. Infinite loop triggers timeout within bounded wall-clock time
    t0 = time.perf_counter()
    timeout_res = run_code_in_sandbox(
        "while True:\n    pass\n",
        timeout_s=0.6,
        max_memory_mb=256,
    )
    elapsed = time.perf_counter() - t0
    assert timeout_res.passed is False
    assert timeout_res.timed_out is True
    assert elapsed < 4.0

    # 3c. Excessive allocation triggers memory limit failure
    mem_hog_code = "x = bytearray(300 * 1024 * 1024)\nprint(len(x))\n"
    mem_res = run_code_in_sandbox(
        mem_hog_code,
        timeout_s=4.0,
        max_memory_mb=64,
    )
    assert mem_res.passed is False
    assert mem_res.memory_exceeded is True or mem_res.exit_code != 0


# ==============================================================================
# 4. Mandatory Test: Changing n_loops alters output and execution time + logit lens
# ==============================================================================
def test_dory_n_loops_changes_output_and_runtime() -> None:
    model, tokenizer = _make_tiny_dory(n_loops=1)
    prompt = "step 1 we need to add two numbers"
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"]

    logits_by_loop: dict[int, list[float]] = {}
    timings_by_loop: dict[int, float] = {}

    # Warm-up pass
    _ = model(input_ids=input_ids, n_loops=1)

    for k in (1, 2, 4):
        set_dory_loops(model, k)
        assert model.config.n_recurrent_loops == k

        t0 = time.perf_counter()
        for _ in range(25):
            out = model(input_ids=input_ids)
        timings_by_loop[k] = time.perf_counter() - t0
        logits_by_loop[k] = _flatten_floats(out.logits)

    # Outputs (logits) must differ across 1, 2, and 4 loops
    assert not _vectors_close(logits_by_loop[1], logits_by_loop[2], atol=1e-5)
    assert not _vectors_close(logits_by_loop[2], logits_by_loop[4], atol=1e-5)
    assert not _vectors_close(logits_by_loop[1], logits_by_loop[4], atol=1e-5)

    # 4 loops executes 4x the middle recurrent block vs 1 loop -> longer execution time
    assert timings_by_loop[4] > timings_by_loop[1]

    # Test logit_lens_per_loop
    lens = logit_lens_per_loop(
        model,
        tokenizer,
        prompt,
        n_loops=4,
        top_k=3,
        last_n_positions=4,
    )
    assert len(lens) == 4
    for idx, loop_entry in enumerate(lens, start=1):
        assert loop_entry["loop_idx"] == idx
        assert len(loop_entry["positions"]) == 4
        for pos_info in loop_entry["positions"]:
            assert len(pos_info["top_tokens"]) == 3
            assert len(pos_info["top_probs"]) == 3


# ==============================================================================
# 5. Mandatory Test: New tokens with untied embeddings initialized + round-trip
# ==============================================================================
def test_new_tokens_untied_embeddings_init_and_roundtrip() -> None:
    model, tokenizer = _make_tiny_dory(n_loops=2)
    assert model.config.tie_word_embeddings is False

    old_vocab_size = len(tokenizer)
    in_emb_before = model.get_input_embeddings().weight.detach().clone()
    out_emb_before = model.get_output_embeddings().weight.detach().clone()

    for_id = tokenizer.convert_tokens_to_ids("for")
    loop_id = tokenizer.convert_tokens_to_ids("loop")
    if torch is not None:
        expected_in_vec = _flatten_floats(in_emb_before[[for_id, loop_id]].float().mean(dim=0))
        expected_out_vec = _flatten_floats(out_emb_before[[for_id, loop_id]].float().mean(dim=0))
    else:
        expected_in_vec = in_emb_before[[for_id, loop_id]].mean_axis0()
        expected_out_vec = out_emb_before[[for_id, loop_id]].mean_axis0()

    new_tokens_map = {
        "<FOR_LOOP>": "for loop",
        "<IF_BRANCH>": "if else",
        "<LATENT>": "step 1 2",
    }
    assigned = init_new_tokens_embeddings(model, tokenizer, new_tokens_map)

    assert len(tokenizer) == old_vocab_size + 3
    assert model.get_input_embeddings().weight.shape[0] == old_vocab_size + 3
    assert model.get_output_embeddings().weight.shape[0] == old_vocab_size + 3
    # Untied weights must not share storage
    assert (
        model.get_input_embeddings().weight.data_ptr()
        != model.get_output_embeddings().weight.data_ptr()
    )

    for tok_str, tok_id in assigned.items():
        in_vec = _flatten_floats(model.get_input_embeddings().weight[tok_id])
        out_vec = _flatten_floats(model.get_output_embeddings().weight[tok_id])
        assert not any(math.isnan(v) for v in in_vec)
        assert not any(math.isnan(v) for v in out_vec)
        assert sum(abs(v) for v in in_vec) > 0.0
        assert sum(abs(v) for v in out_vec) > 0.0

    # Verify exact subtoken mean initialization for `<FOR_LOOP>`
    for_loop_id = assigned["<FOR_LOOP>"]
    assert _vectors_close(
        model.get_input_embeddings().weight[for_loop_id],
        expected_in_vec,
        atol=1e-5,
    )
    assert _vectors_close(
        model.get_output_embeddings().weight[for_loop_id],
        expected_out_vec,
        atol=1e-5,
    )

    # Verify encode/decode round-trip
    sample_dsl = "<think> <FOR_LOOP> x <IF_BRANCH> return sum </think>"
    encoded_ids = tokenizer.encode(sample_dsl, add_special_tokens=False)
    assert assigned["<FOR_LOOP>"] in encoded_ids
    assert assigned["<IF_BRANCH>"] in encoded_ids
    decoded_str = tokenizer.decode(encoded_ids, skip_special_tokens=False)
    assert decoded_str == sample_dsl


# ==============================================================================
# 6. Additional Tests: Registry, Pareto Frontier, Phase Gate, Checkpoints, Evals
# ==============================================================================
def test_registry_and_pareto_and_phase_gate() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        results_file = tmp_path / "results" / "results.jsonl"

        r1 = append_result(
            run_id="F0_dory_L2",
            phase="F0",
            model="dory-4b",
            checkpoint="base",
            n_loops=2,
            eval_tier="quick",
            bench="lcb_v6_quick",
            pass1=0.45,
            tokens_cot_mean=1000.0,
            tokens_total_mean=1200.0,
            latency_mean_s=2.5,
            results_path=results_file,
        )
        assert tuple(r1.keys()) == RESULT_FIELDS
        assert abs(r1["acerto_por_1k_tokens"] - 0.45) < 1e-6

        r2 = append_result(
            run_id="F1_grpo_lam0.1",
            phase="F1",
            model="dory-4b",
            checkpoint="ckpt-200",
            n_loops=2,
            eval_tier="quick",
            bench="lcb_v6_quick",
            pass1=0.46,
            tokens_cot_mean=650.0,
            tokens_total_mean=820.0,
            latency_mean_s=1.7,
            results_path=results_file,
        )

        loaded = load_results(results_file)
        assert len(loaded) == 2

        frontier = compute_pareto_frontier(loaded)
        assert len(frontier) == 1
        assert frontier[0]["run_id"] == "F1_grpo_lam0.1"

        plot_path = tmp_path / "results" / "plots" / "pareto_test.png"
        saved_plot = plot_pareto(loaded, plot_path, bench_filter="lcb_v6_quick")
        assert saved_plot.exists() and saved_plot.stat().st_size > 0

        verdict_pass = check_phase_gate(
            [r2],
            [r1],
            baseline_scicode_pass1=0.30,
            candidate_scicode_pass1=0.29,
        )
        assert verdict_pass["passed"] is True
        assert verdict_pass["status"] == "PASSOU"

        verdict_fail_scicode = check_phase_gate(
            [r2],
            [r1],
            baseline_scicode_pass1=0.30,
            candidate_scicode_pass1=0.25,
        )
        assert verdict_fail_scicode["passed"] is False
        assert verdict_fail_scicode["status"] == "REPROVOU"


def test_checkpoint_manager_and_resumable_progress() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        mgr = CheckpointManager(tmp_path / "ckpts", max_to_keep=2, save_steps=10)
        mgr.save_state(10, {"loss": 1.2})
        mgr.save_state(20, {"loss": 1.0})
        mgr.save_state(30, {"loss": 0.8})

        ckpts = mgr.list_checkpoints()
        assert len(ckpts) == 2
        assert [p.name for p in ckpts] == ["checkpoint-20", "checkpoint-30"]
        assert mgr.find_latest_checkpoint() == ckpts[-1]

        prog_file = tmp_path / "eval_cache" / "progress.jsonl"
        prog = ResumableProgress(prog_file)
        prog.record("prob_1", {"passed": True, "n_cot_tokens": 42})
        prog2 = ResumableProgress(prog_file)
        assert prog2.is_done("prob_1")
        assert prog2.get("prob_1")["n_cot_tokens"] == 42

        paths = setup_drive_and_cache(local_fallback_root=tmp_path / "drive_proj")
        import os
        assert os.environ["HF_HOME"] == "/root/.cache/huggingface"
        assert os.environ["HF_HUB_CACHE"] == "/root/.cache/huggingface/hub"
        assert os.environ["HF_DATASETS_CACHE"] == "/root/.cache/huggingface/datasets"
        assert str(paths["hf_home"]).replace("\\", "/") == "/root/.cache/huggingface"


def test_hw_config_and_lcb_quick_split() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        profile = detect_hardware_profile()
        assert profile in ("a100_80", "a100_40")
        cfg_80 = get_hw_config("grpo", profile="a100_80")
        cfg_40 = get_hw_config("grpo", profile="a100_40")
        assert cfg_80["num_generations"] == 8
        assert cfg_40["num_generations"] == 4
        assert cfg_40["load_in_4bit"] is True

        synthetic_lcb = [
            {"question_id": f"q_{i}", "contest_date": f"2025-01-{i:02d}", "question_content": "print(1)"}
            for i in range(1, 15)
        ]
        split_path = tmp_path / "data" / "lcb_quick.json"
        split = build_lcb_quick_split(split_path, n_problems=5, preloaded_items=synthetic_lcb)
        assert len(split) == 5
        assert [x["question_id"] for x in split] == ["q_14", "q_13", "q_12", "q_11", "q_10"]


# ==============================================================================
# 9. New Mandatory Tests: hf_overrides keys, exact token_ids CoT count, allow_hf_fallback
# ==============================================================================
def test_hf_overrides_keys_match_dory_config() -> None:
    cfg = DoryConfig()
    cfg_dict = cfg.__dict__
    overrides = build_dory_hf_overrides(n_loops=4, recurrent_kv_cache_mode="per_loop")

    assert tuple(overrides.keys()) == DORY_VLLM_OVERRIDE_KEYS
    assert set(overrides.keys()) == {"n_recurrent_loops", "recurrent_kv_cache_mode"}
    for key in overrides:
        assert hasattr(cfg, key) and key in cfg_dict, (
            f"Chave '{key}' de hf_overrides não existe nos campos do DoryConfig/config.json!"
        )
    assert overrides["n_recurrent_loops"] == 4
    assert overrides["recurrent_kv_cache_mode"] == "per_loop"


def test_cot_token_counting_exact_from_token_ids() -> None:
    tokenizer = DeterministicWordTokenizer()
    think_open_id = tokenizer.convert_tokens_to_ids("<think>")
    think_close_id = tokenizer.convert_tokens_to_ids("</think>")

    # Skewed text where CoT has 3 tokens (very short text) and answer has 2 tokens (very long text):
    # A character-ratio heuristic would estimate ~1 CoT token instead of the exact 3 tokens.
    skewed_text = "<think> a b x </think> " + ("super_long_identifier_name " * 40).strip()
    exact_token_ids = [
        think_open_id,
        tokenizer.convert_tokens_to_ids("a"),
        tokenizer.convert_tokens_to_ids("b"),
        tokenizer.convert_tokens_to_ids("x"),
        think_close_id,
        tokenizer.convert_tokens_to_ids("return"),
        tokenizer.convert_tokens_to_ids("sum"),
    ]

    n_cot = count_cot_tokens(
        skewed_text,
        tokenizer=tokenizer,
        token_ids=exact_token_ids,
        is_full_generation=True,
    )
    assert n_cot == 3

    # Case where <think> was already in the prompt (prompt_ended_with_think=True): 4 tokens before </think>
    ids_without_open = [
        tokenizer.convert_tokens_to_ids("step"),
        tokenizer.convert_tokens_to_ids("1"),
        tokenizer.convert_tokens_to_ids("add"),
        tokenizer.convert_tokens_to_ids("a"),
        think_close_id,
        tokenizer.convert_tokens_to_ids("return"),
    ]
    n_cot_no_open = count_cot_tokens(
        "step 1 add a </think> " + ("long_answer_padding " * 50),
        tokenizer=tokenizer,
        token_ids=ids_without_open,
        is_full_generation=True,
        prompt_ended_with_think=True,
    )
    assert n_cot_no_open == 4


def test_vllm_failure_raises_when_allow_hf_fallback_false() -> None:
    tokenizer = DeterministicWordTokenizer()

    class FailingMockVLLMEngine:
        def generate(self, prompts: list[str], sampling_params: Any) -> Any:
            raise RuntimeError("Simulated vLLM engine failure")

    # 1. allow_hf_fallback=False (default) must raise when vLLM fails
    raised = False
    try:
        generate(
            FailingMockVLLMEngine(),
            ["Solve x"],
            tokenizer=tokenizer,
            backend="vllm",
            allow_hf_fallback=False,
        )
    except RuntimeError:
        raised = True
    assert raised is True, "generate(..., backend='vllm', allow_hf_fallback=False) deve levantar erro!"

    # 2. allow_hf_fallback=True falls back to HF and records backend='hf'
    def mock_hf_callable(prompt: str, **kwargs: Any) -> str:
        return "<think> step 1 </think> return x"

    fallback_outs = generate(
        mock_hf_callable,
        ["Solve x"],
        tokenizer=tokenizer,
        backend="vllm",
        allow_hf_fallback=True,
    )
    assert len(fallback_outs) == 1
    assert fallback_outs[0].backend == "hf"
    assert fallback_outs[0].n_cot_tokens == 2


def test_each_loop_spawns_new_vllm_subprocess() -> None:
    import json
    import os

    parent_pid = os.getpid()
    events: list[str] = []
    spawned_pids_equiv: list[int] = []

    def fake_equiv_subprocess_runner(in_json: Path, out_json: Path) -> None:
        req = json.loads(in_json.read_text(encoding="utf-8"))
        # Execute inside a real spawned OS child process via run_vllm_in_subprocess(subprocess_runner=None)
        child_data = run_vllm_in_subprocess(req, subprocess_runner=None)
        child_pid = int(child_data["pid"])
        spawned_pids_equiv.append(child_pid)
        events.append(f"vllm_subproc_loop_{req['n_loops']}")
        out_json.write_text(json.dumps(child_data, ensure_ascii=False), encoding="utf-8")

    def fake_hf_runner(
        prompts: list[str],
        vllm_loop_items: list[dict[str, Any]],
        k_loops: int,
        max_new_tokens: int,
    ) -> list[list[dict[str, Any]]]:
        # When HF runs, all 3 vLLM subprocesses (1, 2, 4) must have already finished
        assert events[:3] == ["vllm_subproc_loop_1", "vllm_subproc_loop_2", "vllm_subproc_loop_4"]
        events.append(f"hf_eval_loop_{k_loops}")
        return [item["steps"] for item in vllm_loop_items]

    report = verify_vllm_hf_equivalence(
        prompts=("Prompt A", "Prompt B"),
        loops=(1, 2, 4),
        max_new_tokens=8,
        subprocess_runner=fake_equiv_subprocess_runner,
        hf_runner=fake_hf_runner,
        fake_vllm=True,
        halt_on_failure=True,
    )

    assert report["passed"] is True
    assert len(spawned_pids_equiv) == 3
    # Verify default gpu_memory_utilization in verify_vllm_hf_equivalence is 0.45
    import inspect as _insp
    assert _insp.signature(verify_vllm_hf_equivalence).parameters["gpu_memory_utilization"].default == 0.45
    # Every loop count (1, 2, 4) spawned a brand-new OS process distinct from parent and from each other
    assert all(pid != parent_pid for pid in spawned_pids_equiv)
    assert len(set(spawned_pids_equiv)) == 3
    assert report["execution_order"] == [
        "vllm_loop_1",
        "vllm_loop_2",
        "vllm_loop_4",
        "hf_load_and_eval",
    ]
    assert events == [
        "vllm_subproc_loop_1",
        "vllm_subproc_loop_2",
        "vllm_subproc_loop_4",
        "hf_eval_loop_1",
        "hf_eval_loop_2",
        "hf_eval_loop_4",
    ]

    # Also verify _generate_vllm opens a separate subprocess for each Dory loop (1, 2, 4) and for Qwen
    gen_subproc_records: list[dict[str, Any]] = []

    def fake_gen_subprocess_runner(in_json: Path, out_json: Path) -> None:
        req = json.loads(in_json.read_text(encoding="utf-8"))
        child_data = run_vllm_in_subprocess(req, subprocess_runner=None)
        gen_subproc_records.append(
            {
                "pid": int(child_data["pid"]),
                "model_name": req["model_name"],
                "n_loops": req["n_loops"],
                "hf_overrides": req["hf_overrides"],
                "limit_mm_per_prompt": req["limit_mm_per_prompt"],
            }
        )
        out_json.write_text(json.dumps(child_data, ensure_ascii=False), encoding="utf-8")

    tokenizer = DeterministicWordTokenizer()
    for k in (1, 2, 4):
        outs_k = generate(
            DORY_HF_REPO,
            ["Solve task"],
            n_loops=k,
            tokenizer=tokenizer,
            backend="vllm",
            subprocess_runner=fake_gen_subprocess_runner,
            fake_vllm=True,
        )
        assert len(outs_k) == 1
        assert outs_k[0].backend == "vllm"
        assert outs_k[0].n_cot_tokens == 6

    outs_qwen = generate(
        QWEN_CONTROL_HF_REPO,
        ["Solve task"],
        n_loops=None,
        tokenizer=tokenizer,
        backend="vllm",
        subprocess_runner=fake_gen_subprocess_runner,
        fake_vllm=True,
    )
    assert len(outs_qwen) == 1
    assert outs_qwen[0].backend == "vllm"

    assert len(gen_subproc_records) == 4
    gen_pids = [r["pid"] for r in gen_subproc_records]
    assert all(pid != parent_pid for pid in gen_pids)
    assert len(set(gen_pids)) == 4
    assert [r["n_loops"] for r in gen_subproc_records[:3]] == [1, 2, 4]
    assert gen_subproc_records[3]["model_name"] == QWEN_CONTROL_HF_REPO
    assert gen_subproc_records[3]["limit_mm_per_prompt"] == {"image": 0, "video": 0}


class CriticalModulesTestSuite(unittest.TestCase):
    """Unittest wrapper so tests can also be run via `python -m unittest` if pytest is absent."""

    def test_1_cot_token_counting_known_example(self) -> None:
        test_cot_token_counting_known_example()

    def test_2_reward_synthetic_cases(self) -> None:
        test_reward_synthetic_cases()

    def test_3_sandbox_timeout_and_memory_limit(self) -> None:
        test_sandbox_timeout_and_memory_limit()

    def test_4_dory_n_loops_changes_output_and_runtime(self) -> None:
        test_dory_n_loops_changes_output_and_runtime()

    def test_5_new_tokens_untied_embeddings_init_and_roundtrip(self) -> None:
        test_new_tokens_untied_embeddings_init_and_roundtrip()

    def test_6_registry_and_pareto_and_phase_gate(self) -> None:
        test_registry_and_pareto_and_phase_gate()

    def test_7_checkpoint_manager_and_resumable_progress(self) -> None:
        test_checkpoint_manager_and_resumable_progress()

    def test_8_hw_config_and_lcb_quick_split(self) -> None:
        test_hw_config_and_lcb_quick_split()

    def test_9_hf_overrides_keys_match_dory_config(self) -> None:
        test_hf_overrides_keys_match_dory_config()

    def test_10_cot_token_counting_exact_from_token_ids(self) -> None:
        test_cot_token_counting_exact_from_token_ids()

    def test_11_vllm_failure_raises_when_allow_hf_fallback_false(self) -> None:
        test_vllm_failure_raises_when_allow_hf_fallback_false()

    def test_12_each_loop_spawns_new_vllm_subprocess(self) -> None:
        test_each_loop_spawns_new_vllm_subprocess()


if __name__ == "__main__":
    unittest.main(verbosity=2)
