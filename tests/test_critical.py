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
    TOKENIZER_TEST_TEXTS,
    DoryConfig,
    DoryForCausalLM,
    init_new_tokens_embeddings,
    logit_lens_per_loop,
    set_dory_loops,
    verify_tokenizer_equivalence,
    verify_vllm_hf_equivalence,
)
from lib.evals import VLLM_EVAL_BATCH_SIZE, _run_problems_loop, build_lcb_quick_split
from lib.gen import (
    DORY_VLLM_OVERRIDE_KEYS,
    build_dory_hf_overrides,
    count_cot_tokens,
    generate,
    split_cot_and_answer,
)
from lib.hw import detect_hardware_profile, get_hw_config
from lib.io import CheckpointManager, ResumableProgress, ensure_model_files, setup_drive_and_cache
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

    # CoT truncada com código correto dentro do raciocínio: não passa (senão o GRPO premia truncar)
    truncated = "add a and b ```python\ndef add(a, b):\n    return a + b\n```"  # <think> veio no prompt
    r_trunc = compute_reward(
        truncated, test_code=test_suite, tokenizer=tokenizer, lam=0.1, l_ref=100.0, prompt_ended_with_think=True
    )
    assert r_trunc.passed_tests is False

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


def test_logit_lens_last_loop_with_output_block_matches_forward() -> None:
    """Com o bloco de saída (padrão), o logit lens do ÚLTIMO loop reproduz o forward normal:
    mesmo top-1 (e mesmas probabilidades) nas últimas posições. Sem o bloco, não reproduz."""
    model, tokenizer = _make_tiny_dory(n_loops=1)
    prompt = "step 1 we need to add two numbers"
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    n_pos = 5
    k_loops = 3
    set_dory_loops(model, k_loops)
    model.eval()
    with torch.no_grad():
        # O modelo minúsculo tem logits quase planos (std ~0.02); amplifica o lm_head para que o top-1 e
        # as probabilidades sejam discriminativos (com o bloco de saída ignorado o teste falharia).
        model.lm_head.weight.mul_(300.0)
        ref_logits = model(input_ids=input_ids, n_loops=k_loops).logits[0, -n_pos:, :].float()
    ref_probs = torch.softmax(ref_logits, dim=-1)
    ref_top1 = ref_probs.argmax(dim=-1).tolist()
    ref_top1_prob = ref_probs.max(dim=-1).values.tolist()

    import inspect

    assert inspect.signature(logit_lens_per_loop).parameters["apply_output_block"].default is True
    lens = logit_lens_per_loop(model, tokenizer, prompt, n_loops=k_loops, top_k=2, last_n_positions=n_pos)
    last = lens[-1]
    assert last["loop_idx"] == k_loops
    assert [pos["top_token_ids"][0] for pos in last["positions"]] == ref_top1
    for pos, ref_p in zip(last["positions"], ref_top1_prob):
        assert abs(pos["top_probs"][0] - ref_p) < 1e-4

    raw = logit_lens_per_loop(
        model, tokenizer, prompt, n_loops=k_loops, top_k=2, last_n_positions=n_pos, apply_output_block=False
    )
    raw_p = [pos["top_probs"][0] for pos in raw[-1]["positions"]]
    assert any(abs(a - b) > 1e-4 for a, b in zip(raw_p, ref_top1_prob)), (
        "sem o bloco de saída o lens não deveria reproduzir o forward"
    )


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


def _install_fake_vllm(llm_instances: list[Any]) -> Any:
    """Context manager: instala um módulo `vllm` FALSO em `sys.modules` (hermético: igual com ou sem
    vllm real instalado, como no Colab). `LLM(...)` levanta erro se for instanciado e registra a tentativa."""
    import sys as _sys
    import types as _types

    class _FakeSamplingParams:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class _ExplodingLLM:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            llm_instances.append((args, kwargs))
            raise AssertionError("vllm.LLM NÃO deveria ser instanciado neste teste!")

    fake = _types.ModuleType("vllm")
    fake.SamplingParams = _FakeSamplingParams  # type: ignore[attr-defined]
    fake.LLM = _ExplodingLLM  # type: ignore[attr-defined]

    import unittest.mock as _mock

    return _mock.patch.dict(_sys.modules, {"vllm": fake})


def test_vllm_failure_raises_when_allow_hf_fallback_false() -> None:
    tokenizer = DeterministicWordTokenizer()
    llm_instances: list[Any] = []

    class FailingMockVLLMEngine:
        def generate(self, prompts: list[str], sampling_params: Any) -> Any:
            raise RuntimeError("Simulated vLLM engine failure")

    # vllm falso (com LLM explosivo) instalado: o teste não depende de o vllm real existir ou não.
    with _install_fake_vllm(llm_instances):
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
    assert llm_instances == [], "um callable jamais pode criar um motor vLLM"


def test_callable_and_hf_model_never_become_vllm_model_name() -> None:
    """Com `vllm` (falso) instalado, callables/mocks e modelos HF sem nome válido NÃO instanciam o LLM:
    com allow_hf_fallback=True gravam backend='hf'; com False levantam erro claro."""
    from lib.gen import _resolve_vllm_model_name

    tokenizer = DeterministicWordTokenizer()
    llm_instances: list[Any] = []

    def mock_callable(prompt: str, **kwargs: Any) -> str:
        return "<think> a b </think> ok"

    model, model_tok = _make_tiny_dory(n_loops=1)  # config._name_or_path vazio: sem checkpoint nomeado

    with _install_fake_vllm(llm_instances):
        # callable e modelo HF (sem nome válido) + fallback permitido -> HF, sem instanciar o LLM
        outs = generate(mock_callable, ["x"], tokenizer=tokenizer, backend="vllm", allow_hf_fallback=True)
        assert [o.backend for o in outs] == ["hf"]
        outs_m = generate(
            model, ["step 1 add"], max_new_tokens=4, tokenizer=model_tok, backend="vllm", allow_hf_fallback=True
        )
        assert outs_m and all(o.backend == "hf" for o in outs_m)
        # fallback proibido -> erro claro, também sem instanciar o LLM
        for obj, tok in ((mock_callable, tokenizer), (model, model_tok)):
            try:
                generate(obj, ["x"], tokenizer=tok, backend="vllm", allow_hf_fallback=False)
                raise AssertionError("deveria ter levantado erro")
            except RuntimeError as exc:
                assert "vLLM" in str(exc)
        assert llm_instances == [], "LLM não pode ser instanciado para callable/modelo sem nome válido"

    # resolução de nomes: str passa; callable -> TypeError; modelo HF só com repo/caminho válido
    assert _resolve_vllm_model_name("org/modelo") == "org/modelo"
    for bad in (mock_callable, object(), 123):
        try:
            _resolve_vllm_model_name(bad)
            raise AssertionError("deveria ter levantado TypeError")
        except TypeError:
            pass
    try:
        _resolve_vllm_model_name(model)
        raise AssertionError("modelo sem nome válido deveria levantar ValueError")
    except ValueError:
        pass
    model.config._name_or_path = "org/Modelo-4B"
    assert _resolve_vllm_model_name(model) == "org/Modelo-4B"
    with tempfile.TemporaryDirectory() as tmp:
        model.config._name_or_path = tmp
        assert _resolve_vllm_model_name(model) == tmp
    model.config._name_or_path = "nao/e/repo/valido"
    try:
        _resolve_vllm_model_name(model)
        raise AssertionError("nome inválido deveria levantar ValueError")
    except ValueError:
        pass


def test_from_pretrained_roundtrip_preserves_logits() -> None:
    """Regressão: no transformers 5.x, buffers não persistentes (ex.: inv_freq do RoPE) ficavam com lixo
    após from_pretrained, desligando o RoPE em silêncio (teste de equivalência deu 70% de divergência).
    Salvar e recarregar o modelo deve reproduzir exatamente os mesmos logits."""
    if torch is None:
        return
    model, tokenizer = _make_tiny_dory(n_loops=2)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    input_ids = tokenizer("step 1 we need to add two numbers and then return", return_tensors="pt")["input_ids"]
    with torch.no_grad():
        ref = model(input_ids=input_ids).logits
    with tempfile.TemporaryDirectory() as tmp:
        cfg_cls = type(model.config)
        orig_validate = getattr(cfg_cls, "validate", None)
        cfg_cls.validate = lambda self: None  # transformers 5.x rejeita layer_types == "mlp" ao salvar
        try:
            model.save_pretrained(tmp)
            loaded = DoryForCausalLM.from_pretrained(tmp).eval()
        finally:
            if orig_validate is not None:
                cfg_cls.validate = orig_validate
        with torch.no_grad():
            out = loaded(input_ids=input_ids).logits
    assert torch.allclose(ref, out, atol=1e-5), f"max diff = {(ref - out).abs().max().item()}"


def test_hf_cache_never_points_to_drive_and_ensure_model_files() -> None:
    """Garante que nenhuma variável de cache do HF aponta para /content/drive, que hf_cache foi
    removido dos caminhos, e que ensure_model_files rebaixa arquivos JSON corrompidos com force_download=True.
    """
    import json
    import os

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        stale_cache = tmp_path / "hf_cache"
        stale_cache.mkdir(parents=True, exist_ok=True)
        (stale_cache / "corrupted.json").write_text("{bad", encoding="utf-8")

        # Hermético: no Colab `is_colab()` é True e o código usaria o Drive (montagem + pasta real) em vez
        # do `local_fallback_root` temporário. Força o ramo local e restaura as variáveis de ambiente ao sair.
        import unittest.mock as _mock

        with _mock.patch("lib.io.is_colab", return_value=False), _mock.patch.dict(os.environ):
            paths = setup_drive_and_cache(
                drive_root="/content/drive/MyDrive/coder-comprimido",
                local_fallback_root=tmp_path,
            )

            assert "hf_cache" not in paths
            for key, val in paths.items():
                assert "hf_cache" not in str(val), f"Caminho '{key}' não pode conter hf_cache: {val}"

            hf_cache_vars = ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE")
            for var in hf_cache_vars:
                val_str = os.environ.get(var, "")
                assert "/content/drive" not in val_str.replace("\\", "/"), (
                    f"Variável {var}={val_str!r} jamais pode apontar para /content/drive!"
                )

            assert os.environ["HF_HOME"] == "/root/.cache/huggingface"
            assert os.environ["HF_HUB_CACHE"] == "/root/.cache/huggingface/hub"
            assert os.environ["HF_DATASETS_CACHE"] == "/root/.cache/huggingface/datasets"
            assert not stale_cache.exists()

        # Test ensure_model_files with corrupted tokenizer.json
        model_dir = tmp_path / "hub_mock"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text('{"model_type": "dory"}', encoding="utf-8")
        (model_dir / "tokenizer_config.json").write_text('{"tokenizer_class": "PreTrainedTokenizerFast"}', encoding="utf-8")
        (model_dir / "tokenizer.json").write_text('{"version": "1.0", "trunc', encoding="utf-8")

        download_calls: list[tuple[str, str, bool]] = []

        def fake_download(repo: str, filename: str, force_download: bool = False) -> str:
            download_calls.append((repo, filename, bool(force_download)))
            target_file = model_dir / filename
            if force_download and filename == "tokenizer.json":
                target_file.write_text('{"version": "1.0", "model": {"type": "BPE"}}', encoding="utf-8")
            return str(target_file)

        verified = ensure_model_files("nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16", download_fn=fake_download)
        assert set(verified.keys()) == {"config.json", "tokenizer_config.json", "tokenizer.json"}
        for p in verified.values():
            with p.open("r", encoding="utf-8") as f:
                assert isinstance(json.load(f), dict)

        assert ("nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16", "tokenizer.json", False) in download_calls
        assert ("nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16", "tokenizer.json", True) in download_calls


def test_verify_vllm_hf_equivalence_params_and_vllm_batches_of_128() -> None:
    """Verifica parâmetros gpu_memory_utilization=0.60 e max_model_len=2048 (>=1024) em verify_vllm_hf_equivalence,
    lotes de 128 com checkpoint por lote no vLLM, e guarda __main__ em todos os scripts de scripts/.
    """
    import ast
    import inspect

    sig = inspect.signature(verify_vllm_hf_equivalence)
    assert sig.parameters["gpu_memory_utilization"].default == 0.60
    assert sig.parameters["max_model_len"].default == 2048

    # max_model_len < 1024 (sliding window) must raise ValueError
    raised_short_ctx = False
    try:
        verify_vllm_hf_equivalence(prompts=["hi"], loops=(1,), max_model_len=512)
    except ValueError:
        raised_short_ctx = True
    assert raised_short_ctx is True, "max_model_len < 1024 deve levantar ValueError!"

    # Verify VLLM_EVAL_BATCH_SIZE == 128 and checkpointing per batch of 128
    assert VLLM_EVAL_BATCH_SIZE == 128

    batch_sizes_seen: list[int] = []
    checkpoint_counts_after_each_batch: list[int] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cache_dir = tmp_path / "eval_cache"
        results_path = tmp_path / "results" / "results.jsonl"
        run_id = "test_vllm_batch128"
        bench_name = "synthetic_bench"
        cache_file = cache_dir / f"{run_id}_{bench_name}_quick.jsonl"

        class BatchTrackingVLLMEngine:
            def generate(self, prompts: list[str], sampling_params: Any) -> list[Any]:
                # Before generating the next batch, check how many items are already checkpointed on disk
                if cache_file.exists():
                    lines = [ln for ln in cache_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
                    checkpoint_counts_after_each_batch.append(len(lines))
                else:
                    checkpoint_counts_after_each_batch.append(0)
                batch_sizes_seen.append(len(prompts))

                class _Sample:
                    def __init__(self) -> None:
                        self.text = "<think> step 1 </think>\n```python\ndef f(x):\n    return x + 1\n```"
                        self.token_ids = [3, 5, 6, 4, 25, 26]

                class _ReqOut:
                    def __init__(self) -> None:
                        self.outputs = [_Sample()]

                return [_ReqOut() for _ in prompts]

        # Provide a lightweight mock vllm.SamplingParams if vllm is not installed locally
        import sys as _sys
        import types as _types

        had_vllm = "vllm" in _sys.modules
        if not had_vllm:
            fake_vllm_mod = _types.ModuleType("vllm")

            class _FakeSamplingParams:
                def __init__(self, **kwargs: Any) -> None:
                    self.kwargs = kwargs

            fake_vllm_mod.SamplingParams = _FakeSamplingParams  # type: ignore[attr-defined]
            _sys.modules["vllm"] = fake_vllm_mod

        try:
            problems = [
                {
                    "task_id": f"task_{i}",
                    "prompt": "def f(x):",
                    "test_code": None,
                    "entry_point": None,
                }
                for i in range(300)
            ]
            tok = DeterministicWordTokenizer()
            _run_problems_loop(
                problems=problems,
                model=BatchTrackingVLLMEngine(),
                tokenizer=tok,
                run_id=run_id,
                phase="F0",
                model_name="dory-4b",
                checkpoint="base",
                n_loops=2,
                eval_tier="quick",
                bench_name=bench_name,
                max_new_tokens=64,
                temperature=0.0,
                batch_size=4,
                cache_dir=cache_dir,
                results_path=results_path,
                smoke=False,
                smoke_n=5,
                force=True,
                backend="vllm",
            )
        finally:
            if not had_vllm:
                _sys.modules.pop("vllm", None)

        assert batch_sizes_seen == [128, 128, 44], f"Esperado lotes [128, 128, 44], obtido {batch_sizes_seen}"
        assert checkpoint_counts_after_each_batch == [0, 128, 256]

    # Verify all scripts in scripts/ have main() + if __name__ == "__main__": and no loose top-level code
    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    script_files = list(scripts_dir.glob("*.py"))
    assert len(script_files) >= 4
    for sf in script_files:
        tree = ast.parse(sf.read_text(encoding="utf-8"), filename=str(sf))
        has_main_def = False
        has_main_guard = False
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "main":
                has_main_def = True
            elif isinstance(node, ast.If):
                # Must be `if __name__ == "__main__":`
                test_src = ast.unparse(node.test)
                assert "__name__" in test_src and "__main__" in test_src, (
                    f"If no nível de módulo em {sf.name} não é guarda __main__: {test_src}"
                )
                has_main_guard = True
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                # Module docstring is allowed
                continue
            elif isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            else:
                raise AssertionError(
                    f"Código solto no nível do módulo detectado em {sf.name}: {ast.dump(node)}"
                )
        assert has_main_def, f"Script {sf.name} precisa definir def main()"
        assert has_main_guard, f"Script {sf.name} precisa ter if __name__ == '__main__':"


def test_tokenizer_20_texts_hf_vs_vllm_and_fix_mistral_regex() -> None:
    """Testa que TOKENIZER_TEST_TEXTS contém 20 textos variados (código, inglês, espaços e quebras de linha)
    e que verify_tokenizer_equivalence detecta divergência de regex entre HF e vLLM e ativa fix_mistral_regex=True.
    """
    assert len(TOKENIZER_TEST_TEXTS) == 20
    # Ensure coverage of code, English, multiple spaces, tabs, and newlines
    assert any("def " in t for t in TOKENIZER_TEST_TEXTS)
    assert any("\t" in t for t in TOKENIZER_TEST_TEXTS)
    assert any("\n\n" in t for t in TOKENIZER_TEST_TEXTS)
    assert any("don't" in t or "it's" in t for t in TOKENIZER_TEST_TEXTS)

    class RegexSensitiveMockTokenizer:
        def __init__(self, fix_mistral_regex: bool = False) -> None:
            self.fix_mistral_regex = bool(fix_mistral_regex)
            self.pad_token = "<pad>"
            self.eos_token = "<eos>"

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            if self.fix_mistral_regex:
                # Canonical tokenization matching vLLM with fixed Mistral regex
                return [ord(c) % 97 for c in text]
            # Unfixed regex bug splits contractions/newlines/spaces differently
            if "  " in text or "\n" in text or "'" in text:
                return [(ord(c) + 13) % 97 for c in text]
            return [ord(c) % 97 for c in text]

    vllm_tok = RegexSensitiveMockTokenizer(fix_mistral_regex=True)

    def mock_hf_loader(repo: str, fix_mistral_regex: bool | None = None) -> Any:
        return RegexSensitiveMockTokenizer(fix_mistral_regex=bool(fix_mistral_regex))

    report = verify_tokenizer_equivalence(
        "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16",
        texts=TOKENIZER_TEST_TEXTS,
        hf_tokenizer=RegexSensitiveMockTokenizer(fix_mistral_regex=False),
        vllm_tokenizer=vllm_tok,
        hf_tokenizer_loader=mock_hf_loader,
    )
    assert report["n_texts"] == 20
    assert report["divergences_initial"] > 0
    assert report["divergences_final"] == 0
    assert report["matched"] is True
    assert report["fix_mistral_regex"] is True


def test_dory_cached_decode_matches_uncached() -> None:
    """Regressão: o generate() do transformers 5.x não suportava o DoryLoopCache (erro de forma na
    atenção). O laço próprio com cache deve gerar exatamente os mesmos tokens que a geração sem cache,
    inclusive em lote com padding à esquerda."""
    from lib.gen import _dory_decode

    model, tokenizer = _make_tiny_dory(n_loops=2)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    a = tokenizer("step 1 we need to add two numbers", return_tensors="pt")["input_ids"]
    b = tokenizer("return the sum", return_tensors="pt")["input_ids"]
    with torch.no_grad():
        def _ref(x):
            seq = x
            for _ in range(8):
                nxt = model(input_ids=seq, use_cache=False).logits[:, -1, :].argmax(-1, keepdim=True)
                seq = torch.cat([seq, nxt], dim=1)
            return seq[0, -8:].tolist()

        ref_a, ref_b = _ref(a), _ref(b)
        pad = a.shape[1] - b.shape[1]
        batch = torch.cat([a, torch.cat([torch.zeros(1, pad, dtype=torch.long), b], 1)], 0)
        mask = torch.cat(
            [torch.ones_like(a), torch.cat([torch.zeros(1, pad, dtype=torch.long), torch.ones_like(b)], 1)], 0
        )
        out = _dory_decode(
            model, batch, mask, max_new_tokens=8, temperature=0.0, top_p=1.0,
            eos_ids=set(), pad_id=0, n_loops=None,
        )
    assert out[0, -8:].tolist() == ref_a
    assert out[1, -8:].tolist() == ref_b


# ==============================================================================
# 10. Protocolos oficiais: SciCode (alvos do h5), LiveCodeBench (todos os testes) e tokenizer vLLM
# ==============================================================================
def _make_synthetic_scicode(tmp_path: Path, *, target_add_1: float = 3.0) -> tuple[dict[str, Any], Path]:
    """Problema sintético de 2 passos + h5 sintético com a MESMA estrutura do `test_data.h5` oficial
    (`{step_id}/test{n}/{var}`; um grupo com 1 variável vira valor, com 2+ variáveis vira tupla)."""
    import h5py  # type: ignore[import-not-found]
    import numpy as np

    tmp_path.mkdir(parents=True, exist_ok=True)
    h5_path = tmp_path / "test_data.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("1.1/test1/var1", data=target_add_1)  # add(1, 2)
        f.create_dataset("1.1/test2/var1", data=4.0)  # add(2, 2)
        f.create_dataset("1.2/test1/var1", data=6.0)  # double_add(1, 2)
        f.create_dataset("1.2/test2/var1", data=np.array([4.0, 6.0]))  # tupla (var1, var2)
        f.create_dataset("1.2/test2/var2", data=np.array([1.0, 2.0]))
    prob = {
        "problem_id": "1",
        "required_dependencies": "import numpy as np",
        "sub_steps": [
            {
                "step_number": "1.1",
                "step_description_prompt": "Implement add(a, b) returning the sum.",
                "step_background": "Sum background.",
                "function_header": "def add(a, b):\n    # Return a + b.",
                "return_line": "    return c",
                "ground_truth_code": "def add(a, b):\n    return a + b",
                "test_cases": ["assert add(1, 2) == target", "assert add(2, 2) == target"],
            },
            {
                "step_number": "1.2",
                "step_description_prompt": "Implement double_add using add.",
                "step_background": "Double background.",
                "function_header": "def double_add(a, b):\n    # Return 2 * add(a, b).",
                "return_line": "    return d",
                "ground_truth_code": "def double_add(a, b):\n    return 2 * add(a, b)",
                "test_cases": [
                    "assert double_add(1, 2) == target",
                    "from scicode.compare.cmp import cmp_tuple_or_list\n"
                    "assert cmp_tuple_or_list((np.array([double_add(1, 2), 6.0]) * np.array([2 / 3, 1.0]), "
                    "np.array([1.0, 2.0])), target)",
                ],
            },
        ],
    }
    return prob, h5_path


def _scicode_run_step(prob: dict[str, Any], h5: Path, responses: dict[int, str], idx: int) -> Any:
    """Gera prompt+script do passo `idx` (0-based) com as respostas do 'modelo' nos passos anteriores."""
    from lib.scicode_eval import (
        build_scicode_prompt,
        build_scicode_test_script,
        extract_python_script,
        previous_codes,
        run_scicode_script,
    )

    prev = previous_codes(prob, idx + 1, lambda i: extract_python_script(responses[i]))
    prompt, prefix = build_scicode_prompt(prob, idx + 1, prev)
    script = build_scicode_test_script(prob, idx + 1, prefix, responses[idx], h5)
    return prompt, script, run_scicode_script(script, timeout_s=120)


def test_scicode_official_protocol_with_h5_targets() -> None:
    pytest = __import__("pytest")
    pytest.importorskip("h5py")
    pytest.importorskip("scipy")
    pytest.importorskip("sympy")
    from lib.scicode_eval import is_skipped_step, load_scicode_targets, previous_codes

    good1 = "```python\ndef add(a, b):\n    return a + b\n```"
    bad1 = "```python\ndef add(a, b):\n    return a - b\n```"
    good2 = "```python\n# Background: soma dupla\ndef double_add(a, b):\n    return 2 * add(a, b)\n```"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        prob, h5 = _make_synthetic_scicode(tmp_path / "ok")

        # (c) os targets vêm do h5, lidos pelo mesmo loader do oficial
        t11 = load_scicode_targets("1.1", 2, h5)
        assert [float(x) for x in t11] == [3.0, 4.0]
        t12 = load_scicode_targets("1.2", 2, h5)
        assert float(t12[0]) == 6.0 and isinstance(t12[1], tuple) and len(t12[1]) == 2

        # (a) solução correta passa no passo 1 (compara com o target do h5)
        prompt1, script1, res1 = _scicode_run_step(prob, h5, {0: good1}, 0)
        assert res1.passed, res1.stderr
        assert "process_hdf5_to_tuple('1.1', 2," in script1 and "target = targets[1]" in script1
        assert "import numpy as np" in script1  # required_dependencies
        # prompt segue o template oficial (sem background anotado): header, descrição e dependências
        assert "def add(a, b):" in prompt1 and "Implement add(a, b) returning the sum." in prompt1
        assert "Sum background." not in prompt1 and "Background: " in prompt1
        assert "DEPENDENCIES:" in prompt1 and "import numpy as np" in prompt1

        # (b) solução errada falha
        _, _, res_bad = _scicode_run_step(prob, h5, {0: bad1}, 0)
        assert res_bad.passed is False and res_bad.exit_code != 0

        # passo 2 usa o código do PRÓPRIO modelo no passo 1: o prompt traz o código anterior
        prompt2, _, res2 = _scicode_run_step(prob, h5, {0: good1, 1: good2}, 1)
        assert "return a + b" in prompt2 and "Implement add(a, b) returning the sum." in prompt2
        # o passo 1.2 também confere uma TUPLA vinda do h5 via `scicode.compare.cmp` vendorizado
        assert res2.passed, res2.stderr
        # passo 1 errado propaga para o passo 2 (o código anterior vem do modelo, não do gold)
        _, _, res2_bad_dep = _scicode_run_step(prob, h5, {0: bad1, 1: good2}, 1)
        assert res2_bad_dep.passed is False

        # os targets realmente vêm do h5: com outro valor no h5, a solução correta falha
        prob_w, h5_wrong = _make_synthetic_scicode(tmp_path / "wrong", target_add_1=99.0)
        _, _, res_wrong_target = _scicode_run_step(prob_w, h5_wrong, {0: good1}, 0)
        assert res_wrong_target.passed is False

        # sem o h5 todos os testes falham: o bug antigo (sem `target`) não passa despercebido
        _, _, res_missing = _scicode_run_step(prob, tmp_path / "nao_existe.h5", {0: good1}, 0)
        assert res_missing.passed is False

    # passos especiais do oficial (13.6, 62.1, 76.3) são pulados; o código deles vem dos arquivos oficiais
    assert is_skipped_step({"problem_id": "13"}, 5) and is_skipped_step({"problem_id": "62"}, 0)
    assert is_skipped_step({"problem_id": "76"}, 2) and not is_skipped_step({"problem_id": "76"}, 1)
    prob62 = {
        "problem_id": "62",
        "sub_steps": [{"function_header": "class Block:\n    def __init__(self, length):"}, {"function_header": "def f():"}],
    }
    assert "self.length" in previous_codes(prob62, 2, lambda i: "x")[0]


def test_scicode_gold_check_and_loop_waves_and_cache() -> None:
    """Loop em ondas: o prompt do passo 2 usa a resposta gerada para o passo 1, e o resultado leva as
    métricas oficiais e o protocolo no config (resultado de outro protocolo não é reaproveitado)."""
    pytest = __import__("pytest")
    pytest.importorskip("h5py")
    pytest.importorskip("scipy")
    pytest.importorskip("sympy")
    from unittest import mock

    from lib.gen import Output
    from lib.scicode_eval import (
        SCICODE_PROTOCOL_ID,
        build_scicode_problems,
        make_scicode_extra_metrics_fn,
        run_scicode_gold_check,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        prob, h5 = _make_synthetic_scicode(tmp_path / "h5")

        gold = run_scicode_gold_check([prob], h5, timeout_s=120)
        assert gold == {"n_steps": 2, "n_pass": 2, "failed": []}

        prompts_seen: list[str] = []

        def fake_generate(model, prompts, **kwargs):  # noqa: ANN001, ANN003
            outs = []
            for pr in prompts:
                prompts_seen.append(pr)
                if "Implement double_add" in pr.split("NEXT STEP")[1]:
                    ans = "```python\ndef double_add(a, b):\n    return 2 * add(a, b)\n```"
                else:
                    ans = "```python\ndef add(a, b):\n    return a + b\n```"
                outs.append(
                    Output(text=ans, cot_text="", answer_text=ans, n_cot_tokens=0, n_total_tokens=5, latency_s=0.0, backend="hf")
                )
            return outs

        problems = build_scicode_problems([prob], h5)
        assert [p["stage"] for p in problems] == [0, 1]
        results_path = tmp_path / "results.jsonl"
        common = dict(
            model=None, tokenizer=None, phase="F0", model_name="m", checkpoint="c", n_loops=2, eval_tier="gate",
            bench_name="scicode", max_new_tokens=16, temperature=0.0, batch_size=4, cache_dir=tmp_path / "cache",
            results_path=results_path, smoke=False, smoke_n=5, force=False, backend="hf",
            protocol_id=SCICODE_PROTOCOL_ID, cache_suffix="_official",
        )
        with mock.patch("lib.evals.generate", fake_generate):
            res = _run_problems_loop(
                problems=problems, run_id="r", extra_config_fn=make_scicode_extra_metrics_fn([prob]), **common
            )
        assert res["pass1"] == 1.0
        assert res["config"]["eval_protocol"] == SCICODE_PROTOCOL_ID
        assert res["config"]["scicode_main_rate_official"] == 1.0
        assert res["config"]["scicode_subproblem_rate_official"] == 1.0
        # onda 2: o prompt do passo 2 contém o código que o modelo gerou no passo 1
        assert len(prompts_seen) == 2 and "return a + b" in prompts_seen[1]

        # resultado de OUTRO protocolo (sem eval_protocol) no results.jsonl não é reaproveitado
        append_result(
            run_id="r2", phase="F0", model="m", checkpoint="c", n_loops=2, eval_tier="gate", bench="scicode",
            pass1=0.0, tokens_cot_mean=0.0, tokens_total_mean=0.0, latency_mean_s=0.0, config={},
            results_path=results_path,
        )
        with mock.patch("lib.evals.generate", fake_generate):
            res2 = _run_problems_loop(problems=build_scicode_problems([prob], h5), run_id="r2", **common)
        assert res2["pass1"] == 1.0 and res2["config"]["eval_protocol"] == SCICODE_PROTOCOL_ID


def test_scicode_ensure_test_data_errors_clearly_and_accepts_valid_file() -> None:
    pytest = __import__("pytest")
    pytest.importorskip("h5py")
    from unittest import mock

    from lib import scicode_eval

    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        import h5py  # type: ignore[import-not-found]
        import numpy as np

        with h5py.File(data_dir / "test_data.h5", "w") as f:
            f.create_dataset("1.1/test1/var1", data=np.zeros(300_000))
        # arquivo válido (acima do tamanho mínimo) é aceito sem baixar nada
        with mock.patch.object(scicode_eval, "SCICODE_H5_MIN_BYTES", 1000):
            assert scicode_eval.ensure_scicode_test_data(data_dir) == data_dir / "test_data.h5"

        # arquivo inválido + download que falha => RuntimeError com instrução de download manual
        (data_dir / "test_data.h5").write_bytes(b"<html>quota</html>")

        class FakeGdown:
            @staticmethod
            def download_folder(**kwargs):  # noqa: ANN003
                raise OSError("cota do Drive excedida")

        import types as _types

        def fake_hf(fn):  # noqa: ANN001
            m = _types.ModuleType("huggingface_hub")
            m.hf_hub_download = fn  # type: ignore[attr-defined]
            return m

        def hf_fails(*a, **k):  # noqa: ANN002, ANN003
            raise OSError("sem rede")

        # Drive e espelho falham => RuntimeError com instrução de download manual
        with mock.patch.dict("sys.modules", {"gdown": FakeGdown, "huggingface_hub": fake_hf(hf_fails)}):
            with pytest.raises(RuntimeError) as exc:
                scicode_eval.ensure_scicode_test_data(data_dir)
        assert "drive.google.com" in str(exc.value) and "test_data.h5" in str(exc.value)

        # Drive falha, espelho válido: só é aceito se o gold check passar
        mirror = Path(tmp) / "mirror" / "test_data.h5"
        mirror.parent.mkdir()
        with h5py.File(mirror, "w") as f:
            f.create_dataset("1.1/test1/var1", data=np.ones(300_000))
        mods = {"gdown": FakeGdown, "huggingface_hub": fake_hf(lambda *a, **k: str(mirror))}
        with mock.patch.object(scicode_eval, "SCICODE_H5_MIN_BYTES", 1000), mock.patch.dict("sys.modules", mods), \
                mock.patch.object(scicode_eval, "load_scicode_rows", lambda split: []):
            with mock.patch.object(scicode_eval, "run_scicode_gold_check", lambda *a, **k: {"n_steps": 50, "n_pass": 10, "failed": []}):
                with pytest.raises(RuntimeError):
                    scicode_eval.ensure_scicode_test_data(data_dir)
            assert not scicode_eval._h5_ok(data_dir / "test_data.h5")  # espelho reprovado não é gravado
            with mock.patch.object(scicode_eval, "run_scicode_gold_check", lambda *a, **k: {"n_steps": 50, "n_pass": 50, "failed": []}):
                assert scicode_eval.ensure_scicode_test_data(data_dir) == data_dir / "test_data.h5"
        with h5py.File(data_dir / "test_data.h5", "r") as f:
            assert float(f["1.1/test1/var1"][0]) == 1.0


def _lcb_io(pairs: list[tuple[str, str]]) -> list[dict[str, str]]:
    return [{"input": i, "output": o} for i, o in pairs]


def test_lcb_stdin_functional_and_all_tests() -> None:
    # --- stdin/stdout ---
    stdin_ok = "```python\nn = int(input())\nprint(n * 2)\n```"
    stdin_bad = "```python\nn = int(input())\nprint(n * 3)\n```"
    tests = _lcb_io([("1\n", "2\n"), ("5\n", "10\n")])
    assert check_solution(stdin_ok, io_examples=tests, timeout_s=6.0, max_memory_mb=1024).passed is True
    bad = check_solution(stdin_bad, io_examples=tests, timeout_s=6.0, max_memory_mb=1024)
    assert bad.passed is False and bad.error_type == "WrongAnswer"
    # outras formas de ler a entrada (sys.stdin.read) e `if __name__ == "__main__"`
    alt = (
        "```python\nimport sys\ndef main():\n    data = sys.stdin.read().split()\n    print(int(data[0]) * 2)\n"
        "if __name__ == '__main__':\n    main()\n```"
    )
    assert check_solution(alt, io_examples=tests, timeout_s=6.0, max_memory_mb=1024).passed is True
    # sem `import` de deque: o prelúdio de imports do oficial cobre
    prelude = "```python\nd = deque([int(input())])\nprint(d[0] * 2)\n```"
    assert check_solution(prelude, io_examples=tests, timeout_s=6.0, max_memory_mb=1024).passed is True
    # tolerância do oficial: espaços no fim da linha, linhas finais e números equivalentes (Decimal)
    tol = "```python\nprint('1.50   ')\nprint()\n```"
    assert check_solution(tol, io_examples=_lcb_io([("", "1.5\n\n\n")]), timeout_s=6.0).passed is True
    assert check_solution("```python\nprint('2 3')\n```", io_examples=_lcb_io([("", "2   3")]), timeout_s=6.0).passed is True
    assert check_solution("```python\nprint('a b')\n```", io_examples=_lcb_io([("", "a  c")]), timeout_s=6.0).passed is False

    # --- funcional / LeetCode (metadata.func_name, classe Solution) ---
    func_ok = "```python\nclass Solution:\n    def addTwo(self, a: int, b: list) -> list:\n        return [a + x for x in b]\n```"
    func_bad = "```python\nclass Solution:\n    def addTwo(self, a: int, b: list) -> list:\n        return [a - x for x in b]\n```"
    func_tests = _lcb_io([("1\n[1, 2]", "[2, 3]"), ("10\n[0]", "[10]")])
    assert check_solution(func_ok, io_examples=func_tests, fn_name="addTwo", timeout_s=6.0, max_memory_mb=1024).passed is True
    fres = check_solution(func_bad, io_examples=func_tests, fn_name="addTwo", timeout_s=6.0, max_memory_mb=1024)
    assert fres.passed is False and fres.error_type == "WrongAnswer"
    # método inexistente => falha; tupla no lugar de lista não é penalizada (oficial)
    assert check_solution(func_ok, io_examples=func_tests, fn_name="naoExiste", timeout_s=6.0).passed is False
    tup = "```python\nclass Solution:\n    def f(self, a):\n        return (a, a)\n```"
    assert check_solution(tup, io_examples=_lcb_io([("3", "[3, 3]")]), fn_name="f", timeout_s=6.0).passed is True

    # --- mais de 15 testes: só o 20º falha (prova que o limite de 15 foi removido) ---
    many = _lcb_io([(f"{i}\n", f"{i * 2}\n") for i in range(25)])
    all_good = "```python\nn = int(input())\nprint(n * 2)\n```"
    fail_at_20 = "```python\nn = int(input())\nprint(-1 if n == 19 else n * 2)\n```"
    assert check_solution(all_good, io_examples=many, timeout_s=6.0).passed is True
    r20 = check_solution(fail_at_20, io_examples=many, timeout_s=6.0)
    assert r20.passed is False and r20.error_type == "WrongAnswer" and "teste 19" in r20.stderr
    assert check_solution(fail_at_20, io_examples=many[:15], timeout_s=6.0).passed is True  # com o corte antigo passaria

    # --- para no primeiro erro (não executa o teste lento depois) e respeita o timeout por teste ---
    slow_after_fail = _lcb_io([("0\n", "wrong\n"), ("1\n", "x\n")])
    t0 = time.perf_counter()
    sleeper = "```python\nimport time\nn = int(input())\nif n == 1:\n    time.sleep(30)\nprint(n)\n```"
    assert check_solution(sleeper, io_examples=slow_after_fail, timeout_s=6.0).passed is False
    assert time.perf_counter() - t0 < 5.0
    t0 = time.perf_counter()
    tle = check_solution("```python\nwhile True:\n    pass\n```", io_examples=_lcb_io([("", "1")]), timeout_s=1.0)
    assert tle.passed is False and tle.timed_out is True and tle.error_type == "TimeoutError"
    assert time.perf_counter() - t0 < 6.0
    # limite global por problema
    from lib.sandbox import run_io_tests_in_sandbox

    t0 = time.perf_counter()
    glob = run_io_tests_in_sandbox(
        "import time\ntime.sleep(0.4)\nprint(1)\n", ["" for _ in range(40)], ["1" for _ in range(40)],
        timeout_s=2.0, total_timeout_s=2.0,
    )
    assert glob.passed is False and glob.timed_out is True
    assert time.perf_counter() - t0 < 8.0


def test_lcb_loader_decodes_private_tests_and_keeps_all_of_them() -> None:
    import base64
    import json
    import pickle
    import zlib
    from unittest import mock

    from lib import evals

    priv = [{"input": f"{i}\n", "output": f"{i * 2}\n", "testtype": "stdin"} for i in range(20)]
    # formato oficial: JSON (string) -> pickle -> zlib -> base64
    enc = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(priv)))).decode("utf-8")
    pub = [{"input": "100\n", "output": "200\n", "testtype": "stdin"}]
    rows = [
        {
            "question_id": "q1", "question_title": "t", "question_content": "c", "starter_code": "",
            "contest_date": "2025-02-01T00:00:00", "difficulty": "easy",
            "public_test_cases": json.dumps(pub), "private_test_cases": enc, "metadata": "{}",
        },
        {
            "question_id": "q2", "question_title": "t", "question_content": "c",
            "starter_code": "class Solution:\n    def f(self, a):\n        pass",
            "contest_date": "2025-01-01T00:00:00", "difficulty": "easy",
            "public_test_cases": json.dumps([{"input": "1", "output": "2", "testtype": "functional"}]),
            "private_test_cases": json.dumps([{"input": "2", "output": "3", "testtype": "functional"}]),
            "metadata": json.dumps({"func_name": "f"}),
        },
    ]
    assert len(evals._decode_lcb_test_cases(enc)) == 20  # antes do conserto: [] (pickle devolve uma str JSON)
    with mock.patch.object(evals, "_iter_lcb_rows", lambda tag: iter(rows)):
        items = evals.load_lcb_v6_dataset()
    assert len(items[0]["io_examples"]) == 21 and items[0]["fn_name"] is None  # 1 público + 20 privados
    assert len(items[1]["io_examples"]) == 2 and items[1]["fn_name"] == "f"
    assert all(it["lcb_schema"] == evals.LCB_SCHEMA_VERSION for it in items)

    # cache antigo (testes truncados, sem lcb_schema) é descartado e refeito
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "lcb_quick.json"
        cache.write_text(
            json.dumps([{"question_id": "velho", "contest_date": "2020-01-01", "io_examples": []}]), encoding="utf-8"
        )
        fresh = build_lcb_quick_split(cache, n_problems=2, preloaded_items=items)
        assert [x["question_id"] for x in fresh] == ["q1", "q2"]
        # cache já no formato novo é reaproveitado
        again = build_lcb_quick_split(cache, n_problems=2, preloaded_items=[])
        assert [x["question_id"] for x in again] == ["q1", "q2"]


def test_tokenizer_report_is_unverified_when_vllm_missing() -> None:
    import sys as _sys
    import types
    from unittest import mock

    from lib.dory import format_tokenizer_report

    class MockTok:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [ord(c) % 97 for c in text]

    # vllm ausente: `None` em sys.modules faz o import levantar ImportError
    names = ("vllm", "vllm.tokenizers", "vllm.tokenizers.registry", "vllm.transformers_utils", "vllm.transformers_utils.tokenizer")
    with mock.patch.dict(_sys.modules, {k: None for k in names}):
        report = verify_tokenizer_equivalence("x/y", texts=TOKENIZER_TEST_TEXTS, hf_tokenizer=MockTok())
    assert report["verified"] is False and report["matched"] is None
    assert report["divergences_initial"] is None and report["details"] == []
    assert len(report["vllm_import_errors"]) == 4
    line = format_tokenizer_report(report)
    assert "verified=False" in line and "matched=None" in line and "matched=True" not in line

    # ordem de tentativa: vllm.tokenizers.get_tokenizer primeiro
    calls: list[str] = []
    mod_new = types.ModuleType("vllm.tokenizers")
    mod_new.get_tokenizer = lambda name, *a, **k: (calls.append("new"), MockTok())[1]  # type: ignore[attr-defined]
    mod_old = types.ModuleType("vllm.transformers_utils.tokenizer")
    mod_old.get_tokenizer = lambda name, *a, **k: (calls.append("old"), MockTok())[1]  # type: ignore[attr-defined]
    fake = {"vllm": types.ModuleType("vllm"), "vllm.tokenizers": mod_new, "vllm.transformers_utils.tokenizer": mod_old}
    with mock.patch.dict(_sys.modules, fake):
        rep_new = verify_tokenizer_equivalence("x/y", texts=TOKENIZER_TEST_TEXTS, hf_tokenizer=MockTok())
    assert calls == ["new"] and rep_new["verified"] is True and rep_new["matched"] is True
    assert rep_new["vllm_tokenizer_source"] == "vllm.tokenizers.get_tokenizer"

    # sem vllm.tokenizers.get_tokenizer: cai para registry.cached_get_tokenizer e, por fim, o caminho antigo
    reg = types.ModuleType("vllm.tokenizers.registry")
    reg.cached_get_tokenizer = lambda name, *a, **k: MockTok()  # type: ignore[attr-defined]
    only_reg = {
        "vllm": types.ModuleType("vllm"),
        "vllm.tokenizers": types.ModuleType("vllm.tokenizers"),
        "vllm.tokenizers.registry": reg,
    }
    with mock.patch.dict(_sys.modules, only_reg):
        rep_reg = verify_tokenizer_equivalence("x/y", texts=TOKENIZER_TEST_TEXTS, hf_tokenizer=MockTok())
    assert rep_reg["vllm_tokenizer_source"] == "vllm.tokenizers.registry.cached_get_tokenizer"
    only_old = {
        "vllm": types.ModuleType("vllm"),
        "vllm.tokenizers": None,
        "vllm.tokenizers.registry": None,
        "vllm.transformers_utils.tokenizer": mod_old,
    }
    with mock.patch.dict(_sys.modules, only_old):
        rep_old = verify_tokenizer_equivalence("x/y", texts=TOKENIZER_TEST_TEXTS, hf_tokenizer=MockTok())
    assert rep_old["vllm_tokenizer_source"] == "vllm.transformers_utils.tokenizer.get_tokenizer"


# ==============================================================================
# Novos testes: sandbox sem tracing, EvalPlus oficial, prompt LCB, LCB Pro, tokenizer, custo, truncamento
# ==============================================================================
def test_sandbox_has_no_line_tracing_and_runs_cpu_heavy_code_fast() -> None:
    """O sandbox usa limites do SO (resource) e timeout de processo, sem sys.settrace/tracemalloc por linha
    (que deixavam o código lento e reprovavam soluções corretas por timeout)."""
    import lib.sandbox as sbx

    for tpl in (sbx._RUNNER_TEMPLATE, sbx._IO_RUNNER_TEMPLATE, sbx._EVALPLUS_RUNNER_TEMPLATE):
        assert "settrace" not in tpl and "tracemalloc" not in tpl

    heavy = "total = 0\nfor i in range(3_000_000):\n    total += i * i % 7\nassert total > 0\n"
    t0 = time.perf_counter()
    res = run_code_in_sandbox(heavy, timeout_s=10.0, max_memory_mb=256)
    assert res.passed is True and res.timed_out is False, res.stderr
    assert time.perf_counter() - t0 < 8.0

    # estouro de memória continua sendo detectado (RLIMIT_AS): MemoryError no stderr
    mem = run_code_in_sandbox("x = bytearray(400 * 1024 * 1024)\n", timeout_s=5.0, max_memory_mb=64)
    assert mem.passed is False and mem.memory_exceeded is True and mem.error_type == "MemoryError"


def test_evalplus_differential_base_and_plus_semantics() -> None:
    """Protocolo EvalPlus oficial: teste diferencial contra a solução canônica em base + plus,
    tolerância de float (atol), oráculos especiais e limite de tempo por teste."""
    from lib.sandbox import check_evalplus_solution, compute_evalplus_groundtruth

    canonical = "def add(a, b):\n    return a + b\n"
    base = [[1, 2]]
    plus = [[1, 2], [10**6, 5], [200, 1]]
    gt = compute_evalplus_groundtruth(canonical, entry_point="add", base_input=base, plus_input=plus)
    assert gt["base"] == [3] and gt["plus"] == [3, 1000005, 201] and len(gt["plus_time"]) == 3

    def run(code: str, **kw: Any) -> Any:
        return check_evalplus_solution(
            code, dataset="humaneval", entry_point="add", base_input=base, plus_input=plus,
            groundtruth=gt, max_memory_mb=512, **kw,
        )

    assert run("```python\ndef add(a, b):\n    return a + b\n```").passed is True
    # passa nos testes base, mas só os testes PLUS pegam o erro (a=200): HumanEval comum aprovaria
    only_plus = "def add(a, b):\n    return a + b if a < 150 else 0\n"
    assert run(only_plus, base_only=True).passed is True
    r = run(only_plus)
    assert r.passed is False and r.error_type == "WrongAnswer"
    # laço infinito numa entrada plus -> timeout por teste (max(1 s, 4 x tempo canônico)) e processo morto
    t0 = time.perf_counter()
    r = run("def add(a, b):\n    while a > 150:\n        pass\n    return a + b\n")
    assert r.passed is False and r.timed_out is True and time.perf_counter() - t0 < 15
    # exit() é desativado pelo reliability_guard oficial => falha, sem derrubar o avaliador
    assert run("def add(a, b):\n    exit()\n").passed is False
    assert run("```python\n```").error_type == "EmptyCode"

    # floats: sem atol, o oficial força atol=1e-6 quando o esperado é float
    gt_f = compute_evalplus_groundtruth(
        "def half(x):\n    return x / 3\n", entry_point="half", base_input=[[1.0]], plus_input=[[2.0], [9.0]]
    )
    kwf = dict(dataset="humaneval", entry_point="half", base_input=[[1.0]], plus_input=[[2.0], [9.0]], groundtruth=gt_f)
    assert check_evalplus_solution("def half(x):\n    return x / 3 + 1e-9\n", **kwf).passed is True
    assert check_evalplus_solution("def half(x):\n    return x / 3 + 1e-3\n", **kwf).passed is False

    # oráculo especial do MBPP: comparação como conjunto (similar_elements)
    gt_s = compute_evalplus_groundtruth(
        "def similar_elements(a, b):\n    return tuple(set(a) & set(b))\n",
        entry_point="similar_elements", base_input=[[[3, 4, 5, 6], [5, 7, 4, 10]]], plus_input=[[[1, 2], [2, 1]]],
    )
    kws = dict(dataset="mbpp", entry_point="similar_elements", base_input=[[[3, 4, 5, 6], [5, 7, 4, 10]]],
               plus_input=[[[1, 2], [2, 1]]], groundtruth=gt_s)
    assert check_evalplus_solution(
        "def similar_elements(a, b):\n    return sorted(set(a) & set(b), reverse=True)\n", **kws
    ).passed is True


def test_evalplus_loader_uses_plus_inputs_not_base_only_test() -> None:
    """Com o pacote `evalplus`, o problema usa base_input + plus_input (grader), NÃO o campo `test`/`assertion`
    (que só traz os testes originais do HumanEval/MBPP)."""
    import sys as _sys
    import types as _types
    import unittest.mock as _mock

    from lib import evals

    item = {
        "prompt": "def add(a, b):\n    '''soma'''\n",
        "entry_point": "add",
        "canonical_solution": "    return a + b\n",
        "test": "def check(candidate):\n    assert True\n",  # teste ORIGINAL (base): não pode ser usado
        "base_input": [[1, 2]],
        "plus_input": [[1, 2], [200, 1]],
        "atol": 0,
    }
    fake_data = _types.ModuleType("evalplus.data")
    fake_data.get_human_eval_plus = lambda: {"HumanEval/0": item}  # type: ignore[attr-defined]
    fake_pkg = _types.ModuleType("evalplus")
    fake_pkg.data = fake_data  # type: ignore[attr-defined]
    with _mock.patch.dict(_sys.modules, {"evalplus": fake_pkg, "evalplus.data": fake_data}):
        problems = evals._load_evalplus_problems("humaneval+")
    assert len(problems) == 1 and problems[0].get("grader") is not None and "test_code" not in problems[0]
    good, extra = problems[0]["grader"]("```python\ndef add(a, b):\n    return a + b\n```", None)
    assert good.passed is True and extra["eval_protocol"] == "evalplus_base_plus"
    bad, _ = problems[0]["grader"]("```python\ndef add(a, b):\n    return a + b if a < 100 else 0\n```", None)
    assert bad.passed is False


def test_lcb_prompt_matches_official_template_and_uses_system_message() -> None:
    from lib.evals import LCB_SYSTEM_MESSAGE, _build_lcb_prompt

    assert LCB_SYSTEM_MESSAGE == (
        "You are an expert Python programmer. You will be given a question (problem specification) and will "
        "generate a correct Python program that matches the specification and passes all tests."
    )
    stdin_prompt = _build_lcb_prompt("Add A and B.\n")
    assert stdin_prompt == (
        "### Question:\nAdd A and B.\n\n\n"
        "### Format: Read the inputs from stdin solve the problem and write the answer to stdout (do not directly "
        "test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python "
        "program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
        "```python\n# YOUR CODE HERE\n```\n\n"
        "### Answer: (use the provided format with backticks)\n\n"
    )
    starter = "class Solution:\n    def f(self, x: int) -> int:\n        "
    fn_prompt = _build_lcb_prompt("Do f.", starter)
    assert fn_prompt == (
        "### Question:\nDo f.\n\n"
        "### Format: You will use the following starter code to write the solution to the problem and enclose "
        f"your code within delimiters.\n```python\n{starter}\n```\n\n"
        "### Answer: (use the provided format with backticks)\n\n"
    )

    # o loop usa a mensagem de sistema oficial via o chat template (aqui: tokenizer sem template -> prefixo)
    seen: list[str] = []

    class Engine:
        def generate(self, prompts: list[str], sampling_params: Any) -> list[Any]:
            seen.extend(prompts)

            class _S:
                text = "<think> step 1 </think>\n```python\nprint(1)\n```"
                token_ids = [3, 5, 6, 4, 25, 26]

            class _R:
                outputs = [_S()]

            return [_R() for _ in prompts]

    with tempfile.TemporaryDirectory() as tmp:
        import sys as _sys
        import types as _types
        import unittest.mock as _mock

        class _FakeSP:
            def __init__(self, **kw: Any) -> None:
                self.kw = kw

        fake = _types.ModuleType("vllm")
        fake.SamplingParams = _FakeSP  # type: ignore[attr-defined]
        with _mock.patch.dict(_sys.modules, {"vllm": fake}):
            _run_problems_loop(
                problems=[{"task_id": "q1", "prompt": stdin_prompt, "system_prompt": LCB_SYSTEM_MESSAGE,
                           "io_examples": [{"input": "", "output": "1"}]}],
                model=Engine(), tokenizer=DeterministicWordTokenizer(), run_id="lcbp", phase="F0",
                model_name="m", checkpoint="base", n_loops=2, eval_tier="quick", bench_name="b",
                max_new_tokens=64, temperature=0.0, batch_size=1, cache_dir=Path(tmp) / "c",
                results_path=Path(tmp) / "r.jsonl", smoke=False, smoke_n=1, force=True, backend="vllm",
            )
    assert len(seen) == 1 and seen[0].startswith(LCB_SYSTEM_MESSAGE) and "### Question:" in seen[0]


def test_lcb_pro_keeps_all_test_cases() -> None:
    import sys as _sys
    import types as _types
    import unittest.mock as _mock

    from lib import evals

    n_cases = 40
    cases = [{"input": f"{i}\n", "output": f"{i}\n"} for i in range(n_cases)]
    row = {"question_id": "p1", "question_content": "echo", "starter_code": "", "public_test_cases": cases}
    fake_ds = _types.ModuleType("datasets")
    fake_ds.load_dataset = lambda *a, **k: [row]  # type: ignore[attr-defined]
    captured: dict[str, Any] = {}

    def fake_loop(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"ok": True}

    with _mock.patch.dict(_sys.modules, {"datasets": fake_ds}), _mock.patch.object(evals, "_run_problems_loop", fake_loop):
        res = evals.eval_lcb_pro(object(), None, run_id="r", cache_dir="x", results_path="y")
    assert res == {"ok": True}
    prob = captured["problems"][0]
    assert len(prob["io_examples"]) == n_cases, "LCB Pro não pode truncar os casos de teste"
    assert prob["system_prompt"] == evals.LCB_SYSTEM_MESSAGE
    # e o runner novo (um subprocesso, para no 1º erro) executa todos os casos
    assert check_solution("```python\nprint(input())\n```", io_examples=prob["io_examples"], timeout_s=6.0).passed is True


def test_enforce_tokenizer_report_policy_and_scripts_use_it() -> None:
    import ast
    import contextlib
    import io as _io

    from lib.dory import enforce_tokenizer_report

    base = {"n_texts": 20, "fix_mistral_regex": False, "vllm_import_errors": ["x"]}
    # verified=False: avisa claramente, não aborta
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        enforce_tokenizer_report({**base, "verified": False, "matched": None,
                                  "divergences_initial": None, "divergences_final": None})
    assert "NÃO verificada" in buf.getvalue() and "verified=False" in buf.getvalue()
    # verified=True e matched=True: segue sem aviso
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        enforce_tokenizer_report({**base, "verified": True, "matched": True,
                                  "divergences_initial": 0, "divergences_final": 0})
    assert "AVISO" not in buf.getvalue() and "ERRO" not in buf.getvalue()
    # verified=True e matched=False: aborta com mensagem clara
    try:
        with contextlib.redirect_stdout(_io.StringIO()):
            enforce_tokenizer_report({**base, "verified": True, "matched": False,
                                      "divergences_initial": 3, "divergences_final": 2})
        raise AssertionError("deveria abortar")
    except SystemExit as exc:
        assert "divergem" in str(exc.code) or "diferentes" in str(exc.code)

    # os dois scripts não podem mais ignorar o relatório de verify_tokenizer_equivalence
    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    for name in ("f0_run_baseline.py", "f0_smoke_and_estimate.py"):
        src = (scripts_dir / name).read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                f = node.value.func
                assert not (getattr(f, "id", None) == "verify_tokenizer_equivalence"), (
                    f"{name}: o relatório de verify_tokenizer_equivalence está sendo descartado"
                )
        assert "enforce_tokenizer_report(verify_tokenizer_equivalence(" in src


def test_cost_estimate_by_tokens_and_aggregate_throughput() -> None:
    import contextlib
    import io as _io

    from lib.metrics import loop_compute_factor, measure_smoke_throughput
    from lib.metrics import estimate_compute_cost as est

    assert abs(loop_compute_factor(2) - 1.0) < 1e-12
    assert abs(loop_compute_factor(1) - 36 / 56) < 1e-12 and abs(loop_compute_factor(4) - 96 / 56) < 1e-12

    # smoke sintético: 32 itens por benchmark em UM lote; latency_mean_s = tempo de parede do lote / 32
    smoke = {
        "humaneval+": {"tokens_total_mean": 6000.0, "latency_mean_s": 1.0, "truncated_frac": 0.0,
                       "config": {"n_problems": 32}},
        "lcb_v6_quick": {"tokens_total_mean": 15000.0, "latency_mean_s": 2.5, "truncated_frac": 0.9,
                         "config": {"n_problems": 32}},
    }
    m = measure_smoke_throughput(smoke)
    assert m["total_tokens"] == (6000 + 15000) * 32 and m["total_wall_s"] == 3.5 * 32
    assert abs(m["throughput_tps"] - 21000 / 3.5) < 1e-9  # 6000 tok/s agregados
    assert m["mean_tokens"] == {"simple": 6000.0, "hard": 15000.0}

    with contextlib.redirect_stdout(_io.StringIO()):
        r = est(mean_tokens=m["mean_tokens"], throughput_tps=m["throughput_tps"], cu_per_hour=8.88)
    # conta feita à mão
    tok_quick = 6000 * 164 + 15000 * 200
    tok_gate = 6000 * (164 + 378) + 15000 * (880 + 288)
    f = {1: 36 / 56, 2: 1.0, 4: 96 / 56}
    sec_quick = sum(tok_quick * f[k] for k in (1, 2, 4)) / 6000 + tok_quick * 1.0 / 6000  # + Qwen só quick
    sec_gate = sum(tok_gate * f[k] for k in (1, 2, 4)) / 6000
    assert abs(r["eval_quick_only"]["seconds"] - sec_quick) < 1e-6
    assert abs(r["total"]["seconds"] - (sec_quick + sec_gate)) < 1e-6
    assert abs(r["total"]["cu"] - (sec_quick + sec_gate) / 3600 * 8.88) < 1e-9
    assert {(c["config"], c["tier"]) for c in r["configs"]} == {
        ("dory_L1", "quick"), ("dory_L1", "gate"), ("dory_L2", "quick"), ("dory_L2", "gate"),
        ("dory_L4", "quick"), ("dory_L4", "gate"), ("qwen", "quick"),
    }
    assert r["is_estimate"] is True and r["cu_per_hour"] == 8.88
    # com throughput agregado de ~6000 tok/s o total fica na casa de poucas dezenas de horas-CU, não ~826 CU
    assert r["total"]["cu"] < 100
    # throughput dobrado => tempo pela metade
    with contextlib.redirect_stdout(_io.StringIO()):
        r2 = est(mean_tokens=m["mean_tokens"], throughput_tps=12000.0, cu_per_hour=8.88)
    assert abs(r2["total"]["seconds"] * 2 - r["total"]["seconds"]) < 1e-6
    # limite de confirmação
    with contextlib.redirect_stdout(_io.StringIO()):
        r3 = est(mean_tokens=m["mean_tokens"], throughput_tps=100.0, confirm_threshold_cu=30.0)
    assert r3["requires_confirmation"] is True


def test_truncated_fraction_is_recorded_and_not_confused_with_closed_think() -> None:
    import sys as _sys
    import types as _types
    import unittest.mock as _mock

    max_new = 8

    class Engine:
        def generate(self, prompts: list[str], sampling_params: Any) -> list[Any]:
            outs = []
            for i, _ in enumerate(prompts):
                class _S:
                    pass

                s = _S()
                code = "```python\ndef f():\n    return 1\n```"
                # O prompt já termina em <think> (como nos templates reais), então o texto gerado não tem <think>.
                if i % 2 == 0:  # truncada: rascunho correto DENTRO da CoT, sem </think> -> tem que ser erro
                    s.text, s.token_ids = f"step 1 {code}", [5, 6, 5, 6, 5, 6, 5, 6]
                else:  # terminou no limite, mas fechou </think>: NÃO é truncada
                    s.text, s.token_ids = f"step 1 </think> {code}", [5, 6, 4, 25, 26, 25, 26, 25]
                r = _S()
                r.outputs = [s]
                outs.append(r)
            return outs

    class _FakeSP:
        def __init__(self, **kw: Any) -> None:
            self.kw = kw

    fake = _types.ModuleType("vllm")
    fake.SamplingParams = _FakeSP  # type: ignore[attr-defined]
    with tempfile.TemporaryDirectory() as tmp, _mock.patch.dict(_sys.modules, {"vllm": fake}):
        res_path = Path(tmp) / "results.jsonl"
        problems = [{"task_id": f"t{i}", "prompt": "x", "test_code": "assert f() == 1", "entry_point": None} for i in range(4)]
        common = dict(
            problems=problems, model=Engine(), tokenizer=DeterministicWordTokenizer(), run_id="trunc", phase="F0",
            model_name="m", checkpoint="base", n_loops=2, eval_tier="quick", bench_name="b",
            max_new_tokens=max_new, temperature=0.0, batch_size=4, cache_dir=Path(tmp) / "c",
            results_path=res_path, smoke_n=4, force=True, backend="vllm",
        )
        full = _run_problems_loop(smoke=False, **common)
        assert full["config"]["truncated_frac"] == 0.5 and full["config"]["n_truncated"] == 2
        assert full["pass1"] == 0.5  # só as fechadas passam; o rascunho da CoT truncada não conta
        assert load_results(res_path)[-1]["config"]["truncated_frac"] == 0.5
        smoke = _run_problems_loop(smoke=True, **{**common, "run_id": "trunc_s"})
        assert smoke["truncated_frac"] == 0.5 and smoke["config"]["truncated_frac"] == 0.5


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

    def test_12_hf_cache_never_points_to_drive_and_ensure_model_files(self) -> None:
        test_hf_cache_never_points_to_drive_and_ensure_model_files()

    def test_13_verify_vllm_hf_equivalence_params_and_vllm_batches_of_128(self) -> None:
        test_verify_vllm_hf_equivalence_params_and_vllm_batches_of_128()

    def test_14_tokenizer_20_texts_hf_vs_vllm_and_fix_mistral_regex(self) -> None:
        test_tokenizer_20_texts_hf_vs_vllm_and_fix_mistral_regex()

    def test_15_scicode_official_protocol_with_h5_targets(self) -> None:
        test_scicode_official_protocol_with_h5_targets()

    def test_16_scicode_gold_check_and_loop_waves_and_cache(self) -> None:
        test_scicode_gold_check_and_loop_waves_and_cache()

    def test_17_scicode_ensure_test_data(self) -> None:
        test_scicode_ensure_test_data_errors_clearly_and_accepts_valid_file()

    def test_18_lcb_stdin_functional_and_all_tests(self) -> None:
        test_lcb_stdin_functional_and_all_tests()

    def test_19_lcb_loader_decodes_private_tests(self) -> None:
        test_lcb_loader_decodes_private_tests_and_keeps_all_of_them()

    def test_20_tokenizer_unverified_when_vllm_missing(self) -> None:
        test_tokenizer_report_is_unverified_when_vllm_missing()

    def test_21_new_sandbox_evalplus_lcb_cost_and_truncation(self) -> None:
        test_sandbox_has_no_line_tracing_and_runs_cpu_heavy_code_fast()
        test_evalplus_differential_base_and_plus_semantics()
        test_evalplus_loader_uses_plus_inputs_not_base_only_test()
        test_lcb_prompt_matches_official_template_and_uses_system_message()
        test_lcb_pro_keeps_all_test_cases()
        test_enforce_tokenizer_report_policy_and_scripts_use_it()
        test_cost_estimate_by_tokens_and_aggregate_throughput()
        test_truncated_fraction_is_recorded_and_not_confused_with_closed_think()
        test_callable_and_hf_model_never_become_vllm_model_name()
        test_logit_lens_last_loop_with_output_block_matches_forward()


if __name__ == "__main__":
    unittest.main(verbosity=2)
