"""Dry unit tests for `lab/bench_summary.py` (stdlib + pytest only, no GPU/torch/vllm)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent
REPO_ROOT = LAB_DIR.parent
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

from bench_summary import generate_summary  # noqa: E402


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _make_sample_row(item_id: str, passed_flags: list[bool], tokens_per_sample: int) -> dict:
    samples = [
        {
            "n_total_tokens": tokens_per_sample,
            "passed": flag,
            "truncated": False,
        }
        for flag in passed_flags
    ]
    return {
        "item_id": item_id,
        "p_hat": sum(passed_flags) / len(passed_flags),
        "samples": samples,
    }


def test_generate_summary_full_paired_and_failed_graphs(tmp_path: Path) -> None:
    drive_root = tmp_path / "drive"
    out_dir = drive_root / "results" / "lab_validation" / "lab-pc-01"
    out_dir.mkdir(parents=True)

    (out_dir / "env.json").write_text(
        json.dumps(
            {
                "torch": "2.14.0+cu130",
                "vllm": "0.32.1rc1",
                "ram_gib": 7.7,
                "gpu": "NVIDIA GeForce RTX 2060 SUPER",
                "capability": "7.5",
                "vram_gib": 8.0,
            }
        ),
        encoding="utf-8",
    )
    (out_dir / "smoke_int8_eager.json").write_text(
        json.dumps(
            {
                "tok_s": 48.5,
                "kv_cache_size_tokens": 24576,
                "max_concurrency": 2.0,
                "load_s": 18.2,
                "gen_s": 41.0,
                "tokens_generated": 1988,
                "attention_backend": "TRITON_ATTN",
            }
        ),
        encoding="utf-8",
    )
    (out_dir / "smoke_int8_graphs.error.txt").write_text(
        "exit_code=1\nCUDA out of memory\n",
        encoding="utf-8",
    )

    _write_jsonl(
        out_dir / "quality" / "throughput_int8.jsonl",
        [
            {"variant": "int8", "problemas": 1, "tokens": 4000, "elapsed_s": 120, "load_s": 20},
            {"variant": "int8", "problemas": 1, "tokens": 5000, "elapsed_s": 120, "load_s": 20},
        ],
    )
    _write_jsonl(
        out_dir / "quality" / "samples_efficient_int8.jsonl",
        [
            _make_sample_row("taco/1", [True, False], 1200),
            _make_sample_row("taco/2", [True, True], 800),
        ],
    )
    _write_jsonl(
        drive_root / "data" / "f1a" / "samples_efficient_L2_n8_t1_p0.95_m8192.jsonl",
        [
            _make_sample_row("taco/1", [True, True], 1000),
            _make_sample_row("taco/2", [False, True], 900),
            _make_sample_row("taco/999", [True, True], 500),  # extra unmatched problem
        ],
    )

    summary = generate_summary(
        out_dir=out_dir,
        drive_root=drive_root,
        repo_root=REPO_ROOT,
        hostname="lab-pc-01",
        driver_version="580.10",
        git_sha="abc1234",
        lab_vram_gib=7.25,
    )

    assert summary["smokes"]["chosen_mode"] == "eager"
    assert summary["smokes"]["graphs"]["status"] == "FALHOU"
    assert summary["throughput"]["n_problems_done"] == 2
    assert summary["throughput"]["total_tokens"] == 9000
    assert summary["throughput"]["total_gen_s"] == 200.0
    assert summary["throughput"]["real_tok_s"] == 45.0
    assert summary["projections"]["farm_15pcs_tok_s"] == 675.0
    assert summary["projections"]["farm_vs_a100_ratio"] == 0.84

    q = summary["quality"]
    assert q["n_problems"] == 2
    assert q["indicative_only"] is True
    assert q["metrics"]["p_hat"]["a100_bf16"] == 0.75
    assert q["metrics"]["p_hat"]["rtx_2060s_int8"] == 0.75
    assert q["metrics"]["p_hat"]["delta"] == 0.0
    assert "menor_trace_correto" in q["metrics"]

    md_text = (out_dir / "bench_summary.md").read_text(encoding="utf-8")
    assert "RTX 2060 SUPER (INT8)" in md_text
    assert "A100 (BF16)" in md_text
    assert "apenas indicativa" in md_text
    assert "675.0" in md_text
    assert (out_dir / "bench_summary.json").exists()


def test_generate_summary_graphs_faster_and_skip_quality(tmp_path: Path) -> None:
    drive_root = tmp_path / "drive"
    out_dir = drive_root / "results" / "lab_validation" / "lab-pc-02"
    out_dir.mkdir(parents=True)

    (out_dir / "smoke_int8_eager.json").write_text(
        json.dumps({"tok_s": 40.0, "kv_cache_size_tokens": 24576, "max_concurrency": 2.0, "load_s": 15.0}),
        encoding="utf-8",
    )
    (out_dir / "smoke_int8_graphs.json").write_text(
        json.dumps({"tok_s": 55.2, "kv_cache_size_tokens": 20480, "max_concurrency": 1.67, "load_s": 25.0}),
        encoding="utf-8",
    )

    summary = generate_summary(
        out_dir=out_dir,
        drive_root=drive_root,
        repo_root=REPO_ROOT,
        hostname="lab-pc-02",
    )

    assert summary["smokes"]["chosen_mode"] == "graphs"
    assert summary["throughput"]["n_runs"] == 0
    assert summary["throughput"]["real_tok_s"] is None
    assert summary["projections"]["farm_15pcs_tok_s"] is None
    assert summary["quality"]["n_problems"] == 0

    md_text = (out_dir / "bench_summary.md").read_text(encoding="utf-8")
    assert "CUDA Graphs" in md_text
    assert "etapa `quality` pulada" in md_text
