"""Experiment registry for results/results.jsonl (Section 3.3 of ESPEC_ANTIGRAVITY.md)."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any

RESULT_FIELDS: tuple[str, ...] = (
    "run_id",
    "phase",
    "model",
    "checkpoint",
    "n_loops",
    "eval",
    "bench",
    "pass1",
    "tokens_cot_mean",
    "tokens_total_mean",
    "latency_mean_s",
    "acerto_por_1k_tokens",
    "config",
    "git_sha",
    "timestamp",
)

DEFAULT_RESULTS_PATH = Path("results") / "results.jsonl"


def get_git_sha(cwd: str | Path | None = None) -> str:
    """Return short git SHA of the current repository, or 'unknown' if not a git repo."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except Exception:
        pass
    return "unknown"


def append_result(
    *,
    run_id: str,
    phase: str,
    model: str,
    checkpoint: str,
    n_loops: int | None,
    eval_tier: str,
    bench: str,
    pass1: float,
    tokens_cot_mean: float,
    tokens_total_mean: float,
    latency_mean_s: float,
    config: dict[str, Any] | None = None,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    git_sha: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Append a standardized benchmark result line to `results/results.jsonl`.

    Automatically computes `acerto_por_1k_tokens = pass1 / max(tokens_cot_mean / 1000.0, 1e-6)`
    (using `tokens_total_mean` as fallback only when `tokens_cot_mean <= 0`).
    """
    denom_tokens = tokens_cot_mean if tokens_cot_mean > 0 else max(tokens_total_mean, 1.0)
    acerto_1k = float(pass1) / (denom_tokens / 1000.0) if denom_tokens > 0 else 0.0

    record: dict[str, Any] = {
        "run_id": str(run_id),
        "phase": str(phase),
        "model": str(model),
        "checkpoint": str(checkpoint),
        "n_loops": int(n_loops) if n_loops is not None else None,
        "eval": str(eval_tier),
        "bench": str(bench),
        "pass1": round(float(pass1), 6),
        "tokens_cot_mean": round(float(tokens_cot_mean), 3),
        "tokens_total_mean": round(float(tokens_total_mean), 3),
        "latency_mean_s": round(float(latency_mean_s), 4),
        "acerto_por_1k_tokens": round(float(acerto_1k), 6),
        "config": dict(config or {}),
        "git_sha": git_sha or get_git_sha(),
        "timestamp": timestamp or dt.datetime.now(dt.timezone.utc).isoformat(),
    }

    path = Path(results_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()

    return record


def load_results(
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    *,
    phase: str | None = None,
    eval_tier: str | None = None,
    bench: str | None = None,
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    """Load and optionally filter experiment records from `results/results.jsonl`."""
    path = Path(results_path)
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if phase is not None and row.get("phase") != phase:
                continue
            if eval_tier is not None and row.get("eval") != eval_tier:
                continue
            if bench is not None and row.get("bench") != bench:
                continue
            if run_id is not None and row.get("run_id") != run_id:
                continue
            records.append(row)
    return records


def has_completed_run(
    *,
    run_id: str,
    bench: str,
    eval_tier: str | None = None,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
) -> dict[str, Any] | None:
    """Return the existing record if `(run_id, bench, eval_tier)` is already logged."""
    matches = load_results(
        results_path=results_path,
        run_id=run_id,
        bench=bench,
        eval_tier=eval_tier,
    )
    return matches[-1] if matches else None
