"""F0 Smoke Test & Compute Units (CU) Cost Estimate Script.
Runs `eval_quick` with `smoke=True` on Dory L2 via vLLM in an isolated process and computes
the cost estimate for all 4 baselines (Dory L1/L2/L4 on eval_quick + eval_gate, Qwen on eval_quick).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from lib.dory import DORY_HF_REPO, load_dory_tokenizer, verify_tokenizer_equivalence
    from lib.evals import eval_quick
    from lib.hw import get_hw_config
    from lib.io import DEFAULT_DRIVE_ROOT, ensure_model_files, setup_drive_and_cache
    from lib.metrics import estimate_compute_cost

    paths = setup_drive_and_cache(drive_root=DEFAULT_DRIVE_ROOT, local_fallback_root=repo_root)
    ensure_model_files(DORY_HF_REPO)
    verify_tokenizer_equivalence(DORY_HF_REPO)
    tok = load_dory_tokenizer(DORY_HF_REPO)

    eval_hw_cfg = get_hw_config("eval")
    smoke_n = 5

    t0 = time.perf_counter()
    smoke_summary = eval_quick(
        DORY_HF_REPO,
        tok,
        run_id="F0_dory_L2_smoke",
        phase="F0",
        model_name=DORY_HF_REPO,
        checkpoint="base",
        n_loops=2,
        lcb_quick_path=paths["data"] / "lcb_quick.json",
        max_new_tokens=16000,
        temperature=1.0,
        top_p=0.95,
        seed=42,
        backend="vllm",
        batch_size=eval_hw_cfg["batch_size"],
        cache_dir=paths["eval_cache"],
        results_path=paths["results"] / "results.jsonl",
        smoke=True,
        smoke_n=smoke_n,
    )
    smoke_elapsed = time.perf_counter() - t0
    print("Resultado Smoke (Dory L=2):", smoke_summary)

    n_items_eval_quick = 164 + 200
    n_items_eval_gate = 164 + 378 + 880 + 288
    total_items_all_baselines = 3 * (n_items_eval_quick + n_items_eval_gate) + 1 * n_items_eval_quick
    cost_est = estimate_compute_cost(
        smoke_elapsed_s=smoke_elapsed,
        smoke_n_items=smoke_n * 2,
        full_n_items=total_items_all_baselines,
        cu_per_hour=12.0,
        confirm_threshold_cu=30.0,
    )
    est_path = paths["results"] / "f0_cost_estimate.json"
    est_path.write_text(json.dumps(cost_est, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Estimativa salva em: {est_path}")


if __name__ == "__main__":
    main()
