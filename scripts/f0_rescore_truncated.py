"""Recorrige os runs já feitos: item truncado (sem </think>) que "passou" com rascunho da CoT vira erro.

Corrige o `eval_cache` (passed=False, error_type=EmptyCode, o que o harness atual produziria) e
recalcula `pass1`/`acerto_por_1k_tokens` em `results.jsonl`. Idempotente. Sem GPU, sem regerar.
Depois rode `python scripts/f0_run_baseline.py plot` para refazer os gráficos.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))
    from lib.io import DEFAULT_DRIVE_ROOT, setup_drive_and_cache
    from lib.metrics import acerto_por_1k_tokens

    paths = setup_drive_and_cache(drive_root=DEFAULT_DRIVE_ROOT, local_fallback_root=repo_root)
    results_jsonl = paths["results"] / "results.jsonl"
    records = [json.loads(line) for line in results_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]

    for rec in records:
        if rec.get("config", {}).get("smoke"):
            continue
        caches = list(paths["eval_cache"].glob(f"{rec['run_id']}_{rec['bench']}_{rec['eval']}_*.jsonl"))
        if len(caches) != 1:
            print(f"[pula] {rec['run_id']} {rec['bench']}: {len(caches)} caches encontrados")
            continue
        rows = [json.loads(line) for line in caches[0].read_text(encoding="utf-8").splitlines() if line.strip()]
        n_fixed = 0
        for row in rows:
            if row.get("passed") and row.get("truncated"):
                row["passed"], row["error_type"] = False, "EmptyCode"
                n_fixed += 1
        if n_fixed:
            caches[0].write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        old = rec["pass1"]
        rec["pass1"] = round(sum(bool(r["passed"]) for r in rows) / len(rows), 6)
        rec["acerto_por_1k_tokens"] = round(
            acerto_por_1k_tokens(rec["pass1"], rec["tokens_cot_mean"], rec["tokens_total_mean"]), 6
        )
        rec["config"]["rescored_truncated_as_fail"] = True
        print(f"{rec['run_id']:16s} {rec['bench']:14s} pass1 {old:.4f} -> {rec['pass1']:.4f} ({n_fixed} itens corrigidos)")

    results_jsonl.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


if __name__ == "__main__":
    main()
