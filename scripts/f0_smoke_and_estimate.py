"""F0 Smoke Test & estimativa de custo (Compute Units, CU).

Roda `eval_quick` com `smoke=True` no Dory L2 via vLLM, com um LOTE de N prompts (N configurável, padrão 32)
por benchmark numa única chamada, mede o throughput AGREGADO (tokens_totais / tempo_de_parede, sem o
carregamento do motor) e os tokens médios por item, e estima o custo do F0 completo (Dory L1/L2/L4 em
quick+gate, Qwen só em quick) por tokens / throughput. É uma ESTIMATIVA (ver `estimate_compute_cost`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    parser = argparse.ArgumentParser(description="Smoke test do Dory L2 + estimativa de custo do F0.")
    parser.add_argument("--n", type=int, default=32, help="Prompts por benchmark rodados de uma vez (padrão 32).")
    parser.add_argument("--cu-per-hour", type=float, default=8.88, help="CU/h da A100 no Colab (padrão 8.88).")
    parser.add_argument("--max-new-tokens", type=int, default=16000)
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n deve ser >= 1")

    from lib.dory import (
        DORY_HF_REPO,
        enforce_tokenizer_report,
        load_dory_tokenizer,
        verify_tokenizer_equivalence,
    )
    from lib.evals import eval_quick
    from lib.hw import get_hw_config
    from lib.io import DEFAULT_DRIVE_ROOT, ensure_model_files, setup_drive_and_cache
    from lib.metrics import estimate_compute_cost, measure_smoke_throughput

    paths = setup_drive_and_cache(drive_root=DEFAULT_DRIVE_ROOT, local_fallback_root=repo_root)
    ensure_model_files(DORY_HF_REPO)
    # Avisa se não verificado (verified=False); aborta se verificado e divergente (matched=False).
    enforce_tokenizer_report(verify_tokenizer_equivalence(DORY_HF_REPO))
    tok = load_dory_tokenizer(DORY_HF_REPO)

    eval_hw_cfg = get_hw_config("eval")

    smoke_summary = eval_quick(
        DORY_HF_REPO,
        tok,
        run_id=f"F0_dory_L2_smoke_n{args.n}",  # N no nome: o cache por problema não mistura lotes de tamanhos diferentes
        phase="F0",
        model_name=DORY_HF_REPO,
        checkpoint="base",
        n_loops=2,
        lcb_quick_path=paths["data"] / "lcb_quick.json",
        max_new_tokens=args.max_new_tokens,
        temperature=1.0,
        top_p=0.95,
        seed=42,
        backend="vllm",
        batch_size=eval_hw_cfg["batch_size"],
        cache_dir=paths["eval_cache"],
        results_path=paths["results"] / "results.jsonl",
        smoke=True,
        smoke_n=args.n,
    )

    print(f"\nResultado do smoke (Dory L=2, lote de {args.n} prompts por benchmark):")
    for bench, res in smoke_summary.items():
        trunc = res.get("truncated_frac")
        trunc_s = "n/d" if trunc is None else f"{100.0 * trunc:.1f}%"
        print(
            f"  {bench:13s} backend={res['backend']} pass@1={res['pass1']:.3f} "
            f"tokens_total_médio={res['tokens_total_mean']:.0f} tokens_cot_médio={res['tokens_cot_mean']:.0f} "
            f"truncadas (max_new_tokens sem </think>)={trunc_s}"
        )

    # Se caiu no fallback HF (smoke permite), o throughput NÃO representa o vLLM: não estima.
    non_vllm = [b for b, r in smoke_summary.items() if r.get("backend") != "vllm"]
    if non_vllm:
        sys.exit(
            f"[ERRO] O smoke de {non_vllm} não rodou no vLLM (backend != 'vllm'); o throughput medido "
            "não serve para estimar o custo. Corrija o vLLM e rode de novo."
        )

    meas = measure_smoke_throughput(smoke_summary)
    print(
        f"\nThroughput agregado medido (Dory L2): {meas['throughput_tps']:.0f} tokens/s "
        f"({meas['total_tokens']:.0f} tokens em {meas['total_wall_s']:.0f} s de parede)."
    )

    cost_est = estimate_compute_cost(
        mean_tokens=meas["mean_tokens"],
        throughput_tps=meas["throughput_tps"],
        cu_per_hour=args.cu_per_hour,
        confirm_threshold_cu=30.0,
    )
    cost_est["smoke"] = {
        "n_per_bench": args.n,
        "per_bench": meas["per_bench"],
        "measured_throughput_tps": meas["throughput_tps"],
    }
    est_path = paths["results"] / "f0_cost_estimate.json"
    est_path.write_text(json.dumps(cost_est, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Estimativa salva em: {est_path}")


if __name__ == "__main__":
    main()
