"""F0 Baseline Runner Script: Runs ONE vLLM engine per process (`L1`, `L2`, `L4`, or `qwen`),
or `plot` (summary table + Pareto plots), or `all` (spawns one separate subprocess per baseline).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _run_single_target(target: str, repo_root: Path, scope: str = "quick") -> None:
    from lib.dory import (
        DORY_HF_REPO,
        QWEN_CONTROL_HF_REPO,
        enforce_tokenizer_report,
        load_dory_tokenizer,
        official_sampling,
        verify_tokenizer_equivalence,
    )
    from lib.evals import eval_custom, eval_gate, eval_quick
    from lib.hw import get_hw_config
    from lib.io import DEFAULT_DRIVE_ROOT, ensure_model_files, setup_drive_and_cache
    from lib.metrics import plot_pareto
    from lib.custom_bench import CUSTOM_MAX_NEW_TOKENS, CUSTOM_PROTOCOL_ID
    from lib.registry import load_results

    paths = setup_drive_and_cache(drive_root=DEFAULT_DRIVE_ROOT, local_fallback_root=repo_root)
    results_jsonl = paths["results"] / "results.jsonl"
    plots_dir = paths["plots"]
    eval_cache_dir = paths["eval_cache"]
    lcb_quick_path = paths["data"] / "lcb_quick.json"
    eval_hw_cfg = get_hw_config("eval")

    if target in ("L1", "L2", "L4", "qwen"):
        if target == "qwen":
            repo, run_id, k_loops = QWEN_CONTROL_HF_REPO, "F0_qwen3_5_4b", None
            ensure_model_files(repo)
        else:
            repo, k_loops = DORY_HF_REPO, int(target[1:])
            run_id = f"F0_dory_L{k_loops}"
            ensure_model_files(repo)
            enforce_tokenizer_report(verify_tokenizer_equivalence(repo))
        tok = load_dory_tokenizer(repo)
        sampling = official_sampling(repo)  # temperatura/top_p oficiais; top_k/presence o vLLM aplica sozinho
        common = dict(
            run_id=run_id,
            phase="F0",
            model_name=repo,
            checkpoint="base",
            n_loops=k_loops,
            max_new_tokens=16000,  # benchmarks públicos; o custom usa o orçamento dele (CUSTOM_MAX_NEW_TOKENS)
            temperature=float(sampling["temperature"]),
            top_p=float(sampling["top_p"]),
            seed=42,
            backend="vllm",
            batch_size=eval_hw_cfg["batch_size"],
            cache_dir=eval_cache_dir,
            results_path=results_jsonl,
            smoke=False,
        )

        if scope in ("quick", "both"):
            print(f"\n>>> [Subprocesso {target}] eval_quick (vLLM, lotes de 128) para {run_id} (n_loops={k_loops})...")
            eval_quick(repo, tok, lcb_quick_path=lcb_quick_path, **common)
        if scope in ("gate", "both") and target != "qwen":  # Qwen não tem gate
            print(f"\n>>> [Subprocesso {target}] eval_gate (vLLM, lotes de 128) para {run_id} (n_loops={k_loops})...")
            eval_gate(repo, tok, **common)
        if scope.startswith("custom_"):
            subset = scope.split("_", 1)[1]
            print(
                f"\n>>> [Subprocesso {target}] benchmark custom ({subset}, até {CUSTOM_MAX_NEW_TOKENS} tokens, "
                f"amostragem {sampling}) para {run_id} (n_loops={k_loops})..."
            )
            eval_custom(repo, tok, subset=subset, **{**common, "max_new_tokens": CUSTOM_MAX_NEW_TOKENS})
        print(f"\n=== {run_id}: CONCLUÍDO ===")
        return

    if target == "plot":
        f0_records = [  # custom de versão antiga do benchmark não entra na tabela nem no gráfico
            r for r in load_results(results_jsonl, phase="F0")
            if not str(r.get("bench", "")).startswith("custom_")
            # custom_protocol, não eval_protocol: o da Qwen ganha sufixo da amostragem (ex.: custom_v3_k20_pp1.5)
            or (r.get("config") or {}).get("custom_protocol") == CUSTOM_PROTOCOL_ID
        ]
        print(f"Total de registros F0 em {results_jsonl}: {len(f0_records)}\n")
        header = (
            f"{'run_id':18s} | {'eval':6s} | {'bench':14s} | {'loops':5s} | "
            f"{'pass@1':8s} | {'cot_tok':8s} | {'lat(s)':7s} | {'pass1/1k_tok':12s}"
        )
        print(header)
        print("-" * len(header))
        for r in f0_records:
            loops_str = str(r["n_loops"]) if r.get("n_loops") is not None else "-"
            print(
                f"{r['run_id']:18s} | {r['eval']:6s} | {r['bench']:14s} | {loops_str:5s} | "
                f"{r['pass1']:8.4f} | {r['tokens_cot_mean']:8.1f} | {r['latency_mean_s']:7.2f} | "
                f"{r['acerto_por_1k_tokens']:12.4f}"
            )

        for r in f0_records:
            cfg = r.get("config") or {}
            if str(r.get("bench", "")).startswith("custom_"):
                tiers = " | ".join(
                    f"{t}: pass@1 {cfg.get(f'{t}_pass1')} cot média {cfg.get(f'{t}_tokens_cot_mean')} "
                    f"mediana {cfg.get(f'{t}_tokens_cot_median')} trunc {cfg.get(f'{t}_truncated_frac')}"
                    for t in ("easy", "medium", "hard")
                )
                print(
                    f"\n{r['run_id']} {r['bench']} (até {cfg.get('max_new_tokens')} tokens): {tiers}"
                    f"\n  pass@1 se o limite fosse B: geral {cfg.get('pass1_at_budget')}"
                    f" | hard {cfg.get('hard_pass1_at_budget')}"
                    f"\n  falhas no hard {cfg.get('hard_fail_breakdown')}"
                    f" | cot nos acertos {cfg.get('tokens_cot_mean_correct')}"
                    f" | {cfg.get('throughput_tok_s')} tok/s | falhas {cfg.get('fail_breakdown')}"
                    f"\n  por família {cfg.get('family_pass1')}"
                )

        for bench_name in ("custom_quick", "custom_full", "lcb_v6_quick", "humaneval+", "lcb_v6_full", "mbpp+", "scicode"):
            bench_rows = [r for r in f0_records if r.get("bench") == bench_name]
            if not bench_rows:
                continue
            out_plot = plots_dir / f"F0_pareto_{bench_name.replace('+', 'plus')}.png"
            plot_pareto(
                bench_rows,
                out_plot,
                bench_filter=bench_name,
                baseline_phase="F0",
                title=f"F0 Baselines — Fronteira de Pareto ({bench_name})",
            )
            print(f"Gráfico salvo em: {out_plot}")

        local_results_dir = repo_root / "results"
        if local_results_dir.resolve() != paths["results"].resolve() and results_jsonl.exists():
            local_results_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(results_jsonl, local_results_dir / "results.jsonl")
            local_plots_dir = local_results_dir / "plots"
            local_plots_dir.mkdir(parents=True, exist_ok=True)
            for p in plots_dir.glob("*.png"):
                shutil.copy2(p, local_plots_dir / p.name)
        return

    raise ValueError(f"Alvo desconhecido: {target}")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    parser = argparse.ArgumentParser(description="Executa um baseline da F0 num processo isolado.")
    parser.add_argument(
        "--target",
        choices=("L1", "L2", "L4", "qwen", "plot", "all"),
        default=None,
        help="Baseline alvo ('L1', 'L2', 'L4', 'qwen', 'plot' ou 'all').",
    )
    parser.add_argument(
        "--scope",
        choices=("custom_quick", "custom_full", "quick", "gate", "both"),
        default="custom_quick",
        help=(
            "'custom_quick' (126 problemas custom, padrão) ou 'custom_full' (504); benchmarks públicos: "
            "'quick' (HE+ + LCB quick), 'gate' (HE+, MBPP+, LCB full/Pro, SciCode; Qwen não tem) ou 'both'."
        ),
    )
    parser.add_argument("positional_target", nargs="?", default=None)
    args = parser.parse_args()

    target = args.target or args.positional_target
    if not target:
        parser.error("Informe --target {L1,L2,L4,qwen,plot,all}.")

    if target == "all":
        script_path = str(Path(__file__).resolve())
        for sub_target in ("L1", "L2", "L4", "qwen"):
            print(f"\n{'=' * 20} SUBPROCESSO BASELINE = {sub_target} {'=' * 20}")
            proc = subprocess.run([sys.executable, script_path, "--target", sub_target, "--scope", args.scope], check=False)
            if proc.returncode != 0:
                sys.exit(proc.returncode)
        _run_single_target("plot", repo_root)
    else:
        _run_single_target(target, repo_root, scope=args.scope)


if __name__ == "__main__":
    main()
