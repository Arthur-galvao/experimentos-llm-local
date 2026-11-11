"""F0 Baseline Runner Script: Runs ONE vLLM engine per process (`L1`, `L2`, `L4`, or `qwen`),
or `plot` (summary table + Pareto plots), or `all` (spawns one separate subprocess per baseline).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _run_single_target(target: str, repo_root: Path) -> None:
    from lib.dory import (
        DORY_HF_REPO,
        QWEN_CONTROL_HF_REPO,
        load_dory_tokenizer,
        verify_tokenizer_equivalence,
    )
    from lib.evals import eval_gate, eval_quick
    from lib.hw import get_hw_config
    from lib.io import DEFAULT_DRIVE_ROOT, ensure_model_files, setup_drive_and_cache
    from lib.metrics import plot_pareto
    from lib.registry import load_results

    paths = setup_drive_and_cache(drive_root=DEFAULT_DRIVE_ROOT, local_fallback_root=repo_root)
    results_jsonl = paths["results"] / "results.jsonl"
    plots_dir = paths["plots"]
    eval_cache_dir = paths["eval_cache"]
    lcb_quick_path = paths["data"] / "lcb_quick.json"
    eval_hw_cfg = get_hw_config("eval")

    if target in ("L1", "L2", "L4"):
        k_loops = int(target[1:])
        run_id = f"F0_dory_L{k_loops}"
        ensure_model_files(DORY_HF_REPO)
        verify_tokenizer_equivalence(DORY_HF_REPO)
        tok = load_dory_tokenizer(DORY_HF_REPO)

        print(f"\n>>> [Subprocesso {target}] Executando eval_quick (vLLM, lotes de 128) para {run_id} (n_loops={k_loops})...")
        eval_quick(
            DORY_HF_REPO,
            tok,
            run_id=run_id,
            phase="F0",
            model_name=DORY_HF_REPO,
            checkpoint="base",
            n_loops=k_loops,
            lcb_quick_path=lcb_quick_path,
            max_new_tokens=16000,
            temperature=1.0,
            top_p=0.95,
            seed=42,
            backend="vllm",
            batch_size=eval_hw_cfg["batch_size"],
            cache_dir=eval_cache_dir,
            results_path=results_jsonl,
            smoke=False,
        )

        print(f"\n>>> [Subprocesso {target}] Executando eval_gate (vLLM, lotes de 128) para {run_id} (n_loops={k_loops})...")
        eval_gate(
            DORY_HF_REPO,
            tok,
            run_id=run_id,
            phase="F0",
            model_name=DORY_HF_REPO,
            checkpoint="base",
            n_loops=k_loops,
            max_new_tokens=16000,
            temperature=1.0,
            top_p=0.95,
            seed=42,
            backend="vllm",
            batch_size=eval_hw_cfg["batch_size"],
            cache_dir=eval_cache_dir,
            results_path=results_jsonl,
            smoke=False,
        )
        print(f"\n=== {run_id}: CONCLUÍDO ===")
        return

    if target == "qwen":
        run_id = "F0_qwen3_5_4b"
        ensure_model_files(QWEN_CONTROL_HF_REPO)
        tok = load_dory_tokenizer(QWEN_CONTROL_HF_REPO)

        print(f"\n>>> [Subprocesso qwen] Executando eval_quick (vLLM, só texto, lotes de 128) para {QWEN_CONTROL_HF_REPO}...")
        eval_quick(
            QWEN_CONTROL_HF_REPO,
            tok,
            run_id=run_id,
            phase="F0",
            model_name=QWEN_CONTROL_HF_REPO,
            checkpoint="base",
            n_loops=None,
            lcb_quick_path=lcb_quick_path,
            max_new_tokens=16000,
            temperature=1.0,
            top_p=0.95,
            seed=42,
            backend="vllm",
            batch_size=eval_hw_cfg["batch_size"],
            cache_dir=eval_cache_dir,
            results_path=results_jsonl,
            smoke=False,
        )
        print(f"\n=== {run_id}: CONCLUÍDO ===")
        return

    if target == "plot":
        f0_records = load_results(results_jsonl, phase="F0")
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

        for bench_name in ("lcb_v6_quick", "humaneval+", "lcb_v6_full", "mbpp+", "scicode"):
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
    parser.add_argument("positional_target", nargs="?", default=None)
    args = parser.parse_args()

    target = args.target or args.positional_target
    if not target:
        parser.error("Informe --target {L1,L2,L4,qwen,plot,all}.")

    if target == "all":
        script_path = str(Path(__file__).resolve())
        for sub_target in ("L1", "L2", "L4", "qwen"):
            print(f"\n{'=' * 20} SUBPROCESSO BASELINE = {sub_target} {'=' * 20}")
            proc = subprocess.run([sys.executable, script_path, "--target", sub_target], check=False)
            if proc.returncode != 0:
                sys.exit(proc.returncode)
        _run_single_target("plot", repo_root)
    else:
        _run_single_target(target, repo_root)


if __name__ == "__main__":
    main()
