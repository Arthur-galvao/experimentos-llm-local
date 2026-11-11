"""Efficiency metrics, Pareto frontier calculation, plotting, and phase gate verification (Section 3.4 & 3.5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def acerto_por_1k_tokens(pass1: float, tokens_cot_mean: float, tokens_total_mean: float = 0.0) -> float:
    """Compute primary north-star efficiency metric: accuracy (pass@1) per 1,000 CoT tokens."""
    denom = tokens_cot_mean if tokens_cot_mean > 0 else tokens_total_mean
    if denom <= 0:
        return 0.0
    return float(pass1) / (float(denom) / 1000.0)


def is_pareto_dominated(candidate: dict[str, Any], others: list[dict[str, Any]]) -> bool:
    """A point `candidate` is Pareto-dominated if another point in `others` has:
      - pass1 >= candidate['pass1'] AND tokens_cot_mean <= candidate['tokens_cot_mean']
      - with at least one strict inequality.
    """
    c_pass = float(candidate["pass1"])
    c_tok = float(candidate["tokens_cot_mean"])
    for other in others:
        if other is candidate:
            continue
        o_pass = float(other["pass1"])
        o_tok = float(other["tokens_cot_mean"])
        if o_pass >= c_pass and o_tok <= c_tok and (o_pass > c_pass or o_tok < c_tok):
            return True
    return False


def compute_pareto_frontier(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the non-dominated Pareto frontier sorted by `tokens_cot_mean` ascending."""
    if not records:
        return []
    frontier = [r for r in records if not is_pareto_dominated(r, records)]
    frontier.sort(key=lambda r: (float(r["tokens_cot_mean"]), -float(r["pass1"])))
    # Deduplicate identical coordinates
    deduped: list[dict[str, Any]] = []
    for pt in frontier:
        if not deduped:
            deduped.append(pt)
            continue
        prev = deduped[-1]
        if (
            abs(float(pt["tokens_cot_mean"]) - float(prev["tokens_cot_mean"])) < 1e-6
            and abs(float(pt["pass1"]) - float(prev["pass1"])) < 1e-6
        ):
            continue
        deduped.append(pt)
    return deduped


def plot_pareto(
    records: list[dict[str, Any]],
    output_path: str | Path,
    *,
    bench_filter: str | None = None,
    baseline_phase: str = "F0",
    title: str | None = None,
) -> Path:
    """Generate and save a scatter plot of `pass@1 x tokens_cot_mean` with the Pareto frontier
    highlighted and baseline runs marked (Section 3.4 of ESPEC_ANTIGRAVITY.md).
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        # Minimal valid 1x1 PNG fallback when matplotlib is not installed in a bare environment
        _MINIMAL_PNG = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
            b"\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        out.write_bytes(_MINIMAL_PNG)
        return out

    filtered = [
        r for r in records if (bench_filter is None or r.get("bench") == bench_filter)
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=150)
    if not filtered:
        ax.set_title(title or "Fronteira de Pareto (sem dados)")
        ax.set_xlabel("Tokens médios de CoT (menor é melhor)")
        ax.set_ylabel("pass@1 (maior é melhor)")
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        return out

    baselines = [r for r in filtered if r.get("phase") == baseline_phase]
    non_baselines = [r for r in filtered if r.get("phase") != baseline_phase]
    frontier = compute_pareto_frontier(filtered)

    if baselines:
        ax.scatter(
            [float(r["tokens_cot_mean"]) for r in baselines],
            [float(r["pass1"]) for r in baselines],
            marker="D",
            s=80,
            color="#1f77b4",
            label=f"Baseline ({baseline_phase})",
            zorder=3,
        )
    if non_baselines:
        ax.scatter(
            [float(r["tokens_cot_mean"]) for r in non_baselines],
            [float(r["pass1"]) for r in non_baselines],
            marker="o",
            s=80,
            color="#2ca02c",
            label="Fases Experimentais",
            zorder=3,
        )

    if frontier:
        fx = [float(r["tokens_cot_mean"]) for r in frontier]
        fy = [float(r["pass1"]) for r in frontier]
        ax.plot(
            fx,
            fy,
            linestyle="--",
            linewidth=2.0,
            color="#d62728",
            marker="s",
            markersize=5,
            label="Fronteira de Pareto",
            zorder=4,
        )

    for r in filtered:
        label = str(r.get("run_id") or r.get("model", ""))
        if r.get("n_loops") is not None:
            label = f"{label} (L={r['n_loops']})"
        ax.annotate(
            label,
            (float(r["tokens_cot_mean"]), float(r["pass1"])),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8,
            alpha=0.9,
        )

    bench_title = f" — {bench_filter}" if bench_filter else ""
    ax.set_title(title or f"Fronteira de Pareto: pass@1 vs Tokens de CoT{bench_title}")
    ax.set_xlabel("Tokens médios de CoT (menor = mais comprimido)")
    ax.set_ylabel("pass@1 (maior = mais preciso)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def check_phase_gate(
    candidate_records: list[dict[str, Any]],
    previous_records: list[dict[str, Any]],
    *,
    baseline_scicode_pass1: float | None = None,
    candidate_scicode_pass1: float | None = None,
    max_scicode_drop_pp: float = 2.0,
) -> dict[str, Any]:
    """Evaluate the phase gate criterion (Section 3.5 of ESPEC_ANTIGRAVITY.md):
    1. At least one candidate point strictly dominates or extends the previous Pareto frontier:
       - strictly lower `tokens_cot_mean` with equal-or-higher `pass1`, OR
       - strictly higher `pass1` with equal-or-lower `tokens_cot_mean`, OR
       - is non-dominated when merged with `previous_records` and improves the frontier.
    2. SciCode does not drop more than `max_scicode_drop_pp` percentage points (0.02 in [0,1] scale)
       relative to the baseline.

    Prints 'PASSOU' or 'REPROVOU' with justification and returns a structured verdict dict.
    """
    prev_frontier = compute_pareto_frontier(previous_records)
    advances_pareto = False
    winning_points: list[str] = []

    if not prev_frontier and candidate_records:
        advances_pareto = True
        winning_points = [str(r.get("run_id", "candidate")) for r in candidate_records]
    else:
        for cand in candidate_records:
            c_pass = float(cand["pass1"])
            c_tok = float(cand["tokens_cot_mean"])
            # Check if candidate dominates at least one point on the previous frontier
            # or is non-dominated by the previous frontier with strictly better efficiency
            dominated_by_prev = is_pareto_dominated(cand, prev_frontier)
            strictly_beats_prev = any(
                (c_tok < float(p["tokens_cot_mean"]) and c_pass >= float(p["pass1"]))
                or (c_pass > float(p["pass1"]) and c_tok <= float(p["tokens_cot_mean"]))
                for p in prev_frontier
            )
            if not dominated_by_prev and (strictly_beats_prev or not prev_frontier):
                advances_pareto = True
                winning_points.append(str(cand.get("run_id", "candidate")))

    scicode_ok = True
    scicode_msg = "SciCode check N/A"
    if baseline_scicode_pass1 is not None and candidate_scicode_pass1 is not None:
        # Handle both [0, 1] fraction and [0, 100] percentage scales transparently
        scale = 100.0 if (baseline_scicode_pass1 <= 1.0 and candidate_scicode_pass1 <= 1.0) else 1.0
        drop_pp = (baseline_scicode_pass1 - candidate_scicode_pass1) * scale
        scicode_ok = drop_pp <= max_scicode_drop_pp + 1e-9
        scicode_msg = (
            f"SciCode drop = {drop_pp:.2f} pp (limite <= {max_scicode_drop_pp:.2f} pp)"
        )

    passed = advances_pareto and scicode_ok
    status_str = "PASSOU" if passed else "REPROVOU"
    reasons = [
        f"Avança fronteira de Pareto: {'SIM' if advances_pareto else 'NÃO'}"
        + (f" ({', '.join(winning_points)})" if winning_points else ""),
        scicode_msg,
    ]
    summary = f"[{status_str}] " + " | ".join(reasons)
    print(summary)
    return {
        "passed": passed,
        "status": status_str,
        "advances_pareto": advances_pareto,
        "winning_points": winning_points,
        "scicode_ok": scicode_ok,
        "summary": summary,
    }


# Taxa de consumo da A100 no Colab (CU/h) medida no run real; era 12.0 (errado) na versão anterior.
DEFAULT_CU_PER_HOUR = 8.88

# Camadas ("blocos") por parte do Dory: entrada + recorrente x loops + saída. Custo por token ~ 8 + 20*k + 8.
DORY_COST_BLOCKS_INPUT = 8
DORY_COST_BLOCKS_RECURRENT = 20
DORY_COST_BLOCKS_OUTPUT = 8

# Número de itens por benchmark (os mesmos usados em scripts/f0_smoke_and_estimate.py antes da reescrita).
DEFAULT_ITEMS_PER_BENCH: dict[str, int] = {
    "humaneval+": 164,
    "mbpp+": 378,
    "lcb_v6_quick": 200,
    "lcb_v6_full": 880,
    "scicode": 288,
}
# Quais benchmarks entram em cada tier (eval_quick / eval_gate, como em lib/evals.py).
EVAL_TIER_BENCHES: dict[str, tuple[str, ...]] = {
    "quick": ("humaneval+", "lcb_v6_quick"),
    "gate": ("humaneval+", "mbpp+", "lcb_v6_full", "scicode"),
}
# Grupo de tokens/item: o smoke mede HumanEval+ ("simple") e LCB ("hard"); MBPP+ herda o primeiro e
# SciCode herda o segundo (SciCode não roda no smoke: PREMISSA, não medida).
BENCH_TOKEN_GROUP: dict[str, str] = {
    "humaneval+": "simple",
    "mbpp+": "simple",
    "lcb_v6_quick": "hard",
    "lcb_v6_full": "hard",
    "scicode": "hard",
}


def loop_compute_factor(
    n_loops: int,
    *,
    ref_loops: int = 2,
    n_input: int = DORY_COST_BLOCKS_INPUT,
    n_recurrent: int = DORY_COST_BLOCKS_RECURRENT,
    n_output: int = DORY_COST_BLOCKS_OUTPUT,
) -> float:
    """Custo relativo de computação por token com `n_loops` loops, normalizado em `ref_loops` (k=2):
    `(8 + 20*k + 8) / (8 + 20*2 + 8)`  ->  k=1: 0.643, k=2: 1.0, k=4: 1.714."""
    if n_loops < 1 or ref_loops < 1:
        raise ValueError("n_loops e ref_loops devem ser >= 1")
    return (n_input + n_recurrent * n_loops + n_output) / float(n_input + n_recurrent * ref_loops + n_output)


def measure_smoke_throughput(smoke_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Extrai de `eval_quick(smoke=True, smoke_n=N)` o throughput AGREGADO e os tokens médios por item.

    Em cada benchmark, as N respostas saem de UMA chamada em lote ao vLLM, e `latency_mean_s` é o tempo de
    parede dessa chamada dividido por N (o carregamento do motor fica fora dessa medida). Logo:
    `tokens_totais = tokens_total_mean * N`, `tempo_de_parede = latency_mean_s * N` e
    `throughput = soma(tokens_totais) / soma(tempo_de_parede)` em tokens/s. Retorna também
    `mean_tokens` por grupo ("simple" = HumanEval+, "hard" = LCB) e a fração de respostas truncadas.
    """
    per_bench: dict[str, dict[str, float]] = {}
    tot_tokens = 0.0
    tot_wall = 0.0
    for bench, res in smoke_results.items():
        n = int((res.get("config") or {}).get("n_problems") or res.get("n_problems") or 0)
        if n <= 0:
            raise ValueError(f"Resultado de smoke de '{bench}' sem n_problems; não dá para medir throughput.")
        tokens = float(res["tokens_total_mean"]) * n
        wall = float(res["latency_mean_s"]) * n
        per_bench[bench] = {
            "n": n,
            "tokens_total": tokens,
            "wall_s": wall,
            "tokens_per_s": tokens / wall if wall > 0 else 0.0,
            "mean_tokens": float(res["tokens_total_mean"]),
            "truncated_frac": res.get("truncated_frac"),
        }
        tot_tokens += tokens
        tot_wall += wall
    if tot_wall <= 0 or tot_tokens <= 0:
        raise ValueError("Smoke sem tokens ou sem tempo medido; não dá para estimar throughput.")
    mean_tokens: dict[str, float] = {}
    for group in ("simple", "hard"):
        rows = [b for name, b in per_bench.items() if BENCH_TOKEN_GROUP.get(name) == group]
        n_group = sum(r["n"] for r in rows)
        if n_group:
            mean_tokens[group] = sum(r["tokens_total"] for r in rows) / n_group
    return {
        "per_bench": per_bench,
        "throughput_tps": tot_tokens / tot_wall,
        "total_tokens": tot_tokens,
        "total_wall_s": tot_wall,
        "mean_tokens": mean_tokens,
    }


def estimate_compute_cost(
    *,
    mean_tokens: dict[str, float],
    throughput_tps: float,
    cu_per_hour: float = DEFAULT_CU_PER_HOUR,
    confirm_threshold_cu: float = 30.0,
    items_per_bench: dict[str, int] | None = None,
    dory_loops: tuple[int, ...] = (1, 2, 4),
    include_qwen: bool = True,
    qwen_compute_factor: float = 1.0,
    ref_loops: int = 2,
) -> dict[str, Any]:
    """Estima tempo e Compute Units (CU) do F0 completo por TOKENS e THROUGHPUT AGREGADO.

    Não extrapola tempo serial por item (isso superestimava ~10x, pois no run real o vLLM processa lotes
    de 128 em paralelo). Em vez disso, para cada configuração:
        tokens = soma_bench( tokens_medios_por_item[grupo] * n_itens[bench] )
        segundos = tokens / (throughput_L2 / fator_loops(k))
    com `fator_loops(k) = (8 + 20k + 8) / (8 + 20*2 + 8)` (custo por token proporcional à computação).
    Configurações: Dory L1/L2/L4 em quick + gate; Qwen só em quick (`qwen_compute_factor` = custo relativo
    por token do Qwen vs Dory L2, PREMISSA = 1.0).

    Args:
        mean_tokens: `{"simple": tokens/item de HumanEval+/MBPP+, "hard": tokens/item de LCB/SciCode}`,
            medidos no smoke (ver `measure_smoke_throughput`).
        throughput_tps: throughput agregado (tokens/s) do Dory com `ref_loops` loops, medido com um lote
            de N prompts de uma vez.

    ESTIMATIVA, não garantia. Premissas: tokens/item de L1/L4/Qwen = o medido em L2 (L1 e L4 podem pensar
    mais/menos); MBPP+ herda o de HumanEval+ e SciCode o de LCB (não medidos); o throughput de um lote
    pequeno (N=32) tende a ser MENOR que o de lotes de 128 (superestima o tempo), mas a cauda de respostas
    longas (até `max_new_tokens`) num lote real pode tornar o tempo MAIOR (subestima). Prints e retorno
    mostram também a estimativa só do `eval_quick`.
    """
    if throughput_tps <= 0:
        raise ValueError("throughput_tps deve ser > 0")
    for g in ("simple", "hard"):
        if g not in mean_tokens or mean_tokens[g] < 0:
            raise ValueError(f"mean_tokens precisa de '{g}' >= 0 (medido no smoke)")
    items = dict(DEFAULT_ITEMS_PER_BENCH)
    if items_per_bench:
        items.update(items_per_bench)

    def _tier_tokens(tier: str) -> float:
        return sum(mean_tokens[BENCH_TOKEN_GROUP[b]] * items[b] for b in EVAL_TIER_BENCHES[tier])

    configs: list[dict[str, Any]] = []
    for k in dory_loops:
        factor = loop_compute_factor(k, ref_loops=ref_loops)
        for tier in ("quick", "gate"):
            configs.append({"config": f"dory_L{k}", "tier": tier, "tokens": _tier_tokens(tier), "factor": factor})
    if include_qwen:
        configs.append(
            {"config": "qwen", "tier": "quick", "tokens": _tier_tokens("quick"), "factor": float(qwen_compute_factor)}
        )
    for c in configs:
        c["seconds"] = c["tokens"] * c["factor"] / float(throughput_tps)
        c["hours"] = c["seconds"] / 3600.0
        c["cu"] = c["hours"] * cu_per_hour

    def _total(rows: list[dict[str, Any]]) -> dict[str, float]:
        sec = sum(r["seconds"] for r in rows)
        return {
            "tokens": sum(r["tokens"] for r in rows),
            "seconds": sec,
            "hours": sec / 3600.0,
            "cu": sec / 3600.0 * cu_per_hour,
        }

    total = _total(configs)
    quick_only = _total([c for c in configs if c["tier"] == "quick"])
    requires_confirmation = total["cu"] > confirm_threshold_cu

    print(
        f"[Estimativa de Custo] (ESTIMATIVA por tokens / throughput agregado) "
        f"throughput={throughput_tps:.0f} tok/s (L{ref_loops}), tokens/item: "
        f"simples={mean_tokens['simple']:.0f}, difícil={mean_tokens['hard']:.0f}, {cu_per_hour:.2f} CU/h."
    )
    for c in configs:
        print(
            f"  - {c['config']:8s} {c['tier']:5s}: {c['tokens'] / 1e6:7.2f} Mtok x{c['factor']:.2f} -> "
            f"{c['hours']:6.2f} h ({c['cu']:6.1f} CU)"
        )
    print(f"  => eval_quick (todas as configs): ~{quick_only['hours']:.2f} h (~{quick_only['cu']:.1f} CU)")
    print(f"  => F0 completo: ~{total['hours']:.2f} h (~{total['cu']:.1f} CU @ {cu_per_hour:.2f} CU/h)")
    if requires_confirmation:
        print(
            f"[ATENÇÃO] Custo estimado ({total['cu']:.1f} CU) excede o limite de {confirm_threshold_cu:.1f} CU! "
            "Confirme antes de iniciar o run completo."
        )
    return {
        "method": "tokens_x_aggregate_throughput",
        "is_estimate": True,
        "cu_per_hour": cu_per_hour,
        "throughput_tps_ref": throughput_tps,
        "ref_loops": ref_loops,
        "mean_tokens": dict(mean_tokens),
        "items_per_bench": items,
        "configs": configs,
        "eval_quick_only": quick_only,
        "total": total,
        "estimated_seconds": total["seconds"],
        "estimated_hours": total["hours"],
        "estimated_cu": total["cu"],
        "requires_confirmation": requires_confirmation,
        "assumptions": [
            "tokens/item de L1, L4 e Qwen iguais aos medidos em L2",
            "MBPP+ usa tokens/item de HumanEval+; SciCode usa os de LCB (não medidos no smoke)",
            "custo por token proporcional a 8 + 20k + 8 blocos (normalizado em k=2); Qwen = Dory L2 x qwen_compute_factor",
            "throughput de lote pequeno vs lotes de 128 e cauda de respostas longas podem mudar o tempo real",
        ],
    }
