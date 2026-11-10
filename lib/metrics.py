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


def estimate_compute_cost(
    *,
    smoke_elapsed_s: float,
    smoke_n_items: int,
    full_n_items: int,
    cu_per_hour: float = 12.0,
    confirm_threshold_cu: float = 30.0,
) -> dict[str, Any]:
    """Estimate wall-clock time and Colab Compute Units (CU) from a smoke run (Section 5, Rule 6).
    A100 consumes ~11-13 CU/h (default 12.0 CU/h). Prints the estimate and flags if > 30 CU.
    """
    if smoke_n_items <= 0:
        raise ValueError("smoke_n_items must be > 0")
    sec_per_item = smoke_elapsed_s / float(smoke_n_items)
    est_seconds = sec_per_item * float(full_n_items)
    est_hours = est_seconds / 3600.0
    est_cu = est_hours * cu_per_hour
    requires_confirmation = est_cu > confirm_threshold_cu

    print(
        f"[Estimativa de Custo] Smoke: {smoke_n_items} itens em {smoke_elapsed_s:.1f}s "
        f"({sec_per_item:.2f}s/item) -> Run completo ({full_n_items} itens): "
        f"~{est_hours:.2f} h (~{est_cu:.1f} CU @ {cu_per_hour:.1f} CU/h)."
    )
    if requires_confirmation:
        print(
            f"[ATENÇÃO] Custo estimado ({est_cu:.1f} CU) excede o limite de {confirm_threshold_cu:.1f} CU! "
            "Confirme antes de iniciar o run completo."
        )
    return {
        "sec_per_item": sec_per_item,
        "estimated_seconds": est_seconds,
        "estimated_hours": est_hours,
        "estimated_cu": est_cu,
        "requires_confirmation": requires_confirmation,
    }
