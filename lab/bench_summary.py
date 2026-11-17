"""Generate bench_summary.json and bench_summary.md for a lab validation run.

Imports `_row_metrics` and `_paired` from `scripts/t4_validate.py` via `importlib`
without modifying `t4_validate.py`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

A100_REF_TOK_S = 800.0
LAB_N_PCS = 15


def _load_t4_validate(repo_root: Path):
    """Dynamically import `scripts/t4_validate.py` as a module."""
    mod_path = repo_root / "scripts" / "t4_validate.py"
    if not mod_path.exists():
        raise FileNotFoundError(f"scripts/t4_validate.py not found at {mod_path}")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    spec = importlib.util.spec_from_file_location("t4_validate", str(mod_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def _summarize_smoke(json_path: Path, error_path: Path) -> dict[str, Any]:
    data = _read_json(json_path)
    if data is not None:
        return {
            "status": "OK",
            "tok_s": data.get("tok_s"),
            "kv_cache_size_tokens": data.get("kv_cache_size_tokens"),
            "max_concurrency": data.get("max_concurrency"),
            "load_s": data.get("load_s"),
            "gen_s": data.get("gen_s"),
            "tokens_generated": data.get("tokens_generated"),
            "attention_backend": data.get("attention_backend"),
            "error": None,
        }
    if error_path.exists():
        err_text = error_path.read_text(encoding="utf-8", errors="replace").strip()
        return {
            "status": "FALHOU",
            "tok_s": None,
            "kv_cache_size_tokens": None,
            "max_concurrency": None,
            "load_s": None,
            "gen_s": None,
            "tokens_generated": None,
            "attention_backend": None,
            "error": err_text,
        }
    return {
        "status": "NÃO RODOU",
        "tok_s": None,
        "kv_cache_size_tokens": None,
        "max_concurrency": None,
        "load_s": None,
        "gen_s": None,
        "tokens_generated": None,
        "attention_backend": None,
        "error": None,
    }


def _choose_mode(eager: dict[str, Any], graphs: dict[str, Any]) -> str | None:
    eager_ok = eager["status"] == "OK"
    graphs_ok = graphs["status"] == "OK"
    if eager_ok and graphs_ok:
        e_tps = eager.get("tok_s") or 0.0
        g_tps = graphs.get("tok_s") or 0.0
        return "graphs" if g_tps > e_tps else "eager"
    if graphs_ok:
        return "graphs"
    if eager_ok:
        return "eager"
    return None


def _fmt(val: Any, suffix: str = "") -> str:
    if val is None:
        return "—"
    return f"{val}{suffix}"


def _render_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    host = summary["hostname"]
    lines.append(f"# Resumo do Benchmark no Lab — `{host}`")
    lines.append("")
    lines.append(f"- Gerado em: `{summary['generated_at']}`")
    if summary.get("git_sha"):
        lines.append(f"- Commit do código (`git_sha`): `{summary['git_sha']}`")
    if summary.get("lab_vram_gib") is not None:
        lines.append(f"- Orçamento de VRAM (`LAB_VRAM_GIB`): `{summary['lab_vram_gib']} GiB`")
    lines.append("")

    env = summary.get("env") or {}
    lines.append("## 1. Ambiente")
    lines.append("")
    lines.append("| Item | Valor |")
    lines.append("|---|---|")
    lines.append(f"| GPU | {_fmt(env.get('gpu'))} |")
    lines.append(f"| Compute Capability | {_fmt(env.get('capability'))} |")
    lines.append(f"| VRAM total | {_fmt(env.get('vram_gib'), ' GiB')} |")
    lines.append(f"| Driver NVIDIA | {_fmt(summary.get('driver_version'))} |")
    lines.append(f"| RAM do sistema | {_fmt(env.get('ram_gib'), ' GiB')} |")
    lines.append(f"| PyTorch | {_fmt(env.get('torch'))} |")
    lines.append(f"| vLLM | {_fmt(env.get('vllm'))} |")
    lines.append("")

    smokes = summary["smokes"]
    lines.append("## 2. Smoke Tests (contexto 12288 tokens)")
    lines.append("")
    lines.append("Eager e CUDA graphs: 16 sequências × 512 tokens. Longo: 4 sequências × até 4096 tokens, "
                 "o regime mais próximo da F1a com tempo limitado (veja `tokens_generated` no JSON).")
    lines.append("")
    lines.append("| Modo | Status | tok/s | KV cache (tokens) | Concorrência máx. | Carga (s) |")
    lines.append("|---|---|---|---|---|---|")
    rows = [("Eager (`--enforce-eager`)", "eager"), ("CUDA Graphs (`--no-enforce-eager`)", "graphs"),
            ("Longo 4 × 4096 tokens (modo escolhido)", "long")]
    for label, key in rows:
        s = smokes.get(key) or _summarize_smoke(Path("-"), Path("-"))
        lines.append(
            f"| {label} | {s['status']} | {_fmt(s['tok_s'])} | {_fmt(s['kv_cache_size_tokens'])} | "
            f"{_fmt(s['max_concurrency'], 'x' if s['max_concurrency'] is not None else '')} | {_fmt(s['load_s'])} |"
        )
    lines.append("")
    chosen = smokes.get("chosen_mode")
    if chosen == "graphs":
        lines.append("- **Modo escolhido:** CUDA Graphs (`--no-enforce-eager`) por ser mais rápido que eager.")
    elif chosen == "eager":
        lines.append("- **Modo escolhido:** Eager (`--enforce-eager`).")
    else:
        lines.append("- **Modo escolhido:** nenhum smoke completou com sucesso.")

    for label, key in [("Eager", "eager"), ("CUDA Graphs", "graphs"), ("Longo", "long")]:
        err = (smokes.get(key) or {}).get("error")
        if err:
            tail = "\n".join(err.splitlines()[-10:])
            lines.append("")
            lines.append(f"<details><summary>Erro no smoke {label}</summary>\n\n```\n{tail}\n```\n</details>")
    lines.append("")

    tp = summary["throughput"]
    proj = summary["projections"]
    lines.append("## 3. Throughput Real em Sequência Longa (F1a `efficient`) e Projeção do Farm")
    lines.append("")
    if tp["n_runs"] > 0:
        lines.append(f"- Problemas concluídos (`samples_efficient_int8.jsonl`): **{tp['n_problems_done']}**")
        lines.append(f"- Chamadas medidas (`throughput_int8.jsonl`): **{tp['n_runs']}**")
        lines.append(f"- Tokens gerados: **{tp['total_tokens']}** em **{tp['total_gen_s']:.1f} s** de geração (carga total: {tp['total_load_s']:.1f} s)")
        lines.append(f"- **Throughput real por PC (RTX 2060 SUPER INT8):** **{tp['real_tok_s']} tok/s**")
        lines.append("")
        lines.append("| Cenário | tok/s |")
        lines.append("|---|---|")
        lines.append(f"| 1 PC — RTX 2060 SUPER (INT8, real) | {proj['per_pc_tok_s']} |")
        lines.append(f"| {proj['n_pcs']} PCs — Farm do lab (projeção linear ×{proj['n_pcs']}) | {proj['farm_15pcs_tok_s']} |")
        lines.append(f"| Referência — 1× A100 (BF16) | {proj['a100_ref_tok_s']:.0f} |")
        lines.append(f"| Razão Farm ({proj['n_pcs']} PCs) / A100 | {proj['farm_vs_a100_ratio']}× |")
    else:
        lines.append("Nenhum dado de throughput longo disponível (etapa `quality` pulada ou sem problemas concluídos).")
    lines.append("")

    q = summary["quality"]
    lines.append("## 4. Qualidade Pareada: RTX 2060 SUPER (INT8) vs A100 (BF16)")
    lines.append("")
    lines.append(
        "> **AVISO:** Com poucos problemas, esta comparação é **apenas indicativa**. "
        "Estes números (INT8 / lab) **nunca** entram nos gates do projeto (gates rodam exclusivamente na A100 em BF16)."
    )
    lines.append("")
    if q["n_problems"] > 0:
        lines.append(f"Problemas em comum comparados (mesmos `item_id`s): **{q['n_problems']}**")
        lines.append("")
        lines.append("| Métrica | A100 (BF16) | RTX 2060 SUPER (INT8) | Δ (INT8 − BF16) | IC 95% |")
        lines.append("|---|---|---|---|---|")
        for key in ("p_hat", "solved", "tokens", "truncated", "menor_trace_correto"):
            if key in q["metrics"]:
                m = q["metrics"][key]
                lines.append(
                    f"| {key} | {m['a100_bf16']} | {m['rtx_2060s_int8']} | {m['delta']:+} | {m['ic95']} |"
                )
    else:
        lines.append("Sem problemas pareados para comparar qualidade.")
    lines.append("")
    return "\n".join(lines)


def generate_summary(
    *,
    out_dir: Path,
    drive_root: Path,
    repo_root: Path,
    hostname: str,
    driver_version: str | None = None,
    git_sha: str | None = None,
    lab_vram_gib: float | None = None,
) -> dict[str, Any]:
    """Build `bench_summary.json` and `bench_summary.md` inside `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    t4_mod = _load_t4_validate(repo_root)
    from lib.io import ResumableProgress

    t4_out = drive_root / "results" / "t4_validation"
    env = _read_json(out_dir / "env.json") or _read_json(t4_out / "env.json") or {}

    eager = _summarize_smoke(out_dir / "smoke_int8_eager.json", out_dir / "smoke_int8_eager.error.txt")
    graphs = _summarize_smoke(out_dir / "smoke_int8_graphs.json", out_dir / "smoke_int8_graphs.error.txt")
    chosen_mode = _choose_mode(eager, graphs)
    long = _summarize_smoke(out_dir / "smoke_int8_long.json", out_dir / "smoke_int8_long.error.txt")

    tp_path = out_dir / "quality" / "throughput_int8.jsonl"
    if not tp_path.exists():
        tp_path = t4_out / "quality" / "throughput_int8.jsonl"
    runs = _read_jsonl(tp_path)

    cand_path = out_dir / "quality" / "samples_efficient_int8.jsonl"
    if not cand_path.exists():
        cand_path = t4_out / "quality" / "samples_efficient_int8.jsonl"
    ref_path = drive_root / "data" / "f1a" / "samples_efficient_L2_n8_t1_p0.95_m8192.jsonl"

    n_problems_done = 0
    cand_records: list[dict[str, Any]] = []
    cand_prog = None
    if cand_path.exists():
        cand_prog = ResumableProgress(cand_path)
        cand_records = cand_prog.all_records()
        n_problems_done = len(cand_records)

    total_tokens = sum(int(r.get("tokens", 0)) for r in runs)
    total_load_s = sum(float(r.get("load_s", 0)) for r in runs)
    total_gen_s = sum(max(1.0, float(r.get("elapsed_s", 0)) - float(r.get("load_s", 0))) for r in runs) if runs else 0.0
    real_tok_s = round(total_tokens / max(1.0, total_gen_s), 1) if runs else None

    if real_tok_s is not None:
        farm_tok_s = round(real_tok_s * LAB_N_PCS, 1)
        farm_ratio = round(farm_tok_s / A100_REF_TOK_S, 2)
    else:
        farm_tok_s = None
        farm_ratio = None

    quality_metrics: dict[str, Any] = {}
    paired_ids: list[str] = []
    if cand_prog is not None and ref_path.exists():
        ref_prog = ResumableProgress(ref_path)
        paired_ids = [str(r["item_id"]) for r in cand_records if ref_prog.is_done(str(r["item_id"]))]
        if paired_ids:
            mr = [t4_mod._row_metrics(ref_prog.get(pid)) for pid in paired_ids]
            mc = [t4_mod._row_metrics(cand_prog.get(pid)) for pid in paired_ids]
            for key in ("p_hat", "solved", "tokens", "truncated"):
                p = t4_mod._paired([float(m[key]) for m in mr], [float(m[key]) for m in mc])
                quality_metrics[key] = {
                    "a100_bf16": p["ref"],
                    "rtx_2060s_int8": p["t4"],
                    "delta": p["delta"],
                    "ic95": p["ic95"],
                }
            both = [
                (float(r["tokens_correct"]), float(c["tokens_correct"]))
                for r, c in zip(mr, mc)
                if r["tokens_correct"] is not None and c["tokens_correct"] is not None
            ]
            if both:
                p_min = t4_mod._paired([b[0] for b in both], [b[1] for b in both])
                quality_metrics["menor_trace_correto"] = {
                    "a100_bf16": p_min["ref"],
                    "rtx_2060s_int8": p_min["t4"],
                    "delta": p_min["delta"],
                    "ic95": p_min["ic95"],
                }

    summary: dict[str, Any] = {
        "hostname": hostname,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "driver_version": driver_version,
        "git_sha": git_sha,
        "lab_vram_gib": lab_vram_gib,
        "env": env,
        "smokes": {
            "eager": eager,
            "graphs": graphs,
            "long": long,
            "chosen_mode": chosen_mode,
        },
        "throughput": {
            "n_runs": len(runs),
            "n_problems_done": n_problems_done,
            "total_tokens": total_tokens,
            "total_gen_s": round(total_gen_s, 1),
            "total_load_s": round(total_load_s, 1),
            "real_tok_s": real_tok_s,
            "runs": runs,
        },
        "projections": {
            "per_pc_tok_s": real_tok_s,
            "n_pcs": LAB_N_PCS,
            "farm_15pcs_tok_s": farm_tok_s,
            "a100_ref_tok_s": A100_REF_TOK_S,
            "farm_vs_a100_ratio": farm_ratio,
        },
        "quality": {
            "labels": {
                "ref": "A100 (BF16)",
                "cand": "RTX 2060 SUPER (INT8)",
            },
            "n_problems": len(paired_ids),
            "indicative_only": True,
            "note": (
                "Com poucos problemas esta comparação é apenas indicativa e nunca entra nos gates "
                "do projeto (gates rodam exclusivamente na A100 em BF16)."
            ),
            "metrics": quality_metrics,
        },
    }

    (out_dir / "bench_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "bench_summary.md").write_text(
        _render_markdown(summary),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--drive-root", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--hostname", required=True)
    ap.add_argument("--driver-version", default=None)
    ap.add_argument("--git-sha", default=None)
    ap.add_argument("--lab-vram-gib", type=float, default=None)
    args = ap.parse_args()

    generate_summary(
        out_dir=args.out_dir,
        drive_root=args.drive_root,
        repo_root=args.repo_root,
        hostname=args.hostname,
        driver_version=args.driver_version,
        git_sha=args.git_sha,
        lab_vram_gib=args.lab_vram_gib,
    )


if __name__ == "__main__":
    main()
