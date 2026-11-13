"""F1a: autodestilação. O Dory L2 gera N amostras por problema do TACO, o sandbox roda os testes e o SFT
fica com o trace correto de menor CoT.

Saídas no Drive (`data/`):
- `f1a/problems.jsonl`: problemas escolhidos (enunciado + testes), já sem sobreposição com os benchmarks;
- `f1a/samples_<config>.jsonl`: TODAS as amostras, com o resultado do sandbox. As erradas (tipo do erro e
  teste que falhou) são os ramos mortos e os casos de borda do SFT arbóreo, e o p̂ de cada problema
  separa direto de armadilha no currículo e no bônus (1 - p̂)² da F1c (arquitetura_teorica_consolidada_v1);
- `short_cot_sft.jsonl`: menor trace correto por problema, com ~20% de exemplos de CoT vazia;
- `f1a/stats.json`: redução de tokens e distribuição do p̂.

Uso:
  python scripts/f1a_self_distill.py --prepare-only     # só seleção e dedup (sem GPU)
  python scripts/f1a_self_distill.py --n-problems 500   # gera, corrige e monta o SFT; retoma de onde parou
  python scripts/f1a_self_distill.py --build-only       # remonta o SFT a partir das amostras já geradas
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# TACO-verified filtrado pelo DeepCoder (MIT, enunciados humanos, >= 5 testes por problema). Ficam fora:
# KodCode (gerado pelo GPT-4o), primeintellect (sintético) e lcbv5 (são problemas do LCB v6 da avaliação).
DATASET = ("agentica-org/DeepCoder-Preview-Dataset", "taco")
SEED = 42
NGRAM = 13
OVERLAP_THRESHOLD = 0.3  # fração dos 13-gramas em comum com UM problema de benchmark que já conta como cópia
MAX_PROMPT_TOKENS = 3500  # o motor vLLM reserva max_new_tokens + 4096 de contexto
BATCH_PROBLEMS = 64  # x N amostras por chamada ao vLLM; o cache grava ao fim de cada lote
EMPTY_COT_FRAC = 0.2
TEST_TIMEOUT_S = 6.0  # igual ao LCB
TEST_MEMORY_MB = 4096


def _ngrams(text: str) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(words[i : i + NGRAM]) for i in range(len(words) - NGRAM + 1)}


def _bench_texts() -> dict[str, str]:
    """Enunciados de tudo que mede as fases ou a demonstração final."""
    from lib.custom_bench import build_custom_suite
    from lib.evals import _load_evalplus_problems, load_lcb_v6_dataset

    texts = {f"lcb/{p['question_id']}": p["question_content"] for p in load_lcb_v6_dataset("release_v6")}
    for bench in ("humaneval+", "mbpp+"):
        texts.update({f"{bench}/{p['task_id']}": p["prompt"] for p in _load_evalplus_problems(bench)})
    texts.update({f"custom/{p['task_id']}": p["prompt"] for p in build_custom_suite("full")})
    return texts


def _overlap_checker(bench: dict[str, str]):
    index: dict[tuple[str, ...], list[str]] = {}
    sizes: dict[str, int] = {}
    for bid, text in bench.items():
        grams = _ngrams(text)
        sizes[bid] = len(grams)
        for g in grams:
            index.setdefault(g, []).append(bid)

    def max_overlap(text: str) -> float:
        grams = _ngrams(text)
        hits = Counter(b for g in grams for b in index.get(g, ()))
        return max((h / max(1, min(len(grams), sizes[b])) for b, h in hits.items()), default=0.0)

    return max_overlap


def select_problems(n: int, tok, problems_path: Path) -> list[dict]:
    """Os n primeiros problemas válidos numa ordem embaralhada fixa: aumentar n só acrescenta problemas."""
    if problems_path.exists():
        cached = [json.loads(line) for line in problems_path.open(encoding="utf-8")]
        if len(cached) >= n:
            return cached[:n]

    from datasets import load_dataset

    from lib.evals import LCB_SYSTEM_MESSAGE, _build_lcb_prompt
    from lib.gen import format_chat_prompt

    print(f"[f1a] Carregando {DATASET[0]} ({DATASET[1]}) e os benchmarks para o dedup...")
    ds = load_dataset(*DATASET, split="train")
    max_overlap = _overlap_checker(_bench_texts())
    order = list(range(len(ds)))
    random.Random(SEED).shuffle(order)

    chosen, dropped = [], Counter()
    for idx in order:
        if len(chosen) >= n:
            break
        row = ds[idx]
        try:
            tests = json.loads(row["tests"])
        except (TypeError, ValueError):
            tests = None
        if not isinstance(tests, dict):
            dropped["testes ilegíveis"] += 1
            continue
        ins, outs = tests.get("inputs"), tests.get("outputs")
        if tests.get("fn_name"):  # shortcut: só stdin/stdout; o funcional do TACO não traz a assinatura no enunciado
            dropped["funcional"] += 1
            continue
        if not isinstance(ins, list) or not isinstance(outs, list) or len(ins) < 5 or len(ins) != len(outs) or not all(
            isinstance(x, str) for x in ins + outs
        ):
            dropped["testes inválidos"] += 1
            continue
        if max_overlap(row["problem"]) >= OVERLAP_THRESHOLD:
            dropped["sobreposição com benchmark"] += 1
            continue
        prompt = format_chat_prompt(
            tok, _build_lcb_prompt(row["problem"], ""), system_prompt=LCB_SYSTEM_MESSAGE, enable_thinking=True
        )
        if len(tok.encode(prompt, add_special_tokens=False)) > MAX_PROMPT_TOKENS:
            dropped["prompt longo"] += 1
            continue
        chosen.append(
            {
                "task_id": f"taco/{idx}",
                "source_idx": idx,
                "problem": row["problem"],
                "tests": [{"input": i, "output": o} for i, o in zip(ins, outs)],
            }
        )
    print(f"[f1a] {len(chosen)} problemas escolhidos; descartados: {dict(dropped)}")
    if len(chosen) < n:
        print(f"[f1a] AVISO: só {len(chosen)} problemas válidos (pedidos {n}).")
    problems_path.parent.mkdir(parents=True, exist_ok=True)
    with problems_path.open("w", encoding="utf-8") as f:
        for p in chosen:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return chosen


def generate_samples(problems, tok, progress, *, n_samples, max_new_tokens, temperature, top_p) -> None:
    from lib.dory import DORY_HF_REPO
    from lib.evals import LCB_SYSTEM_MESSAGE, _build_lcb_prompt
    from lib.gen import THINK_CLOSE, format_chat_prompt, generate
    from lib.metrics import repetition_score
    from lib.sandbox import check_solution

    pending = [p for p in problems if not progress.is_done(p["task_id"])]
    print(f"[f1a] {len(pending)} problemas pendentes de {len(problems)} ({n_samples} amostras cada).")
    pool = ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2))

    def record(batch, outs, futs) -> None:
        for j, p in enumerate(batch):
            samples = []
            for s in range(n_samples):
                o, sb = outs[j * n_samples + s], futs[j * n_samples + s].result()
                samples.append(
                    {
                        "cot": o.cot_text,
                        "answer": o.answer_text,
                        "n_cot_tokens": o.n_cot_tokens,
                        "n_total_tokens": o.n_total_tokens,
                        "passed": sb.passed,
                        "error_type": sb.error_type,
                        "error_msg": sb.stderr[:300],  # WA traz o índice do teste que falhou
                        "truncated": bool(o.n_total_tokens >= max_new_tokens and THINK_CLOSE not in o.text),
                        "loop_score": repetition_score(o.cot_text),
                    }
                )
            progress.record(
                p["task_id"],
                {
                    "source_idx": p["source_idx"],
                    "p_hat": sum(s["passed"] for s in samples) / n_samples,
                    "samples": samples,
                },
            )

    inflight, gen_times, n_tokens = None, [], 0
    for start in range(0, len(pending), BATCH_PROBLEMS):
        batch = pending[start : start + BATCH_PROBLEMS]
        prompts = [
            format_chat_prompt(
                tok, _build_lcb_prompt(p["problem"], ""), system_prompt=LCB_SYSTEM_MESSAGE, enable_thinking=True
            )
            for p in batch
        ]
        t0 = time.perf_counter()
        outs = generate(
            DORY_HF_REPO,
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            n_loops=2,
            n_samples=n_samples,
            tokenizer=tok,
            backend="vllm",
            top_p=top_p,
            seed=SEED,
        )
        gen_times.append(time.perf_counter() - t0)
        assert len(outs) == len(batch) * n_samples, (len(outs), len(batch), n_samples)
        # A correção do lote anterior roda na CPU enquanto este gerava; se a sessão cair, perde-se 1 lote.
        if inflight is not None:
            record(*inflight)
        futs = [
            pool.submit(
                check_solution,
                o.answer_text,
                io_examples=batch[k // n_samples]["tests"],
                timeout_s=TEST_TIMEOUT_S,
                max_memory_mb=TEST_MEMORY_MB,
            )
            for k, o in enumerate(outs)
        ]
        inflight = (batch, outs, futs)
        n_tokens += sum(o.n_total_tokens for o in outs)
        left = len(pending) - start - len(batch)
        eta_h = statistics.mean(gen_times) * (left / BATCH_PROBLEMS) / 3600
        print(
            f"[f1a] lote {start // BATCH_PROBLEMS + 1}: {len(batch)} problemas em {gen_times[-1] / 60:.1f} min "
            f"({n_tokens / sum(gen_times):.0f} tokens/s); faltam {left} problemas, ~{eta_h:.1f} h de GPU."
        )
    if inflight is not None:
        record(*inflight)
    pool.shutdown()


def build_sft(rows: list[dict], problems: dict[str, dict], empty_frac: float = EMPTY_COT_FRAC):
    """Menor CoT correta por problema. CoT vazia vai para os problemas que o modelo acerta sempre (p̂ = 1)
    com a menor CoT, até `empty_frac` dos exemplos: lá a resposta não depende do raciocínio."""
    examples = []
    for r in rows:
        ok = [s for s in r["samples"] if s["passed"]]
        if not ok:
            continue
        best = min(ok, key=lambda s: s["n_cot_tokens"])
        examples.append(
            {
                "task_id": r["item_id"],
                "problem": problems[r["item_id"]]["problem"],
                "cot": best["cot"],
                "answer": best["answer"],
                "n_cot_tokens": best["n_cot_tokens"],
                "empty_cot": False,
                "p_hat": r["p_hat"],
            }
        )
    always = sorted((e for e in examples if e["p_hat"] == 1.0), key=lambda e: e["n_cot_tokens"])
    for e in always[: int(empty_frac * len(examples))]:
        e.update(empty_cot=True, cot="", orig_n_cot_tokens=e["n_cot_tokens"], n_cot_tokens=0)
    return examples


def sft_stats(rows: list[dict], examples: list[dict]) -> dict:
    every = [s for r in rows for s in r["samples"]]
    correct = [s for s in every if s["passed"]]
    kept = [e["n_cot_tokens"] for e in examples if not e["empty_cot"]]
    mean_all = statistics.mean(s["n_cot_tokens"] for s in every) if every else 0.0
    mean_kept = statistics.mean(kept) if kept else 0.0
    p_hats = [r["p_hat"] for r in rows]
    return {
        "n_problems": len(rows),
        "n_examples": len(examples),
        "n_empty_cot": sum(e["empty_cot"] for e in examples),
        "solve_rate_any": round(len(examples) / max(1, len(rows)), 4),
        "pass1_mean": round(statistics.mean(p_hats), 4) if p_hats else 0.0,
        "cot_mean_all_samples": round(mean_all, 1),
        "cot_mean_correct_samples": round(statistics.mean(s["n_cot_tokens"] for s in correct), 1) if correct else 0.0,
        "cot_mean_kept": round(mean_kept, 1),
        "cot_median_kept": statistics.median(kept) if kept else 0,
        # redução pedida na ESPEC: trace escolhido vs média de todas as amostras
        "reduction_vs_all_samples": round(1 - mean_kept / mean_all, 4) if mean_all else 0.0,
        "truncated_frac": round(sum(s["truncated"] for s in every) / max(1, len(every)), 4),
        "error_types": dict(Counter(str(s["error_type"]) for s in every if not s["passed"])),
        "p_hat_hist": {
            "0": sum(p == 0 for p in p_hats),
            "(0,0.5]": sum(0 < p <= 0.5 for p in p_hats),
            "(0.5,1)": sum(0.5 < p < 1 for p in p_hats),
            "1": sum(p == 1 for p in p_hats),
        },
    }


def _self_check() -> None:
    def row(tid, lens_passed):
        samples = [{"n_cot_tokens": n, "passed": ok, "truncated": False, "error_type": None if ok else "WrongAnswer",
                    "cot": f"c{n}", "answer": f"a{n}"} for n, ok in lens_passed]
        return {"item_id": tid, "p_hat": sum(ok for _, ok in lens_passed) / len(lens_passed), "samples": samples}

    rows = [row(f"t{i}", [(100 + i, True), (50 + i, True)]) for i in range(5)] + [
        row("hard", [(900, False), (300, True)]),
        row("none", [(10, False), (20, False)]),
    ]
    ex = build_sft(rows, {r["item_id"]: {"problem": ""} for r in rows})
    by = {e["task_id"]: e for e in ex}
    assert "none" not in by and by["hard"]["cot"] == "c300" and not by["hard"]["empty_cot"]
    assert [e["task_id"] for e in ex if e["empty_cot"]] == ["t0"]  # int(0.2 * 6) = 1, o mais curto dos p̂ = 1
    assert by["t0"]["cot"] == "" and by["t0"]["orig_n_cot_tokens"] == 50 and by["t1"]["cot"] == "c51"


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-problems", type=int, default=500)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=16384)  # amostra truncada conta como erro
    ap.add_argument("--temperature", type=float, default=None, help="padrão: a oficial do Dory")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--build-only", action="store_true")
    args = ap.parse_args()
    _self_check()

    from lib.dory import DORY_HF_REPO, enforce_tokenizer_report, load_dory_tokenizer, official_sampling
    from lib.dory import verify_tokenizer_equivalence
    from lib.evals import LCB_SYSTEM_MESSAGE, _build_lcb_prompt
    from lib.io import DEFAULT_DRIVE_ROOT, ResumableProgress, ensure_model_files, setup_drive_and_cache

    paths = setup_drive_and_cache(drive_root=DEFAULT_DRIVE_ROOT, local_fallback_root=repo_root)
    out_dir = paths["data"] / "f1a"
    sampling = official_sampling(DORY_HF_REPO)
    temperature = float(sampling["temperature"] if args.temperature is None else args.temperature)
    top_p = float(sampling["top_p"])
    tag = f"L2_n{args.n_samples}_t{temperature:g}_p{top_p:g}_m{args.max_new_tokens}"
    progress = ResumableProgress(out_dir / f"samples_{tag}.jsonl")

    ensure_model_files(DORY_HF_REPO)
    tok = load_dory_tokenizer(DORY_HF_REPO)
    problems = select_problems(args.n_problems, tok, out_dir / "problems.jsonl")
    if args.prepare_only:
        return
    if not args.build_only:
        enforce_tokenizer_report(verify_tokenizer_equivalence(DORY_HF_REPO))
        generate_samples(
            problems, tok, progress,
            n_samples=args.n_samples, max_new_tokens=args.max_new_tokens, temperature=temperature, top_p=top_p,
        )

    by_id = {p["task_id"]: p for p in problems}
    rows = [progress.get(p["task_id"]) for p in problems if progress.is_done(p["task_id"])]
    examples = build_sft(rows, by_id)
    sft_path = paths["data"] / "short_cot_sft.jsonl"
    with sft_path.open("w", encoding="utf-8") as f:
        for e in examples:
            # prompt pronto (o mesmo da geração): a F1b só aplica o chat template
            msgs = {"system": LCB_SYSTEM_MESSAGE, "user": _build_lcb_prompt(e["problem"], "")}
            f.write(json.dumps({**e, **msgs, "samples_file": progress.cache_file.name}, ensure_ascii=False) + "\n")
    stats = {"config": {"dataset": "/".join(DATASET), "samples": tag, "seed": SEED}, **sft_stats(rows, examples)}
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[f1a] {sft_path} ({len(examples)} exemplos)\n" + json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
