"""F1a: autodestilação em cascata. O Dory L2 tenta cada problema do TACO do modo mais barato ao mais caro
(sem raciocínio -> raciocínio "efficient" -> raciocínio completo), o sandbox roda os testes e o SFT fica
com o trace correto de menos tokens. Um problema resolvido num estágio não entra nos seguintes, então os
fáceis nunca pagam 8 amostras longas.

Saídas no Drive (`data/`):
- `f1a/problems_<id>.jsonl`: problemas escolhidos (enunciado + testes), sem sobreposição com os benchmarks;
- `f1a/samples_<estágio>_<config>.jsonl`: TODAS as amostras, com o resultado do sandbox. As erradas (tipo do
  erro e teste que falhou) são os ramos mortos e os casos de borda do SFT arbóreo, e o p̂ de cada problema
  separa direto de armadilha no currículo e no bônus (1 - p̂)² da F1c (arquitetura_teorica_consolidada_v1);
- `short_cot_sft.jsonl`: trace correto mais curto por problema; CoT vazia só onde o modelo acertou todas
  as amostras sem raciocinar, limitada a ~20% dos exemplos;
- `f1a/stats.json`: acerto e tokens por estágio, redução de tokens do trace escolhido.

Uso (um estágio por processo, porque cada um cria um motor vLLM com contexto diferente):
  python scripts/f1a_self_distill.py --prepare-only                  # só seleção e dedup (sem GPU)
  python scripts/f1a_self_distill.py --stage nothink --n-problems 500
  python scripts/f1a_self_distill.py --stage efficient --n-problems 500
  python scripts/f1a_self_distill.py --stage full --n-problems 500   # o caro: só o que sobrou sem solução
  python scripts/f1a_self_distill.py --build-only                    # remonta o SFT do que já foi gerado
Cada estágio retoma de onde parou e remonta o SFT ao terminar.
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
# O task_id é o índice da linha: sem fixar o commit, uma mudança no Hub trocaria os problemas das amostras guardadas.
DATASET_REV = "177913a7bd43791646ef6a43645caa3c871ab3db"
SELECTION_ID = "v2"  # subir ao mudar os filtros de select_problems (o arquivo de problemas é um cache)
SEED = 42
NGRAM = 13
OVERLAP_THRESHOLD = 0.3  # fração dos 13-gramas em comum com UM problema de benchmark que já conta como cópia
# Problema com mais de uma saída válida reprova solução certa na comparação exata, e ela viraria ramo morto falso.
MULTI_ANSWER_RE = re.compile(
    r"print any|output any|any of them|any (valid|correct|suitable|such)|"
    r"(multiple|several|many) (possible |valid |correct |optimal )?(answers|solutions)",
    re.IGNORECASE,
)
MAX_PROMPT_TOKENS = 3500  # o motor vLLM reserva max_new_tokens + 4096 de contexto
BATCH_PROBLEMS = 64  # x N amostras por chamada ao vLLM; o cache grava ao fim de cada lote
EMPTY_COT_FRAC = 0.2
TEST_TIMEOUT_S = 6.0  # igual ao LCB
TEST_MEMORY_MB = 4096
REF_TPS = 800  # tokens/s do L2 na A100 medidos no F0 (results/f0_cost_estimate.json), só para o teto de custo
A100_CU_PER_HOUR = 8.88

# Cascata, do mais barato ao mais caro. `template` são os kwargs do chat template oficial do Dory
# (chat_template.jinja, conferido em 2026-11-14): enable_thinking=False pré-preenche `<think></think>` e
# medium_effort=True acrescenta "{reasoning effort: efficient}" à mensagem do usuário.
# Teto de 8192 nos estágios com raciocínio: os rollouts da F1c param em 4096, trace maior não treina nada lá.
STAGES = {
    "nothink": {"template": {"enable_thinking": False}, "n_samples": 4, "max_new_tokens": 2048},
    "efficient": {"template": {"enable_thinking": True, "medium_effort": True}, "n_samples": 8, "max_new_tokens": 8192},
    "full": {"template": {"enable_thinking": True}, "n_samples": 8, "max_new_tokens": 8192},
}
EFFORT_MARK = "{reasoning effort: efficient}"


def _solved(stage: str, row: dict) -> bool:
    """Sem raciocínio só conta se acertou todas (vira exemplo de CoT vazia); com raciocínio basta um trace certo."""
    return row["p_hat"] == 1.0 if stage == "nothink" else row["p_hat"] > 0


def build_prompt(tok, problem: str, stage: str) -> str:
    from lib.evals import LCB_SYSTEM_MESSAGE, _build_lcb_prompt

    messages = [
        {"role": "system", "content": LCB_SYSTEM_MESSAGE},
        {"role": "user", "content": _build_lcb_prompt(problem, "")},
    ]
    prompt = str(
        tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **STAGES[stage]["template"])
    )
    # O template ignora calado um kwarg que não conhece: conferir que o modo pedido entrou mesmo no prompt.
    tail = "<think></think>" if stage == "nothink" else "<think>\n"
    if not prompt.endswith(tail) or (EFFORT_MARK in prompt) != (stage == "efficient"):
        raise RuntimeError(f"O chat template não aplicou o modo '{stage}'; final do prompt: {prompt[-120:]!r}")
    return prompt


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

    print(f"[f1a] Carregando {DATASET[0]} ({DATASET[1]}) e os benchmarks para o dedup...")
    ds = load_dataset(*DATASET, split="train", revision=DATASET_REV)
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
        if MULTI_ANSWER_RE.search(row["problem"]):
            dropped["mais de uma saída válida"] += 1
            continue
        if max_overlap(row["problem"]) >= OVERLAP_THRESHOLD:
            dropped["sobreposição com benchmark"] += 1
            continue
        # o prompt do modo efficient é o mais longo dos três
        prompt = build_prompt(tok, row["problem"], "efficient")
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


def stage_pending(problems: list[dict], stage: str, progress: dict) -> list[dict]:
    """O que este estágio ainda gera: problemas que todos os estágios anteriores já rodaram e não resolveram."""
    earlier = list(STAGES)[: list(STAGES).index(stage)]
    pending, waiting, solved = [], 0, 0
    for p in problems:
        tid = p["task_id"]
        if progress[stage].is_done(tid):
            continue
        for s in earlier:  # em ordem: quem foi resolvido cedo nem tem linha nos estágios seguintes
            row = progress[s].get(tid)
            if row is None:
                waiting += 1
                break
            if _solved(s, row):
                solved += 1
                break
        else:
            pending.append(p)
    print(
        f"[f1a] estágio '{stage}': {len(pending)} problemas a gerar; {solved} já resolvidos num estágio anterior; "
        f"{waiting} esperando um estágio anterior rodar."
    )
    return pending


def generate_samples(pending: list[dict], tok, progress, stage: str, *, temperature: float, top_p: float) -> None:
    from lib.dory import DORY_HF_REPO
    from lib.gen import THINK_CLOSE, generate
    from lib.metrics import repetition_score
    from lib.sandbox import check_solution

    n_samples, max_new_tokens = STAGES[stage]["n_samples"], STAGES[stage]["max_new_tokens"]
    worst_h = len(pending) * n_samples * max_new_tokens / REF_TPS / 3600
    print(
        f"[f1a] {n_samples} amostras de até {max_new_tokens} tokens por problema. Teto de custo (toda amostra "
        f"no limite, {REF_TPS} tokens/s): {worst_h:.1f} h de GPU, ~{worst_h * A100_CU_PER_HOUR:.0f} CU na A100."
    )
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
                        # sem raciocínio o </think> já vem no prompt, então só o tamanho diz se cortou
                        "truncated": bool(
                            o.n_total_tokens >= max_new_tokens and (stage == "nothink" or THINK_CLOSE not in o.text)
                        ),
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
        t0 = time.perf_counter()
        outs = generate(
            DORY_HF_REPO,
            [build_prompt(tok, p["problem"], stage) for p in batch],
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
        # Semente fixa com n amostras: se o motor devolver cópias, o lote inteiro sai repetido e a CU vai fora.
        if len(batch) > 1 and all(
            len({o.text for o in outs[j * n_samples : (j + 1) * n_samples]}) == 1 for j in range(len(batch))
        ):
            raise RuntimeError("As amostras de cada problema saíram idênticas: o vLLM não variou a semente.")
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
            f"({n_tokens / sum(gen_times):.0f} tokens/s); faltam {left} problemas, ~{eta_h:.1f} h de GPU "
            f"(~{eta_h * A100_CU_PER_HOUR:.0f} CU na A100)."
        )
    if inflight is not None:
        record(*inflight)
    pool.shutdown()


def build_sft(problems: list[dict], rows: dict[str, dict[str, dict]], empty_frac: float = EMPTY_COT_FRAC):
    """Um exemplo por problema resolvido. `rows[estágio][task_id]` é a linha gravada por `generate_samples`.

    A escolha é pelo total de tokens (CoT + resposta): pela CoT sozinha ganharia o trace que empurra o
    raciocínio para depois do `</think>`. Devolve (exemplos, CoT vazias cortadas pelo limite `empty_frac`).
    """
    examples = []
    for p in problems:
        tid = p["task_id"]
        mine = {s: rows[s][tid] for s in STAGES if tid in rows.get(s, {})}
        nothink = mine.get("nothink")
        if nothink is not None and _solved("nothink", nothink):
            stage, ok = "nothink", nothink["samples"]
        else:
            cands = [(s, x) for s in ("efficient", "full") if s in mine for x in mine[s]["samples"] if x["passed"]]
            if not cands:
                continue
            stage = min(cands, key=lambda c: c[1]["n_total_tokens"])[0]
            ok = [x for s, x in cands if s == stage]
        best = min(ok, key=lambda x: x["n_total_tokens"])
        examples.append(
            {
                "task_id": tid,
                "problem": p["problem"],
                "cot": best["cot"],
                "answer": best["answer"],
                "n_cot_tokens": best["n_cot_tokens"],
                "n_total_tokens": best["n_total_tokens"],
                "empty_cot": stage == "nothink",
                "stage": stage,
                "p_hat": {s: r["p_hat"] for s, r in mine.items()},
                # para a redução pareada: as outras amostras corretas do mesmo problema e estágio
                "n_correct": len(ok),
                "mean_total_correct": statistics.mean(x["n_total_tokens"] for x in ok),
            }
        )
    n_thinking = sum(not e["empty_cot"] for e in examples)
    max_empty = int(empty_frac / (1 - empty_frac) * n_thinking)
    empties = [e for e in examples if e["empty_cot"]]
    cut = {e["task_id"] for e in empties[max_empty:]}
    return [e for e in examples if e["task_id"] not in cut], len(cut)


def sft_stats(problems: list[dict], rows: dict[str, dict[str, dict]], examples: list[dict], n_cut: int) -> dict:
    stages = {}
    for stage in STAGES:
        rs = [rows[stage][p["task_id"]] for p in problems if p["task_id"] in rows.get(stage, {})]
        if not rs:
            continue
        samples = [s for r in rs for s in r["samples"]]
        stages[stage] = {
            "n_problems": len(rs),
            "n_solved": sum(_solved(stage, r) for r in rs),
            "pass1_mean": round(statistics.mean(r["p_hat"] for r in rs), 4),
            "tokens_mean_per_sample": round(statistics.mean(s["n_total_tokens"] for s in samples), 1),
            "truncated_frac": round(sum(s["truncated"] for s in samples) / len(samples), 4),
            "error_types": dict(Counter(str(s["error_type"]) for s in samples if not s["passed"])),
        }
    thinking = [e for e in examples if not e["empty_cot"]]
    paired = [1 - e["n_total_tokens"] / e["mean_total_correct"] for e in thinking if e["n_correct"] >= 2]
    cots = [e["n_cot_tokens"] for e in thinking]
    return {
        "n_problems": len(problems),
        "n_examples": len(examples),
        "n_empty_cot": len(examples) - len(thinking),
        "n_empty_cot_cut_by_cap": n_cut,
        "examples_by_stage": dict(Counter(e["stage"] for e in examples)),
        "cot_mean_kept": round(statistics.mean(cots), 1) if cots else 0.0,
        # serve para calibrar o L_ref da recompensa da F1c
        "cot_median_kept": statistics.median(cots) if cots else 0,
        "n_over_4096_total_tokens": sum(e["n_total_tokens"] > 4096 for e in thinking),  # não cabem no rollout da F1c
        # trace escolhido vs média das amostras CORRETAS do mesmo problema (só problemas com 2 ou mais)
        "reduction_vs_correct_samples": round(statistics.mean(paired), 4) if paired else 0.0,
        "stages": stages,
    }


def _self_check() -> None:
    def row(spec):  # spec: [(total de tokens, passou)]
        samples = [{"n_cot_tokens": n // 2, "n_total_tokens": n, "passed": ok, "truncated": False,
                    "error_type": None if ok else "WrongAnswer", "cot": f"c{n}", "answer": f"a{n}"} for n, ok in spec]
        return {"p_hat": sum(ok for _, ok in spec) / len(spec), "samples": samples}

    ids = ["easy0", "easy1", "flaky", "eff", "full", "none"] + [f"e{i}" for i in range(3)]
    problems = [{"task_id": t, "problem": ""} for t in ids]
    rows = {
        "nothink": {t: row([(30, True), (20, True)]) for t in ("easy0", "easy1")}
        | {t: row([(30, True), (20, False)]) for t in ids[2:]},  # acertar só às vezes não vira CoT vazia
        "efficient": {"flaky": row([(500, True), (300, True)]), "eff": row([(900, False), (400, True)]),
                      "full": row([(800, False)]), "none": row([(800, False)])}
        | {f"e{i}": row([(600, True)]) for i in range(3)},
        "full": {"full": row([(5000, True), (3000, True)]), "none": row([(8192, False)])},
    }
    ex, n_cut = build_sft(problems, rows)
    by = {e["task_id"]: e for e in ex}
    assert "none" not in by and by["flaky"]["answer"] == "a300" and by["eff"]["answer"] == "a400"
    assert by["full"]["stage"] == "full" and by["full"]["answer"] == "a3000"
    assert by["full"]["mean_total_correct"] == 4000
    # 6 exemplos com raciocínio -> cabe int(0.25 * 6) = 1 CoT vazia; a outra é cortada
    assert by["easy0"]["empty_cot"] and by["easy0"]["answer"] == "a20" and "easy1" not in by and n_cut == 1
    stats = sft_stats(problems, rows, ex, n_cut)
    assert stats["stages"]["nothink"]["n_solved"] == 2
    assert stats["examples_by_stage"] == {"nothink": 1, "efficient": 5, "full": 1}

    class Progress(dict):
        is_done = dict.__contains__

    prog = {s: Progress(r) for s, r in rows.items()}
    assert [p["task_id"] for p in stage_pending(problems, "full", prog)] == []  # full e none já rodaram
    del prog["full"]["none"], prog["efficient"]["e0"]
    assert [p["task_id"] for p in stage_pending(problems, "full", prog)] == ["none"]  # e0 espera o efficient


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-problems", type=int, default=500)
    ap.add_argument("--stage", choices=list(STAGES))
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--build-only", action="store_true")
    args = ap.parse_args()
    if not (args.stage or args.prepare_only or args.build_only):
        ap.error(f"escolha --stage ({', '.join(STAGES)}), --prepare-only ou --build-only")
    _self_check()

    from lib.dory import DORY_HF_REPO, enforce_tokenizer_report, load_dory_tokenizer, official_sampling
    from lib.dory import verify_tokenizer_equivalence
    from lib.evals import LCB_SYSTEM_MESSAGE, _build_lcb_prompt
    from lib.io import DEFAULT_DRIVE_ROOT, ResumableProgress, ensure_model_files, setup_drive_and_cache

    paths = setup_drive_and_cache(drive_root=DEFAULT_DRIVE_ROOT, local_fallback_root=repo_root)
    out_dir = paths["data"] / "f1a"
    sampling = official_sampling(DORY_HF_REPO)
    temperature, top_p = float(sampling["temperature"]), float(sampling["top_p"])
    tags = {
        s: f"{s}_L2_n{c['n_samples']}_t{temperature:g}_p{top_p:g}_m{c['max_new_tokens']}" for s, c in STAGES.items()
    }
    progress = {s: ResumableProgress(out_dir / f"samples_{tag}.jsonl") for s, tag in tags.items()}

    ensure_model_files(DORY_HF_REPO)
    tok = load_dory_tokenizer(DORY_HF_REPO)
    problems = select_problems(args.n_problems, tok, out_dir / f"problems_{SELECTION_ID}.jsonl")
    if args.prepare_only:
        return
    if args.stage and not args.build_only:
        pending = stage_pending(problems, args.stage, progress)
        if pending:
            enforce_tokenizer_report(verify_tokenizer_equivalence(DORY_HF_REPO))
            generate_samples(pending, tok, progress[args.stage], args.stage, temperature=temperature, top_p=top_p)

    rows = {s: {p["task_id"]: pr.get(p["task_id"]) for p in problems if pr.is_done(p["task_id"])}
            for s, pr in progress.items()}
    examples, n_cut = build_sft(problems, rows)
    sft_path = paths["data"] / "short_cot_sft.jsonl"
    with sft_path.open("w", encoding="utf-8") as f:
        for e in examples:
            # Prompt de treino = o da avaliação, sem a marca de esforço: o modelo deve ser curto por padrão.
            msgs = {"system": LCB_SYSTEM_MESSAGE, "user": _build_lcb_prompt(e["problem"], "")}
            f.write(json.dumps({**e, **msgs, "samples_file": progress[e["stage"]].cache_file.name},
                               ensure_ascii=False) + "\n")
    stats = {
        "config": {"dataset": "/".join(DATASET), "revision": DATASET_REV, "selection": SELECTION_ID, "seed": SEED,
                   "samples": tags},
        **sft_stats(problems, rows, examples, n_cut),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[f1a] {sft_path} ({len(examples)} exemplos)\n" + json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
