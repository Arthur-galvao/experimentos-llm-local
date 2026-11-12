"""Critical Module: Evaluation pipelines (`eval_quick` and `eval_gate`) with per-problem
resumability, smoke mode, and automatic logging to `results/results.jsonl` (Section 3).
"""

from __future__ import annotations

import base64
import json
import pickle
import time
import zlib
from pathlib import Path
from typing import Any

from .dory import official_sampling
from .gen import THINK_CLOSE, Output, format_chat_prompt, generate
from .io import ResumableProgress
from .metrics import repetition_score
from .registry import DEFAULT_RESULTS_PATH, append_result, has_completed_run
from .sandbox import check_solution
from .scicode_eval import (
    SCICODE_PROTOCOL_ID,
    SCICODE_TEST_TIMEOUT_S,
    build_scicode_problems,
    ensure_scicode_test_data,
    load_scicode_rows,
    make_scicode_extra_metrics_fn,
)

DEFAULT_LCB_QUICK_PATH = Path("data") / "lcb_quick.json"
VLLM_EVAL_BATCH_SIZE = 128
LCB_TIMEOUT_PER_TEST_S = 6.0  # lcb_runner: timeout padrão por teste
# Protocolo LCB: todos os testes (públicos + privados), parada no 1º erro, stdin e funcional.
# Resultados/caches anteriores (testes truncados em 15 e privados não decodificados) não são reaproveitados.
LCB_PROTOCOL_ID = "lcb_official_all_tests_v2"
# Protocolo EvalPlus: base_input + plus_input contra a solução canônica (teste diferencial oficial).
# Resultados antigos (que usavam só o `test`/`assertion` original, sem os testes plus) não são reaproveitados.
EVALPLUS_PROTOCOL_ID = "evalplus_base_plus_v1"


def _decode_lcb_test_cases(raw_field: Any) -> list[dict[str, str]]:
    """Decode LiveCodeBench `public_test_cases` / `private_test_cases` (JSON or zlib+pickle+base64)."""
    if not raw_field:
        return []
    if isinstance(raw_field, list):
        return [
            {"input": str(tc.get("input", "")), "output": str(tc.get("output", ""))}
            for tc in raw_field
            if isinstance(tc, dict)
        ]
    if isinstance(raw_field, str):
        text = raw_field.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [
                    {"input": str(tc.get("input", "")), "output": str(tc.get("output", ""))}
                    for tc in parsed
                    if isinstance(tc, dict)
                ]
        except Exception:
            pass
        try:
            decompressed = zlib.decompress(base64.b64decode(text.encode("utf-8")))
            try:
                parsed = json.loads(decompressed.decode("utf-8"))
            except Exception:
                # Formato oficial (lcb_runner/benchmarks/code_generation.py): base64 -> zlib -> pickle,
                # e o pickle contém uma STRING JSON, que ainda precisa de json.loads.
                parsed = pickle.loads(decompressed)
            if isinstance(parsed, (str, bytes)):
                parsed = json.loads(parsed)
            if isinstance(parsed, list):
                return [
                    {"input": str(tc.get("input", "")), "output": str(tc.get("output", ""))}
                    for tc in parsed
                    if isinstance(tc, dict)
                ]
        except Exception:
            pass
    return []


_LCB_RELEASE_FILES: dict[str, list[str]] = {
    "release_v1": ["test.jsonl"],
    "release_v2": ["test.jsonl", "test2.jsonl"],
    "release_v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
    "release_v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
    "release_v5": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"],
    "release_v6": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
}


def _iter_lcb_rows(version_tag: str):
    """Lê os .jsonl do repositório `livecodebench/code_generation_lite` diretamente.

    O `datasets` 4.x não executa mais scripts de carregamento (`code_generation_lite.py`), então
    baixamos os arquivos que compõem a release (mesma tabela ALLOWED_FILES do script oficial).
    """
    import json as _json

    from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]

    if version_tag not in _LCB_RELEASE_FILES:
        raise ValueError(f"version_tag desconhecida: {version_tag}")
    for fname in _LCB_RELEASE_FILES[version_tag]:
        path = hf_hub_download("livecodebench/code_generation_lite", fname, repo_type="dataset")
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield _json.loads(line)


# Versão do formato dos itens do LCB em disco. v2 = todos os testes (sem corte em 15), testes
# privados decodificados corretamente e `fn_name` (modo funcional). Caches antigos são refeitos.
LCB_SCHEMA_VERSION = 2


def _lcb_fn_name(raw_metadata: Any) -> str | None:
    """Extrai `func_name` do campo `metadata` do LCB (string JSON ou dict); None => teste stdin/stdout."""
    meta = raw_metadata
    if isinstance(meta, str):
        try:
            meta = json.loads(meta) if meta.strip() else {}
        except Exception:
            meta = {}
    if isinstance(meta, dict):
        name = meta.get("func_name")
        return str(name) if name else None
    return None


def load_lcb_v6_dataset(version_tag: str = "release_v6") -> list[dict[str, Any]]:
    """Load `livecodebench/code_generation_lite` (`release_v6`) lendo os .jsonl diretamente."""
    ds = _iter_lcb_rows(version_tag)

    items: list[dict[str, Any]] = []
    for row in ds:
        q_id = str(row.get("question_id") or row.get("id") or len(items))
        pub_tests = _decode_lcb_test_cases(row.get("public_test_cases"))
        priv_tests = _decode_lcb_test_cases(row.get("private_test_cases"))
        # Protocolo oficial: TODOS os testes públicos + privados (sem limite por problema).
        all_tests = pub_tests + priv_tests
        items.append(
            {
                "question_id": q_id,
                "question_title": str(row.get("question_title", "")),
                "question_content": str(row.get("question_content", "")),
                "starter_code": str(row.get("starter_code", "") or ""),
                "contest_date": str(row.get("contest_date", "") or ""),
                "difficulty": str(row.get("difficulty", "") or ""),
                "io_examples": all_tests,
                # Nome da função (LeetCode/funcional) vem de `metadata.func_name`; None => stdin/stdout.
                "fn_name": _lcb_fn_name(row.get("metadata")),
                "lcb_schema": LCB_SCHEMA_VERSION,
            }
        )
    return items


def build_lcb_quick_split(
    output_path: str | Path = DEFAULT_LCB_QUICK_PATH,
    *,
    n_problems: int = 200,
    version_tag: str = "release_v6",
    force: bool = False,
    preloaded_items: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Generate (or load if already present) the fixed 200 most recent problems from
    LiveCodeBench v6 sorted by `contest_date` descending (`data/lcb_quick.json`).
    """
    out = Path(output_path)
    if out.exists() and not force:
        cached = json.loads(out.read_text(encoding="utf-8"))
        stale = any(
            "io_examples" in it and it.get("lcb_schema") != LCB_SCHEMA_VERSION for it in cached
        )
        if not stale:
            return cached
        print(f"[lcb] Cache {out} está em formato antigo (testes truncados/sem fn_name); regenerando.")

    items = preloaded_items if preloaded_items is not None else load_lcb_v6_dataset(version_tag=version_tag)
    sorted_items = sorted(
        items,
        key=lambda x: (str(x.get("contest_date", "")), str(x.get("question_id", ""))),
        reverse=True,
    )
    selected = sorted_items[:n_problems]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")
    return selected


def _build_coding_prompt(
    problem_statement: str,
    *,
    starter_code: str = "",
    is_stdin: bool = False,
) -> str:
    """Create a clear instruction prompt for code generation benchmarks."""
    if is_stdin and not starter_code.strip():
        return (
            "Solve the following competitive programming problem in Python 3. "
            "Read input from standard input (`sys.stdin`) and write the answer to standard output (`print`). "
            "Provide the final solution inside a ```python ... ``` code block.\n\n"
            f"{problem_statement.strip()}"
        )
    if starter_code.strip():
        return (
            "Solve the following programming problem in Python 3. "
            "Complete the required function/class signature below and return the full implementation "
            "inside a ```python ... ``` code block.\n\n"
            f"{problem_statement.strip()}\n\n"
            f"Starter code:\n```python\n{starter_code.strip()}\n```"
        )
    return (
        "Complete the following Python 3 function and return the full implementation "
        "inside a ```python ... ``` code block.\n\n"
        f"```python\n{problem_statement.strip()}\n```"
    )


# Template oficial do LiveCodeBench para geração de código
# (https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/prompts/code_generation.py):
# `PromptConstants.SYSTEM_MESSAGE_GENERIC`, `FORMATTING_MESSAGE_WITH_STARTER_CODE`,
# `FORMATTING_WITHOUT_STARTER_CODE` e `get_generic_question_template_answer`. A mensagem de sistema vai
# como `system` e o corpo como `user`; o chat template do modelo faz o envolvimento (format_chat_prompt).
LCB_SYSTEM_MESSAGE = (
    "You are an expert Python programmer. You will be given a question (problem specification) and will "
    "generate a correct Python program that matches the specification and passes all tests."
)
_LCB_FORMAT_WITH_STARTER = (
    "You will use the following starter code to write the solution to the problem and enclose your code "
    "within delimiters."
)
_LCB_FORMAT_WITHOUT_STARTER = (
    "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on "
    "the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python "
    "program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."
)


def _build_lcb_prompt(question_content: str, starter_code: str = "") -> str:
    """Corpo `user` do prompt oficial do LiveCodeBench (`get_generic_question_template_answer`).
    Não aplica `.strip()` ao enunciado nem ao starter code, igual ao oficial."""
    prompt = f"### Question:\n{question_content}\n\n"
    if starter_code:
        prompt += f"### Format: {_LCB_FORMAT_WITH_STARTER}\n"
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {_LCB_FORMAT_WITHOUT_STARTER}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def _run_problems_loop(
    *,
    problems: list[dict[str, Any]],
    model: Any,
    tokenizer: Any | None,
    run_id: str,
    phase: str,
    model_name: str,
    checkpoint: str,
    n_loops: int | None,
    eval_tier: str,
    bench_name: str,
    max_new_tokens: int,
    temperature: float,
    batch_size: int,
    cache_dir: str | Path,
    results_path: str | Path,
    smoke: bool,
    smoke_n: int,
    force: bool,
    top_p: float = 0.95,
    seed: int = 42,
    backend: str = "vllm",
    extra_config: dict[str, Any] | None = None,
    protocol_id: str | None = None,
    cache_suffix: str = "",
    extra_config_fn: Any | None = None,
) -> dict[str, Any]:
    """Generic resumable evaluation loop over standardized problem dicts.

    Chaves opcionais por problema (usadas pelo SciCode, onde os passos dependem das respostas anteriores):
    - `stage` (int, padrão 0): problemas rodam em ondas por `stage` crescente (uma onda só começa quando a
      anterior terminou), mantendo o lote vLLM dentro de cada onda;
    - `build_prompt(get_record) -> str`: monta o prompt na hora de gerar (substitui `prompt`);
    - `grader(candidate, get_record) -> (SandboxResult, dict)`: substitui `check_solution`; o dict extra
      é gravado junto ao registro do problema.
    `protocol_id` marca o protocolo de avaliação no config e impede reaproveitar resultados antigos
    (de um protocolo diferente) em `results.jsonl`; `cache_suffix` separa o cache por problema;
    `extra_config_fn({task_id: registro}) -> dict` acrescenta métricas ao config final.
    """
    effective_tier = "smoke" if smoke else eval_tier
    allow_hf_fallback = bool(smoke)
    # Amostragem fora da do Dory (Qwen: top_k/presence) entra no protocolo e no cache: resultado e gerações
    # antigas da Qwen, feitas sem elas, não são reaproveitados.
    samp = official_sampling(model_name)
    if samp["top_k"] or samp["presence_penalty"]:
        tag = f"_k{samp['top_k']}_pp{samp['presence_penalty']}"
        cache_suffix += tag
        protocol_id = protocol_id + tag if protocol_id is not None else None

    if not smoke and not force:
        existing = has_completed_run(
            run_id=run_id,
            bench=bench_name,
            eval_tier=effective_tier,
            results_path=results_path,
        )
        if existing is not None and protocol_id is not None and (
            (existing.get("config") or {}).get("eval_protocol") != protocol_id
        ):
            print(
                f"[eval] Resultado antigo de {run_id} em {bench_name} ({effective_tier}) usa outro protocolo "
                f"(esperado '{protocol_id}'); ele NÃO será reaproveitado e a avaliação será refeita."
            )
            existing = None
        if existing is not None:
            print(f"[eval] Skipping {run_id} on {bench_name} ({effective_tier}): already in {results_path}")
            return existing

    active_problems = problems[:smoke_n] if smoke else problems
    if not active_problems:
        raise ValueError(f"No problems loaded for benchmark '{bench_name}'.")

    cache_file = Path(cache_dir) / f"{run_id}_{bench_name}_{effective_tier}{cache_suffix}.jsonl"
    progress = ResumableProgress(cache_file)

    pending = [p for p in active_problems if not progress.is_done(str(p["task_id"]))]
    if pending:
        print(
            f"[eval] Running {bench_name} ({effective_tier}) for {run_id}: "
            f"{len(pending)} pending / {len(active_problems)} total problems..."
        )

    step_size = VLLM_EVAL_BATCH_SIZE if backend == "vllm" else max(1, batch_size)
    # Ondas por `stage` (padrão: tudo na onda 0 => comportamento idêntico ao anterior).
    batches: list[list[dict[str, Any]]] = []
    for stage in sorted({int(p.get("stage", 0)) for p in pending}):
        stage_items = [p for p in pending if int(p.get("stage", 0)) == stage]
        batches.extend(stage_items[i : i + step_size] for i in range(0, len(stage_items), step_size))
    for batch in batches:
        prompts = [
            format_chat_prompt(
                tokenizer,
                p["build_prompt"](progress.get) if p.get("build_prompt") is not None else p["prompt"],
                system_prompt=p.get("system_prompt"),
                enable_thinking=True,
            )
            for p in batch
        ]
        outputs: list[Output] = generate(
            model,
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            n_loops=n_loops,
            n_samples=1,
            tokenizer=tokenizer,
            backend=backend,  # type: ignore[arg-type]
            allow_hf_fallback=allow_hf_fallback,
            batch_size=batch_size,
            top_p=top_p,
            seed=seed,
        )
        for prob, out in zip(batch, outputs):
            # Só a resposta pós-</think> conta; CoT truncada (resposta vazia) é erro, nunca rascunho da CoT.
            candidate = out.answer_text
            extra_fields: dict[str, Any] = {}
            if prob.get("grader") is not None:
                sb, extra_fields = prob["grader"](candidate, progress.get)
            else:
                sb = check_solution(
                    candidate,
                    test_code=prob.get("test_code"),
                    entry_point=prob.get("entry_point"),
                    io_examples=prob.get("io_examples"),
                    fn_name=prob.get("fn_name"),
                    **prob.get("sandbox_kwargs", {}),
                )
            # Truncada = atingiu `max_new_tokens` sem fechar `</think>` (só mede; não altera nada).
            truncated = bool(out.n_total_tokens >= max_new_tokens and THINK_CLOSE not in out.text)
            progress.record(
                str(prob["task_id"]),
                {
                    "passed": sb.passed,
                    "n_cot_tokens": out.n_cot_tokens,
                    "n_total_tokens": out.n_total_tokens,
                    "latency_s": out.latency_s,
                    "backend": out.backend,
                    "error_type": sb.error_type,
                    "truncated": truncated,
                    "loop_score": repetition_score(out.cot_text),  # ~1 = travou repetindo
                    # final da CoT das truncadas, para ler se era loop ou raciocínio ainda andando
                    **({"tail": out.cot_text[-1500:]} if truncated else {}),
                    **extra_fields,
                },
            )

    rows = [progress.get(str(p["task_id"])) for p in active_problems]
    valid_rows = [r for r in rows if r is not None]
    n_done = max(1, len(valid_rows))

    pass1 = sum(1.0 if r.get("passed") else 0.0 for r in valid_rows) / n_done
    cot_mean = sum(float(r.get("n_cot_tokens", 0)) for r in valid_rows) / n_done
    tot_mean = sum(float(r.get("n_total_tokens", 0)) for r in valid_rows) / n_done
    lat_mean = sum(float(r.get("latency_s", 0.0)) for r in valid_rows) / n_done

    # Fração de respostas truncadas (só entre as linhas que têm o campo; caches antigos não têm).
    trunc_flags = [bool(r["truncated"]) for r in valid_rows if "truncated" in r]
    n_truncated = sum(trunc_flags)
    truncated_frac = (n_truncated / len(trunc_flags)) if trunc_flags else None

    recorded_backends = [str(r.get("backend")) for r in valid_rows if r.get("backend")]
    actual_backend = recorded_backends[-1] if recorded_backends else str(backend)

    cfg = {
        "backend": actual_backend,
        "allow_hf_fallback": allow_hf_fallback,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        # aplicados pelo vLLM a partir da amostragem oficial do modelo (lib/dory.py)
        "top_k": samp["top_k"],
        "presence_penalty": samp["presence_penalty"],
        "seed": seed,
        "batch_size": batch_size,
        "n_problems": len(active_problems),
        "smoke": smoke,
        "truncated_frac": round(truncated_frac, 6) if truncated_frac is not None else None,
        "n_truncated": n_truncated,
    }
    if extra_config:
        cfg.update(extra_config)
    if protocol_id is not None:
        cfg["eval_protocol"] = protocol_id
    if extra_config_fn is not None:
        cfg.update(extra_config_fn({str(p["task_id"]): r for p, r in zip(active_problems, rows)}))

    if smoke:
        denom = cot_mean if cot_mean > 0 else max(tot_mean, 1.0)
        return {
            "run_id": run_id,
            "phase": phase,
            "model": model_name,
            "checkpoint": checkpoint,
            "n_loops": n_loops,
            "eval": "smoke",
            "bench": bench_name,
            "backend": actual_backend,
            "pass1": round(pass1, 6),
            "tokens_cot_mean": round(cot_mean, 3),
            "tokens_total_mean": round(tot_mean, 3),
            "latency_mean_s": round(lat_mean, 4),
            "acerto_por_1k_tokens": round(pass1 / (denom / 1000.0), 6),
            "truncated_frac": cfg["truncated_frac"],
            "n_problems": len(active_problems),
            "config": cfg,
        }

    return append_result(
        run_id=run_id,
        phase=phase,
        model=model_name,
        checkpoint=checkpoint,
        n_loops=n_loops,
        eval_tier=effective_tier,
        bench=bench_name,
        pass1=pass1,
        tokens_cot_mean=cot_mean,
        tokens_total_mean=tot_mean,
        latency_mean_s=lat_mean,
        config=cfg,
        results_path=results_path,
    )


_EVALPLUS_GT_CACHE: dict[str, dict[str, list[Any]]] = {}


def _make_evalplus_grader(bench: str, task_id: str, item: dict[str, Any]) -> Any:
    """Cria o `grader(candidate, get_record)` do EvalPlus oficial para um problema (ver
    `lib.sandbox.check_evalplus_solution`). A saída esperada é calculada uma vez por processo,
    executando `prompt + canonical_solution` em base_input + plus_input (como `get_groundtruth`)."""
    from .sandbox import MBPP_OUTPUT_NOT_NONE_TASKS, check_evalplus_solution, compute_evalplus_groundtruth

    dataset = "humaneval" if bench == "humaneval+" else "mbpp"
    entry_point = str(item.get("entry_point", ""))
    cache_key = f"{dataset}:{task_id}"

    def grader(candidate: str, get_record: Any) -> tuple[Any, dict[str, Any]]:
        gt = _EVALPLUS_GT_CACHE.get(cache_key)
        if gt is None:
            gt = compute_evalplus_groundtruth(
                str(item.get("prompt", "")) + str(item.get("canonical_solution", "")),
                entry_point=entry_point,
                base_input=item["base_input"],
                plus_input=item["plus_input"],
                output_not_none=(dataset == "mbpp" and entry_point in MBPP_OUTPUT_NOT_NONE_TASKS),
            )
            _EVALPLUS_GT_CACHE[cache_key] = gt
        sb = check_evalplus_solution(
            candidate,
            dataset=dataset,
            entry_point=entry_point,
            base_input=item["base_input"],
            plus_input=item["plus_input"],
            groundtruth=gt,
            atol=float(item.get("atol", 0) or 0),
        )
        return sb, {"eval_protocol": "evalplus_base_plus"}

    return grader


def _load_evalplus_problems(bench: str) -> list[dict[str, Any]]:
    """Load HumanEval+ or MBPP+ problems using `evalplus` (or HF `evalplus` fallback)."""
    problems: list[dict[str, Any]] = []
    try:
        if bench == "humaneval+":
            from evalplus.data import get_human_eval_plus  # type: ignore[import-not-found]

            raw = get_human_eval_plus()
        else:
            from evalplus.data import get_mbpp_plus  # type: ignore[import-not-found]

            raw = get_mbpp_plus()

        for task_id, item in raw.items():
            prompt_str = str(item.get("prompt", ""))
            entry_point = str(item.get("entry_point", ""))
            if "base_input" in item and "plus_input" in item and "canonical_solution" in item:
                # Protocolo oficial do EvalPlus: teste diferencial contra a solução canônica em
                # base_input + plus_input (o campo `test`/`assertion` do pacote traz só os testes
                # ORIGINAIS do HumanEval/MBPP, sem os testes "plus").
                problems.append(
                    {
                        "task_id": task_id,
                        "prompt": _build_coding_prompt(prompt_str),
                        "entry_point": entry_point,
                        "grader": _make_evalplus_grader(bench, str(task_id), item),
                    }
                )
                continue
            print(
                f"[evalplus] AVISO: {task_id} sem base_input/plus_input/canonical_solution; "
                "usando só o teste original (equivale a HumanEval/MBPP base, NÃO a HumanEval+/MBPP+)."
            )
            test_code = str(item.get("test", "") or item.get("assertion", ""))
            problems.append(
                {
                    "task_id": task_id,
                    "prompt": _build_coding_prompt(prompt_str),
                    "entry_point": entry_point,
                    "test_code": test_code,
                }
            )
        return problems
    except Exception:
        from datasets import load_dataset  # type: ignore[import-not-found]

        hf_name = "evalplus/humanevalplus" if bench == "humaneval+" else "evalplus/mbppplus"
        print(
            f"[evalplus] AVISO: pacote `evalplus` indisponível; usando o dataset HF '{hf_name}' e o campo "
            "`test` dele. TODO: não foi verificado que esse campo inclua os testes plus nem as tolerâncias "
            "de float/oráculos especiais do EvalPlus oficial. Prefira `pip install evalplus`."
        )
        ds = load_dataset(hf_name, split="test")
        for row in ds:
            task_id = str(row.get("task_id"))
            prompt_str = str(row.get("prompt", ""))
            entry_point = str(row.get("entry_point", ""))
            test_code = str(row.get("test", ""))
            problems.append(
                {
                    "task_id": task_id,
                    "prompt": _build_coding_prompt(prompt_str),
                    "entry_point": entry_point,
                    "test_code": test_code,
                }
            )
        return problems


def eval_humaneval_plus(
    model: Any,
    tokenizer: Any | None = None,
    *,
    run_id: str,
    phase: str = "F0",
    model_name: str = "dory-4b",
    checkpoint: str = "base",
    n_loops: int | None = 2,
    eval_tier: str = "quick",
    max_new_tokens: int = 16000,
    temperature: float = 1.0,
    top_p: float = 0.95,
    seed: int = 42,
    backend: str = "vllm",
    batch_size: int = 4,
    cache_dir: str | Path = "eval_cache",
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    smoke: bool = False,
    smoke_n: int = 5,
    force: bool = False,
) -> dict[str, Any]:
    """Run HumanEval+ (164 problems) evaluation."""
    problems = _load_evalplus_problems("humaneval+")
    return _run_problems_loop(
        problems=problems,
        model=model,
        tokenizer=tokenizer,
        run_id=run_id,
        phase=phase,
        model_name=model_name,
        checkpoint=checkpoint,
        n_loops=n_loops,
        eval_tier=eval_tier,
        bench_name="humaneval+",
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        backend=backend,
        batch_size=batch_size,
        cache_dir=cache_dir,
        results_path=results_path,
        smoke=smoke,
        smoke_n=smoke_n,
        force=force,
        protocol_id=EVALPLUS_PROTOCOL_ID if (problems and problems[0].get("grader")) else None,
        cache_suffix="_epv1" if (problems and problems[0].get("grader")) else "",
    )


def eval_mbpp_plus(
    model: Any,
    tokenizer: Any | None = None,
    *,
    run_id: str,
    phase: str = "F0",
    model_name: str = "dory-4b",
    checkpoint: str = "base",
    n_loops: int | None = 2,
    eval_tier: str = "gate",
    max_new_tokens: int = 16000,
    temperature: float = 1.0,
    top_p: float = 0.95,
    seed: int = 42,
    backend: str = "vllm",
    batch_size: int = 4,
    cache_dir: str | Path = "eval_cache",
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    smoke: bool = False,
    smoke_n: int = 5,
    force: bool = False,
) -> dict[str, Any]:
    """Run MBPP+ evaluation."""
    problems = _load_evalplus_problems("mbpp+")
    return _run_problems_loop(
        problems=problems,
        model=model,
        tokenizer=tokenizer,
        run_id=run_id,
        phase=phase,
        model_name=model_name,
        checkpoint=checkpoint,
        n_loops=n_loops,
        eval_tier=eval_tier,
        bench_name="mbpp+",
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        backend=backend,
        batch_size=batch_size,
        cache_dir=cache_dir,
        results_path=results_path,
        smoke=smoke,
        smoke_n=smoke_n,
        force=force,
        protocol_id=EVALPLUS_PROTOCOL_ID if (problems and problems[0].get("grader")) else None,
        cache_suffix="_epv1" if (problems and problems[0].get("grader")) else "",
    )


def eval_lcb_v6(
    model: Any,
    tokenizer: Any | None = None,
    *,
    run_id: str,
    phase: str = "F0",
    model_name: str = "dory-4b",
    checkpoint: str = "base",
    n_loops: int | None = 2,
    subset: str = "quick",
    lcb_quick_path: str | Path = DEFAULT_LCB_QUICK_PATH,
    max_new_tokens: int = 16000,
    temperature: float = 1.0,
    top_p: float = 0.95,
    seed: int = 42,
    backend: str = "vllm",
    batch_size: int = 4,
    cache_dir: str | Path = "eval_cache",
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    smoke: bool = False,
    smoke_n: int = 5,
    force: bool = False,
) -> dict[str, Any]:
    """Run LiveCodeBench v6 evaluation (`subset='quick'` for the 200 most recent problems
    in `data/lcb_quick.json`, or `subset='full'` for the complete `release_v6` benchmark).
    """
    if subset == "quick":
        raw_items = build_lcb_quick_split(output_path=lcb_quick_path)
        bench_name = "lcb_v6_quick"
        eval_tier = "quick"
    else:
        raw_items = load_lcb_v6_dataset(version_tag="release_v6")
        bench_name = "lcb_v6_full"
        eval_tier = "gate"

    problems: list[dict[str, Any]] = []
    for item in raw_items:
        starter = str(item.get("starter_code", "") or "")
        q_content = str(item.get("question_content", ""))
        problems.append(
            {
                "task_id": str(item.get("question_id")),
                "prompt": _build_lcb_prompt(q_content, starter),
                "system_prompt": LCB_SYSTEM_MESSAGE,
                "io_examples": item.get("io_examples", []),
                "fn_name": item.get("fn_name"),
                # LCB oficial: 6 s por teste (parâmetro `timeout` de lcb_runner); memória do processo 4 GB.
                "sandbox_kwargs": {"timeout_s": LCB_TIMEOUT_PER_TEST_S, "max_memory_mb": 4096},
            }
        )

    return _run_problems_loop(
        problems=problems,
        model=model,
        tokenizer=tokenizer,
        run_id=run_id,
        phase=phase,
        model_name=model_name,
        checkpoint=checkpoint,
        n_loops=n_loops,
        eval_tier=eval_tier,
        bench_name=bench_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        backend=backend,
        batch_size=batch_size,
        cache_dir=cache_dir,
        results_path=results_path,
        smoke=smoke,
        smoke_n=smoke_n,
        force=force,
        protocol_id=LCB_PROTOCOL_ID,
        cache_suffix="_lcbv2",
    )


def eval_lcb_pro(
    model: Any,
    tokenizer: Any | None = None,
    *,
    run_id: str,
    phase: str = "F0",
    model_name: str = "dory-4b",
    checkpoint: str = "base",
    n_loops: int | None = 2,
    max_new_tokens: int = 16000,
    temperature: float = 1.0,
    top_p: float = 0.95,
    seed: int = 42,
    backend: str = "vllm",
    batch_size: int = 4,
    cache_dir: str | Path = "eval_cache",
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    smoke: bool = False,
    smoke_n: int = 5,
    force: bool = False,
) -> dict[str, Any] | None:
    """Attempt to evaluate LiveCodeBench Pro locally without external paid APIs.
    Per Section 3.2 of `ESPEC_ANTIGRAVITY.md`, if LCB Pro lacks local self-contained test cases,
    logs the exact reason and keeps LCB v6 as the primary competitive reference.
    """
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]

        ds = load_dataset("livecodebench/livecodebench_pro", split="test")
        problems: list[dict[str, Any]] = []
        for row in ds:
            io_cases = _decode_lcb_test_cases(
                row.get("public_test_cases") or row.get("test_cases")
            )
            if not io_cases:
                continue
            q_id = str(row.get("question_id") or row.get("id") or len(problems))
            starter = str(row.get("starter_code", "") or "")
            content = str(row.get("question_content") or row.get("problem_statement") or "")
            problems.append(
                {
                    "task_id": q_id,
                    "prompt": _build_lcb_prompt(content, starter),
                    "system_prompt": LCB_SYSTEM_MESSAGE,
                    # Todos os casos (sem corte), executados por `run_io_tests_in_sandbox` (parada no 1º erro).
                    # TODO: limites de tempo/memória e modo funcional do LCB Pro não foram confirmados
                    # no código oficial; usa-se o padrão do LCB v6 (6 s por teste, 4 GB, stdin/stdout).
                    "io_examples": io_cases,
                    "sandbox_kwargs": {"timeout_s": LCB_TIMEOUT_PER_TEST_S, "max_memory_mb": 4096},
                }
            )
        if not problems:
            print(
                "[eval_lcb_pro] Dataset 'livecodebench_pro' carregado, mas sem casos de teste locais "
                "auto-contidos para avaliação offline. Mantendo LCB v6 como referência competitiva."
            )
            return None

        return _run_problems_loop(
            problems=problems,
            model=model,
            tokenizer=tokenizer,
            run_id=run_id,
            phase=phase,
            model_name=model_name,
            checkpoint=checkpoint,
            n_loops=n_loops,
            eval_tier="gate",
            bench_name="lcb_pro",
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            backend=backend,
            batch_size=batch_size,
            cache_dir=cache_dir,
            results_path=results_path,
            smoke=smoke,
            smoke_n=smoke_n,
            force=force,
            protocol_id=LCB_PROTOCOL_ID,
            cache_suffix="_lcbv2",
        )
    except Exception as exc:
        print(
            f"[eval_lcb_pro] LCB Pro indisponível para execução local sem API externa ({exc}). "
            "Mantendo LCB v6 como referência competitiva conforme Seção 3.2."
        )
        return None


def eval_scicode(
    model: Any,
    tokenizer: Any | None = None,
    *,
    run_id: str,
    phase: str = "F0",
    model_name: str = "dory-4b",
    checkpoint: str = "base",
    n_loops: int | None = 2,
    max_new_tokens: int = 16000,
    temperature: float = 1.0,
    top_p: float = 0.95,
    seed: int = 42,
    backend: str = "vllm",
    batch_size: int = 4,
    cache_dir: str | Path = "eval_cache",
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    smoke: bool = False,
    smoke_n: int = 5,
    force: bool = False,
    h5py_file: str | Path | None = None,
    data_dir: str | Path | None = None,
    with_background: bool = False,
    scicode_split: str = "test",
    scicode_timeout_s: float = SCICODE_TEST_TIMEOUT_S,
) -> dict[str, Any]:
    """Run SciCode subproblem evaluation seguindo o protocolo oficial (`scicode-bench/SciCode`).

    Cada subproblema é avaliado com o script oficial: dependências + código gerado pelo modelo nos passos
    anteriores + resposta do passo + `targets` lidos de `test_data.h5` + `test_cases` (ver `lib/scicode_eval.py`).
    `pass1` = fração de subproblemas testados que passam; as métricas oficiais (subproblemas / problemas
    principais) ficam no `config` do resultado. Os passos de um mesmo problema dependem uns dos outros, então
    o loop gera em ondas (todos os passos 1, depois todos os passos 2, ...).

    - `h5py_file`: caminho do `test_data.h5`; se omitido, é baixado (Google Drive oficial) para `data_dir`
      (padrão: pasta `data` irmã de `cache_dir`) por `ensure_scicode_test_data`.
    - `with_background=False` é o modo padrão do oficial (o modelo escreve o background como comentário).
    """
    h5_path = Path(h5py_file) if h5py_file else ensure_scicode_test_data(
        data_dir if data_dir is not None else Path(cache_dir).resolve().parent / "data"
    )
    if not h5_path.is_file():
        raise FileNotFoundError(f"test_data.h5 do SciCode não encontrado em {h5_path}")

    rows = load_scicode_rows(scicode_split)
    problems = build_scicode_problems(
        rows, h5_path, with_background=with_background, timeout_s=scicode_timeout_s
    )

    return _run_problems_loop(
        problems=problems,
        model=model,
        tokenizer=tokenizer,
        run_id=run_id,
        phase=phase,
        model_name=model_name,
        checkpoint=checkpoint,
        n_loops=n_loops,
        eval_tier="gate",
        bench_name="scicode",
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        backend=backend,
        batch_size=batch_size,
        cache_dir=cache_dir,
        results_path=results_path,
        smoke=smoke,
        smoke_n=smoke_n,
        force=force,
        protocol_id=SCICODE_PROTOCOL_ID,
        cache_suffix="_official",
        extra_config={
            "scicode_split": scicode_split,
            "scicode_with_background": with_background,
            "scicode_timeout_s": scicode_timeout_s,
        },
        extra_config_fn=make_scicode_extra_metrics_fn(rows),
    )


def eval_custom(
    model: Any,
    tokenizer: Any | None = None,
    *,
    run_id: str,
    subset: str = "quick",
    phase: str = "F0",
    model_name: str = "dory-4b",
    checkpoint: str = "base",
    n_loops: int | None = 2,
    max_new_tokens: int | None = None,
    temperature: float = 1.0,
    top_p: float = 0.95,
    seed: int = 42,
    backend: str = "vllm",
    batch_size: int = 4,
    cache_dir: str | Path = "eval_cache",
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    smoke: bool = False,
    smoke_n: int = 5,
    force: bool = False,
) -> dict[str, Any]:
    """Benchmark custom (`lib/custom_bench.py`): `subset='quick'` (126 problemas) ou `'full'` (504).
    `max_new_tokens=None` usa o orçamento do protocolo (`CUSTOM_MAX_NEW_TOKENS`)."""
    from .custom_bench import CUSTOM_MAX_NEW_TOKENS, CUSTOM_PROTOCOL_ID, build_custom_suite, custom_extra_metrics

    budget = CUSTOM_MAX_NEW_TOKENS if max_new_tokens is None else int(max_new_tokens)
    return _run_problems_loop(
        problems=build_custom_suite(subset),
        model=model,
        tokenizer=tokenizer,
        run_id=run_id,
        phase=phase,
        model_name=model_name,
        checkpoint=checkpoint,
        n_loops=n_loops,
        eval_tier=subset,
        bench_name=f"custom_{subset}",
        max_new_tokens=budget,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        backend=backend,
        batch_size=batch_size,
        cache_dir=cache_dir,
        results_path=results_path,
        smoke=smoke,
        smoke_n=smoke_n,
        force=force,
        protocol_id=CUSTOM_PROTOCOL_ID,
        cache_suffix=f"_{CUSTOM_PROTOCOL_ID}",  # cache por problema não mistura versões do benchmark
        extra_config_fn=lambda rows: custom_extra_metrics(rows, max_new_tokens=budget),
    )


def eval_quick(
    model: Any,
    tokenizer: Any | None = None,
    *,
    run_id: str,
    phase: str = "F0",
    model_name: str = "dory-4b",
    checkpoint: str = "base",
    n_loops: int | None = 2,
    lcb_quick_path: str | Path = DEFAULT_LCB_QUICK_PATH,
    max_new_tokens: int = 16000,
    temperature: float = 1.0,
    top_p: float = 0.95,
    seed: int = 42,
    backend: str = "vllm",
    batch_size: int = 4,
    cache_dir: str | Path = "eval_cache",
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    smoke: bool = False,
    smoke_n: int = 5,
    force: bool = False,
) -> dict[str, dict[str, Any]]:
    """Run the fast iteration suite (`eval_quick`): HumanEval+ and LCB v6 Quick (200 problems)."""
    he_res = eval_humaneval_plus(
        model,
        tokenizer,
        run_id=run_id,
        phase=phase,
        model_name=model_name,
        checkpoint=checkpoint,
        n_loops=n_loops,
        eval_tier="quick",
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        backend=backend,
        batch_size=batch_size,
        cache_dir=cache_dir,
        results_path=results_path,
        smoke=smoke,
        smoke_n=smoke_n,
        force=force,
    )
    lcb_res = eval_lcb_v6(
        model,
        tokenizer,
        run_id=run_id,
        phase=phase,
        model_name=model_name,
        checkpoint=checkpoint,
        n_loops=n_loops,
        subset="quick",
        lcb_quick_path=lcb_quick_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        backend=backend,
        batch_size=batch_size,
        cache_dir=cache_dir,
        results_path=results_path,
        smoke=smoke,
        smoke_n=smoke_n,
        force=force,
    )
    return {
        "humaneval+": he_res,
        "lcb_v6_quick": lcb_res,
    }


def eval_gate(
    model: Any,
    tokenizer: Any | None = None,
    *,
    run_id: str,
    phase: str = "F0",
    model_name: str = "dory-4b",
    checkpoint: str = "base",
    n_loops: int | None = 2,
    max_new_tokens: int = 16000,
    temperature: float = 1.0,
    top_p: float = 0.95,
    seed: int = 42,
    backend: str = "vllm",
    batch_size: int = 4,
    cache_dir: str | Path = "eval_cache",
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    smoke: bool = False,
    smoke_n: int = 5,
    force: bool = False,
) -> dict[str, dict[str, Any]]:
    """Run the full phase-gate suite (`eval_gate`):
    HumanEval+, MBPP+, LiveCodeBench v6 Full, LiveCodeBench Pro (if available offline), and SciCode.
    """
    results: dict[str, dict[str, Any]] = {}
    results["humaneval+"] = eval_humaneval_plus(
        model,
        tokenizer,
        run_id=run_id,
        phase=phase,
        model_name=model_name,
        checkpoint=checkpoint,
        n_loops=n_loops,
        eval_tier="gate",
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        backend=backend,
        batch_size=batch_size,
        cache_dir=cache_dir,
        results_path=results_path,
        smoke=smoke,
        smoke_n=smoke_n,
        force=force,
    )
    results["mbpp+"] = eval_mbpp_plus(
        model,
        tokenizer,
        run_id=run_id,
        phase=phase,
        model_name=model_name,
        checkpoint=checkpoint,
        n_loops=n_loops,
        eval_tier="gate",
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        backend=backend,
        batch_size=batch_size,
        cache_dir=cache_dir,
        results_path=results_path,
        smoke=smoke,
        smoke_n=smoke_n,
        force=force,
    )
    results["lcb_v6_full"] = eval_lcb_v6(
        model,
        tokenizer,
        run_id=run_id,
        phase=phase,
        model_name=model_name,
        checkpoint=checkpoint,
        n_loops=n_loops,
        subset="full",
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        backend=backend,
        batch_size=batch_size,
        cache_dir=cache_dir,
        results_path=results_path,
        smoke=smoke,
        smoke_n=smoke_n,
        force=force,
    )
    lcb_pro_res = eval_lcb_pro(
        model,
        tokenizer,
        run_id=run_id,
        phase=phase,
        model_name=model_name,
        checkpoint=checkpoint,
        n_loops=n_loops,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        backend=backend,
        batch_size=batch_size,
        cache_dir=cache_dir,
        results_path=results_path,
        smoke=smoke,
        smoke_n=smoke_n,
        force=force,
    )
    if lcb_pro_res is not None:
        results["lcb_pro"] = lcb_pro_res

    results["scicode"] = eval_scicode(
        model,
        tokenizer,
        run_id=run_id,
        phase=phase,
        model_name=model_name,
        checkpoint=checkpoint,
        n_loops=n_loops,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        backend=backend,
        batch_size=batch_size,
        cache_dir=cache_dir,
        results_path=results_path,
        smoke=smoke,
        smoke_n=smoke_n,
        force=force,
    )
    return results
