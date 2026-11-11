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

from .gen import Output, format_chat_prompt, generate
from .io import ResumableProgress
from .registry import DEFAULT_RESULTS_PATH, append_result, has_completed_run
from .sandbox import check_solution

DEFAULT_LCB_QUICK_PATH = Path("data") / "lcb_quick.json"
VLLM_EVAL_BATCH_SIZE = 128


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
                parsed = pickle.loads(decompressed)
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


def load_lcb_v6_dataset(version_tag: str = "release_v6") -> list[dict[str, Any]]:
    """Load `livecodebench/code_generation_lite` (`release_v6`) lendo os .jsonl diretamente."""
    ds = _iter_lcb_rows(version_tag)

    items: list[dict[str, Any]] = []
    for row in ds:
        q_id = str(row.get("question_id") or row.get("id") or len(items))
        pub_tests = _decode_lcb_test_cases(row.get("public_test_cases"))
        priv_tests = _decode_lcb_test_cases(row.get("private_test_cases"))
        # Keep a bounded number of private tests per problem to avoid excessive I/O in quick eval
        all_tests = (pub_tests + priv_tests)[:15]
        items.append(
            {
                "question_id": q_id,
                "question_title": str(row.get("question_title", "")),
                "question_content": str(row.get("question_content", "")),
                "starter_code": str(row.get("starter_code", "") or ""),
                "contest_date": str(row.get("contest_date", "") or ""),
                "difficulty": str(row.get("difficulty", "") or ""),
                "io_examples": all_tests,
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
        return json.loads(out.read_text(encoding="utf-8"))

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
) -> dict[str, Any]:
    """Generic resumable evaluation loop over standardized problem dicts."""
    effective_tier = "smoke" if smoke else eval_tier
    allow_hf_fallback = bool(smoke)

    if not smoke and not force:
        existing = has_completed_run(
            run_id=run_id,
            bench=bench_name,
            eval_tier=effective_tier,
            results_path=results_path,
        )
        if existing is not None:
            print(f"[eval] Skipping {run_id} on {bench_name} ({effective_tier}): already in {results_path}")
            return existing

    active_problems = problems[:smoke_n] if smoke else problems
    if not active_problems:
        raise ValueError(f"No problems loaded for benchmark '{bench_name}'.")

    cache_file = Path(cache_dir) / f"{run_id}_{bench_name}_{effective_tier}.jsonl"
    progress = ResumableProgress(cache_file)

    pending = [p for p in active_problems if not progress.is_done(str(p["task_id"]))]
    if pending:
        print(
            f"[eval] Running {bench_name} ({effective_tier}) for {run_id}: "
            f"{len(pending)} pending / {len(active_problems)} total problems..."
        )

    step_size = VLLM_EVAL_BATCH_SIZE if backend == "vllm" else max(1, batch_size)
    for start in range(0, len(pending), step_size):
        batch = pending[start : start + step_size]
        prompts = [
            format_chat_prompt(tokenizer, p["prompt"], enable_thinking=True)
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
            candidate = out.answer_text if out.answer_text.strip() else out.text
            sb = check_solution(
                candidate,
                test_code=prob.get("test_code"),
                entry_point=prob.get("entry_point"),
                io_examples=prob.get("io_examples"),
            )
            progress.record(
                str(prob["task_id"]),
                {
                    "passed": sb.passed,
                    "n_cot_tokens": out.n_cot_tokens,
                    "n_total_tokens": out.n_total_tokens,
                    "latency_s": out.latency_s,
                    "backend": out.backend,
                    "error_type": sb.error_type,
                },
            )

    rows = [progress.get(str(p["task_id"])) for p in active_problems]
    valid_rows = [r for r in rows if r is not None]
    n_done = max(1, len(valid_rows))

    pass1 = sum(1.0 if r.get("passed") else 0.0 for r in valid_rows) / n_done
    cot_mean = sum(float(r.get("n_cot_tokens", 0)) for r in valid_rows) / n_done
    tot_mean = sum(float(r.get("n_total_tokens", 0)) for r in valid_rows) / n_done
    lat_mean = sum(float(r.get("latency_s", 0.0)) for r in valid_rows) / n_done

    recorded_backends = [str(r.get("backend")) for r in valid_rows if r.get("backend")]
    actual_backend = recorded_backends[-1] if recorded_backends else str(backend)

    cfg = {
        "backend": actual_backend,
        "allow_hf_fallback": allow_hf_fallback,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "batch_size": batch_size,
        "n_problems": len(active_problems),
        "smoke": smoke,
    }
    if extra_config:
        cfg.update(extra_config)

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
                "prompt": _build_coding_prompt(
                    q_content,
                    starter_code=starter,
                    is_stdin=(not starter.strip()),
                ),
                "io_examples": item.get("io_examples", []),
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
                    "prompt": _build_coding_prompt(
                        content,
                        starter_code=starter,
                        is_stdin=(not starter.strip()),
                    ),
                    "io_examples": io_cases[:15],
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
) -> dict[str, Any]:
    """Run SciCode subproblem evaluation (`SciCode1/SciCode` on HuggingFace / `scicode-bench`)."""
    from datasets import load_dataset  # type: ignore[import-not-found]

    ds = load_dataset("SciCode1/SciCode", split="test")
    problems: list[dict[str, Any]] = []
    for row in ds:
        prob_id = str(row.get("problem_id", len(problems)))
        sub_steps = row.get("sub_steps") or []
        if isinstance(sub_steps, list) and sub_steps:
            for s_idx, step in enumerate(sub_steps):
                if not isinstance(step, dict):
                    continue
                step_id = str(step.get("step_number") or f"{prob_id}.{s_idx + 1}")
                desc = str(step.get("step_description_prompt") or step.get("description") or "")
                header = str(step.get("function_header") or "")
                test_list = step.get("test_cases") or []
                test_code = "\n".join(str(t) for t in test_list) if isinstance(test_list, list) else str(test_list)
                if not desc and not header:
                    continue
                problems.append(
                    {
                        "task_id": f"scicode_{step_id}",
                        "prompt": _build_coding_prompt(desc, starter_code=header),
                        "test_code": test_code,
                    }
                )
        else:
            desc = str(row.get("problem_description_main") or row.get("description") or "")
            header = str(row.get("function_header") or "")
            test_code = str(row.get("test_cases") or "")
            problems.append(
                {
                    "task_id": f"scicode_{prob_id}",
                    "prompt": _build_coding_prompt(desc, starter_code=header),
                    "test_code": test_code,
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
