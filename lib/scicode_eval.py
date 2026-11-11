"""SciCode com o protocolo oficial (scicode-bench/SciCode, commit e3158ea).

Espelha o fluxo recomendado do repositório oficial (`eval/inspect_ai/scicode.py`, classes
`ScicodePromptingAssistant` e `ScicodeEvaluator`; equivalente a `eval/scripts/gencode.py` +
`eval/scripts/test_generated_code.py`):

* Prompt por subproblema com o template oficial (`eval/data/background_comment_template.txt`, modo padrão
  SEM background anotado: o modelo escreve o "Background" como comentário; com `with_background=True`
  usa `multistep_template.txt` e a descrição + `step_background` do dataset).
* Passos anteriores do mesmo problema entram no prompt e no script de teste com o código GERADO PELO
  PRÓPRIO MODELO (`extract_python_script(resposta)`), e não com o gold. Exceção oficial: os passos
  13.6, 62.1 e 76.3 (índices 5, 0 e 2) não são gerados nem testados; o código deles vem dos arquivos
  `eval/data/13.6.txt`, `62.1.txt`, `76.3.txt` (copiados em `lib/scicode_data/`).
* Script de teste = `required_dependencies` + código dos passos anteriores + resposta do passo atual +
  `targets = process_hdf5_to_tuple(step_id, n_testes, h5)` + `target = targets[i]` + cada `test_case`.
  Roda com `python script.py` e timeout de 1800 s (oficial). Exit code 0 => passou.
* `test_data.h5` (alvos numéricos) vem do Google Drive oficial; ver `ensure_scicode_test_data`.

Os `test_cases` oficiais importam `scicode.parse.parse` e `scicode.compare.cmp`; usamos cópias
vendorizadas em `lib/scicode_shim/` (PYTHONPATH do subprocesso), sem precisar instalar o pacote `scicode`.
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from .sandbox import SandboxResult, _kill_process_tree

_HERE = Path(__file__).resolve().parent
SCICODE_SHIM_DIR = _HERE / "scicode_shim"
_SCICODE_DATA_DIR = _HERE / "scicode_data"

# Marca o protocolo no config dos resultados (resultados do protocolo antigo, sem `target`, não são reaproveitados).
SCICODE_PROTOCOL_ID = "scicode_official_h5_targets_v2"
SCICODE_H5_FILENAME = "test_data.h5"
# Pasta oficial no Google Drive (README do SciCode, passo 3: "numeric test results").
SCICODE_DRIVE_FOLDER_ID = "1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR"
SCICODE_DRIVE_FOLDER_URL = f"https://drive.google.com/drive/folders/{SCICODE_DRIVE_FOLDER_ID}"
# Tamanho mínimo plausível (bytes) para o h5 baixado; um HTML de erro do Drive é muito menor.
# TODO: confirmar o tamanho/checksum oficial de test_data.h5 (não é publicado no repositório; só dá para ver no Colab).
SCICODE_H5_MIN_BYTES = 1_000_000
SCICODE_TEST_TIMEOUT_S = 1800.0  # oficial: subprocess.run(..., timeout=1800)

# (problem_id, índice do passo) que o oficial não gera nem testa (ver `eval/inspect_ai/scicode.py`).
# Obs.: a FAQ do site cita "76.2", mas o código usa o índice 2 do problema 76 (= passo 76.3); seguimos o código.
SCICODE_SKIPPED_STEPS: frozenset[tuple[str, int]] = frozenset({("13", 5), ("62", 0), ("76", 2)})


# --------------------------------------------------------------------------------------
# Funções auxiliares copiadas do oficial (scicode/parse/parse.py e scicode/gen/models.py)
# --------------------------------------------------------------------------------------
def extract_function_name(function_header: str) -> str:
    """Idêntica a `scicode.parse.parse.extract_function_name`."""
    pattern = r"\bdef\s+(\w+)\s*\("
    match = re.search(pattern, function_header)
    if match:
        return match.group(1)
    pattern = r"\bclass\s+(\w+)\s*\("
    match = re.search(pattern, function_header)
    if match:
        return match.group(1)
    raise ValueError("Function name or class name not found.")


def get_function_from_code(code_string: str | None, function_name: str) -> str | None:
    """Idêntica a `scicode.parse.parse.get_function_from_code`."""
    if code_string is None:
        return None
    try:
        tree = ast.parse(code_string)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == function_name:
                return ast.unparse(node)
    except Exception as e:  # noqa: BLE001 - comportamento oficial
        print(f"{function_name} not found with error: {e}")
        return code_string
    return None


def extract_python_script(response: str) -> str:
    """Idêntica a `scicode.gen.models.extract_python_script` (remove linhas de import do bloco de código)."""
    if "```" in response:
        python_script = (
            response.split("```python")[1].split("```")[0]
            if "```python" in response
            else response.split("```")[1].split("```")[0]
        )
    else:
        python_script = response
    return re.sub(r"^\s*(import .*|from .*\s+import\s+.*)", "", python_script, flags=re.MULTILINE)


def _template(with_background: bool) -> str:
    # No oficial: sem background => DEFAULT_PROMPT_TEMPLATE = background_comment_template.txt;
    # com background => BACKGOUND_PROMPT_TEMPLATE = multistep_template.txt.
    name = "multistep_template.txt" if with_background else "background_comment_template.txt"
    return (_SCICODE_DATA_DIR / name).read_text(encoding="utf-8")


# --------------------------------------------------------------------------------------
# Dados: dataset HF e test_data.h5
# --------------------------------------------------------------------------------------
def load_scicode_rows(split: str = "test") -> list[dict[str, Any]]:
    """Carrega `SciCode1/SciCode` (split 'test' = 65 problemas / 288 subproblemas; 'validation' = 15 / 50).

    Usa `datasets.load_dataset` como o oficial; se falhar (ex.: `datasets` 4.x), lê o .jsonl direto do Hub.
    """
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]

        return [dict(r) for r in load_dataset("SciCode1/SciCode", split=split)]
    except Exception as exc:  # noqa: BLE001
        print(f"[scicode] load_dataset falhou ({type(exc).__name__}: {exc}); lendo o .jsonl direto do Hub.")
        import json

        from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]

        fname = "problems_dev.jsonl" if split in ("validation", "dev") else "problems_test.jsonl"
        path = hf_hub_download("SciCode1/SciCode", fname, repo_type="dataset")
        with open(path, "r", encoding="utf-8") as fh:
            return [json.loads(ln) for ln in fh if ln.strip()]


def _validate_scicode_h5(path: Path) -> None:
    import h5py  # type: ignore[import-not-found]

    with h5py.File(path, "r") as f:
        if len(list(f.keys())) == 0:
            raise ValueError("arquivo h5 sem grupos")


def ensure_scicode_test_data(data_dir: str | Path) -> Path:
    """Garante `data_dir/test_data.h5` (alvos numéricos do SciCode) e devolve o caminho.

    O arquivo NÃO está no HuggingFace nem no GitHub: o README oficial manda baixar da pasta do Google Drive
    https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR e salvar como `test_data.h5`.
    Aqui o download é feito com `gdown` (pré-instalado no Colab), com verificação de existência, tamanho
    mínimo e abertura via h5py. Se falhar, levanta RuntimeError com instruções de download manual.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / SCICODE_H5_FILENAME

    def _ok(p: Path) -> bool:
        if not p.is_file() or p.stat().st_size < SCICODE_H5_MIN_BYTES:
            return False
        try:
            _validate_scicode_h5(p)
            return True
        except Exception:  # noqa: BLE001
            return False

    if _ok(target):
        return target

    manual = (
        f"Baixe manualmente a pasta {SCICODE_DRIVE_FOLDER_URL} e salve o arquivo como {target} "
        "(README oficial do SciCode, passo 3)."
    )
    try:
        import gdown  # type: ignore[import-not-found]
    except ImportError:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
            import gdown  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"`gdown` indisponível ({exc}). {manual}") from exc

    tmp_dl = data_dir / "_scicode_download"
    if tmp_dl.exists():
        shutil.rmtree(tmp_dl)
    tmp_dl.mkdir(parents=True)
    try:
        print(f"[scicode] Baixando {SCICODE_H5_FILENAME} do Google Drive oficial para {data_dir} ...")
        gdown.download_folder(url=SCICODE_DRIVE_FOLDER_URL, output=str(tmp_dl), quiet=False)
        found = sorted(tmp_dl.rglob(SCICODE_H5_FILENAME))
        if not found:
            raise FileNotFoundError(f"{SCICODE_H5_FILENAME} não apareceu na pasta baixada do Drive")
        os.replace(found[0], target)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Falha ao baixar {SCICODE_H5_FILENAME} do Google Drive ({type(exc).__name__}: {exc}). "
            f"O Drive pode ter limitado a cota de downloads. {manual}"
        ) from exc
    finally:
        shutil.rmtree(tmp_dl, ignore_errors=True)

    if not _ok(target):
        size = target.stat().st_size if target.exists() else 0
        raise RuntimeError(
            f"{target} baixado, mas inválido (tamanho {size} bytes < {SCICODE_H5_MIN_BYTES} ou h5 ilegível). {manual}"
        )
    return target


def load_scicode_targets(step_id: str, n_tests: int, h5_path: str | Path) -> list[Any]:
    """Lê os `targets` de um passo do h5 com o MESMO código do oficial (`process_hdf5_to_tuple`)."""
    shim = str(SCICODE_SHIM_DIR)
    if shim not in sys.path:
        sys.path.insert(0, shim)
    from scicode.parse.parse import process_hdf5_to_tuple  # type: ignore[import-not-found]

    return process_hdf5_to_tuple(step_id, n_tests, str(h5_path))


# --------------------------------------------------------------------------------------
# Prompt e script de teste (espelham ScicodePromptingAssistant / ScicodeEvaluator)
# --------------------------------------------------------------------------------------
def scicode_step_id(prob: dict[str, Any], idx: int) -> str:
    return str(prob["sub_steps"][idx].get("step_number") or f"{prob['problem_id']}.{idx + 1}")


def scicode_task_id(prob: dict[str, Any], idx: int) -> str:
    return f"scicode_{scicode_step_id(prob, idx)}"


def is_skipped_step(prob: dict[str, Any], idx: int) -> bool:
    return (str(prob["problem_id"]), idx) in SCICODE_SKIPPED_STEPS


def previous_codes(
    prob: dict[str, Any],
    num_steps: int,
    model_code: Callable[[int], str],
) -> list[str]:
    """Código dos passos 1..num_steps-1: gerado pelo modelo (`model_code(i)`), ou o arquivo oficial
    `eval/data/{problem_id}.{i+1}.txt` para os passos especiais (13.6, 62.1, 76.3)."""
    codes: list[str] = []
    for i in range(num_steps - 1):
        if is_skipped_step(prob, i):
            content = (_SCICODE_DATA_DIR / f"{prob['problem_id']}.{i + 1}.txt").read_text(encoding="utf-8")
            func_name = extract_function_name(prob["sub_steps"][i]["function_header"])
            code = get_function_from_code(content, func_name)
            codes.append(code if code is not None else content)
        else:
            codes.append(model_code(i))
    return codes


def build_scicode_prompt(
    prob: dict[str, Any],
    num_steps: int,
    prev_codes: list[str],
    *,
    with_background: bool = False,
) -> tuple[str, str]:
    """Devolve `(prompt, prefixo_de_código)`; o prefixo é `dependências + código dos passos anteriores`."""
    steps = prob["sub_steps"]

    def _desc(i: int) -> str:
        d = steps[i]["step_description_prompt"]
        return d + "\n" + steps[i]["step_background"] if with_background else d

    output_lines: list[str] = []
    for i in range(num_steps - 1):
        output_lines.append(_desc(i))
        output_lines.append(prev_codes[i])
        output_lines.append("------")
    cur = steps[num_steps - 1]
    next_step = [_desc(num_steps - 1), f"{cur['function_header']}\n\n{cur['return_line']}"]
    problem_steps_str = "\n\n".join(output_lines[:-1])  # remove o último "------"
    next_step_str = "\n\n".join(next_step)
    previous_code_str = "\n".join(prev_codes)
    dependencies = prob["required_dependencies"]
    prompt = _template(with_background).format(
        problem_steps_str=problem_steps_str,
        next_step_str=next_step_str,
        dependencies=dependencies,
    )
    return prompt, f"{dependencies}\n{previous_code_str}\n"


def build_scicode_test_script(
    prob: dict[str, Any],
    num_steps: int,
    code_prefix: str,
    response: str,
    h5_path: str | Path,
) -> str:
    """Monta o script de teste exatamente como `ScicodeEvaluator.test_code` do oficial."""
    sub = prob["sub_steps"][num_steps - 1]
    step_id = scicode_step_id(prob, num_steps - 1)
    test_lst = list(sub["test_cases"])
    parts = [f"{code_prefix}\n{extract_python_script(response)}"]
    parts.append("\n\nfrom scicode.parse.parse import process_hdf5_to_tuple\n\n")
    parts.append(f"targets = process_hdf5_to_tuple('{step_id}', {len(test_lst)}, {str(h5_path)!r})" + "\n")
    for i, test in enumerate(test_lst):
        parts.append(f"target = targets[{i}]\n\n")
        for line in test.split("\n"):
            parts.append(line + "\n")
    return "".join(parts)


def run_scicode_script(script: str, *, timeout_s: float = SCICODE_TEST_TIMEOUT_S, name: str = "step") -> SandboxResult:
    """Executa o script como o oficial (`python script.py`, exit 0 = passou, timeout => falha)."""
    with tempfile.TemporaryDirectory(prefix="scicode_") as tmp:
        script_path = Path(tmp) / f"{name}.py"
        script_path.write_text(script, encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SCICODE_SHIM_DIR), env.get("PYTHONPATH", "")]))
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env.setdefault("MPLBACKEND", "Agg")
        t0 = time.perf_counter()
        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            cwd=tmp,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_process_tree(proc)
            try:
                stdout, stderr = proc.communicate(timeout=1.0)
            except Exception:  # noqa: BLE001
                stdout, stderr = "", "TimeoutExpired"
        elapsed = time.perf_counter() - t0
        code = proc.returncode if proc.returncode is not None else -1
        stderr = (stderr or "")[-2000:]
        error_type: str | None = None
        if timed_out:
            error_type = "TimeoutError"
        elif code != 0:
            last = [ln.strip() for ln in stderr.splitlines() if ln.strip()]
            error_type = last[-1].split(":", 1)[0] if last else f"ExitCode({code})"
        return SandboxResult(
            passed=(not timed_out) and code == 0,
            timed_out=timed_out,
            memory_exceeded="MemoryError" in stderr,
            exit_code=code,
            stdout=(stdout or "")[-2000:],
            stderr=stderr,
            elapsed_s=elapsed,
            error_type=error_type,
        )


# --------------------------------------------------------------------------------------
# Integração com `_run_problems_loop`
# --------------------------------------------------------------------------------------
def build_scicode_problems(
    rows: list[dict[str, Any]],
    h5_path: str | Path,
    *,
    with_background: bool = False,
    timeout_s: float = SCICODE_TEST_TIMEOUT_S,
) -> list[dict[str, Any]]:
    """Uma entrada por subproblema testado (passos especiais ficam de fora, como no oficial).

    `stage` = índice do passo: o loop gera todos os passos 1, depois todos os passos 2, etc., pois o prompt
    do passo k depende da resposta do próprio modelo nos passos anteriores.
    """
    problems: list[dict[str, Any]] = []
    for prob in rows:
        n_steps = len(prob["sub_steps"])
        for idx in range(n_steps):
            if is_skipped_step(prob, idx):
                continue
            problems.append(_make_step_problem(prob, idx, n_steps, h5_path, with_background, timeout_s))
    return problems


def _make_step_problem(
    prob: dict[str, Any],
    idx: int,
    n_steps: int,
    h5_path: str | Path,
    with_background: bool,
    timeout_s: float,
) -> dict[str, Any]:
    def _model_code_getter(get_record: Callable[[str], dict[str, Any] | None]) -> Callable[[int], str]:
        def _get(i: int) -> str:
            rec = get_record(scicode_task_id(prob, i))
            if rec is None or "scicode_code" not in rec:
                raise RuntimeError(
                    f"SciCode: passo {scicode_step_id(prob, i)} ainda não foi gerado, mas o passo "
                    f"{scicode_step_id(prob, idx)} depende dele."
                )
            return str(rec["scicode_code"])

        return _get

    def build_prompt(get_record: Callable[[str], dict[str, Any] | None]) -> str:
        prev = previous_codes(prob, idx + 1, _model_code_getter(get_record))
        return build_scicode_prompt(prob, idx + 1, prev, with_background=with_background)[0]

    def grader(candidate: str, get_record: Callable[[str], dict[str, Any] | None]) -> tuple[SandboxResult, dict[str, Any]]:
        prev = previous_codes(prob, idx + 1, _model_code_getter(get_record))
        _, prefix = build_scicode_prompt(prob, idx + 1, prev, with_background=with_background)
        script = build_scicode_test_script(prob, idx + 1, prefix, candidate, h5_path)
        res = run_scicode_script(script, timeout_s=timeout_s, name=scicode_step_id(prob, idx).replace(".", "_"))
        return res, {"scicode_code": extract_python_script(candidate), "scicode_step": scicode_step_id(prob, idx)}

    return {
        "task_id": scicode_task_id(prob, idx),
        "prompt": "",  # substituído por build_prompt
        "stage": idx,
        "build_prompt": build_prompt,
        "grader": grader,
        "scicode_problem_id": str(prob["problem_id"]),
        "scicode_n_steps": n_steps,
    }


def make_scicode_extra_metrics_fn(rows: list[dict[str, Any]]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Métricas do oficial além do `pass1` (que é a taxa nos subproblemas TESTADOS).

    - `scicode_subproblem_rate_official`: acertos / TODOS os subproblemas (inclui os passos especiais no
      denominador, como o `sub_problem_correctness` do scorer inspect_ai oficial);
    - `scicode_main_rate_official`: problemas principais resolvidos (todos os passos corretos) / problemas;
      como no scorer inspect_ai, problemas com passo especial nunca contam (o passo ignorado fica fora da soma);
    - `scicode_main_rate_tested_steps`: idem, mas exigindo só os passos testados (regra do script
      `test_generated_code.py`), apenas entre problemas cujos passos foram todos avaliados.
    """

    def _fn(rows_by_task: dict[str, Any]) -> dict[str, Any]:
        n_total = n_correct = 0
        main_official = main_tested = n_main_complete = 0
        for prob in rows:
            n_steps = len(prob["sub_steps"])
            tested = [i for i in range(n_steps) if not is_skipped_step(prob, i)]
            recs = [rows_by_task.get(scicode_task_id(prob, i)) for i in tested]
            if any(r is None for r in recs):
                continue  # problema não avaliado por completo (ex.: modo smoke)
            n_main_complete += 1
            ok = sum(1 for r in recs if r.get("passed"))
            n_correct += ok
            n_total += n_steps
            all_ok = ok == len(tested)
            main_tested += int(all_ok)
            main_official += int(all_ok and len(tested) == n_steps)
        return {
            "scicode_n_main_problems_evaluated": n_main_complete,
            "scicode_subproblem_rate_official": (n_correct / n_total) if n_total else None,
            "scicode_main_rate_official": (main_official / n_main_complete) if n_main_complete else None,
            "scicode_main_rate_tested_steps": (main_tested / n_main_complete) if n_main_complete else None,
        }

    return _fn


def run_scicode_gold_check(
    rows: list[dict[str, Any]],
    h5_path: str | Path,
    *,
    with_background: bool = False,
    timeout_s: float = SCICODE_TEST_TIMEOUT_S,
    max_problems: int | None = None,
) -> dict[str, Any]:
    """Sanidade (modo `gold` do oficial): usa `ground_truth_code` como resposta de cada passo e confere que
    tudo passa com o h5 baixado. Só funciona em splits com gabarito (`validation`). Rode no Colab após baixar o h5."""
    n_pass = n_steps_tested = 0
    failed: list[str] = []
    for prob in rows[: max_problems or len(rows)]:
        codes: dict[int, str] = {}
        for idx in range(len(prob["sub_steps"])):
            if is_skipped_step(prob, idx):
                continue
            gold = f"Blah blah\n```python\n{prob['sub_steps'][idx]['ground_truth_code']}\n```\n"
            prev = previous_codes(prob, idx + 1, lambda i: codes[i])
            _, prefix = build_scicode_prompt(prob, idx + 1, prev, with_background=with_background)
            script = build_scicode_test_script(prob, idx + 1, prefix, gold, h5_path)
            res = run_scicode_script(script, timeout_s=timeout_s)
            codes[idx] = extract_python_script(gold)
            n_steps_tested += 1
            if res.passed:
                n_pass += 1
            else:
                failed.append(scicode_step_id(prob, idx))
    return {"n_steps": n_steps_tested, "n_pass": n_pass, "failed": failed}
