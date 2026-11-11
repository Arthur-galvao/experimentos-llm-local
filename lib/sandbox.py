"""Critical Module: Safe subprocess sandbox for executing untrusted model-generated code
with strict wall-clock timeout and OS-level memory limits (Section 4 F1 & Section 6).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass
class SandboxResult:
    """Result of executing candidate code inside the isolated sandbox."""

    passed: bool
    timed_out: bool
    memory_exceeded: bool
    exit_code: int
    stdout: str
    stderr: str
    elapsed_s: float
    error_type: str | None = None


def extract_python_code(text: str) -> str:
    """Extract Python solution code from model output.

    If `<think>...</think>` is still attached, strips the thinking block first.
    If markdown fenced code blocks exist, returns the last ````python ... ```` block
    (since reasoning models often refine code and put the final solution last).
    """
    if not text:
        return ""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    elif "<think>" in text:
        # Unclosed think block -> no final answer produced
        return ""

    matches = _CODE_BLOCK_RE.findall(text)
    if matches:
        return matches[-1].strip()
    return text.strip()


_RUNNER_TEMPLATE = r'''import os
import sys

MAX_MEM_BYTES = int(sys.argv[1])
TARGET_SCRIPT = sys.argv[2]


def _apply_memory_limit(limit_bytes: int) -> None:
    # Mesmo mecanismo do EvalPlus oficial (`reliability_guard` em evalplus/eval/utils.py): limites
    # do SO via `resource`, SEM rastreamento por linha (isso deixava o codigo dezenas de vezes
    # mais lento e reprovava solucoes corretas por timeout).
    if limit_bytes <= 0:
        return
    try:
        import resource
        for name in ("RLIMIT_AS", "RLIMIT_DATA"):
            lim = getattr(resource, name, None)
            if lim is None:
                continue
            try:
                soft, hard = resource.getrlimit(lim)
                new_hard = limit_bytes if hard == resource.RLIM_INFINITY else min(hard, limit_bytes)
                resource.setrlimit(lim, (new_hard, new_hard))
            except Exception:
                pass
    except Exception:
        pass  # sem `resource` (Windows): so o timeout de processo do pai vale


_apply_memory_limit(MAX_MEM_BYTES)

with open(TARGET_SCRIPT, "rb") as _f:
    _code_bytes = _f.read()

_globals = {
    "__name__": "__main__",
    "__file__": TARGET_SCRIPT,
    "__builtins__": __builtins__,
}
exec(compile(_code_bytes, TARGET_SCRIPT, "exec"), _globals)
'''


def _kill_process_tree(proc: subprocess.Popen[Any]) -> None:
    """Forcefully terminate a timed-out child process and any grandchildren."""
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=3,
                check=False,
            )
        else:
            try:
                os.killpg(proc.pid, 9)  # processo lançado com start_new_session=True
            except Exception:
                proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def run_code_in_sandbox(
    code: str,
    *,
    stdin_data: str | None = None,
    timeout_s: float = 4.0,
    max_memory_mb: int = 512,
) -> SandboxResult:
    """Execute Python `code` in an isolated temporary directory and subprocess
    with strict `timeout_s` and `max_memory_mb` limits.
    """
    limit_bytes = int(max_memory_mb) * 1024 * 1024 if max_memory_mb > 0 else 0

    with tempfile.TemporaryDirectory(prefix="sandbox_") as tmpdir:
        tmp_path = Path(tmpdir)
        runner_path = tmp_path / "_runner.py"
        target_path = tmp_path / "solution.py"

        runner_path.write_text(_RUNNER_TEMPLATE, encoding="utf-8")
        target_path.write_text(code, encoding="utf-8")

        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env["OMP_NUM_THREADS"] = "1"  # como o EvalPlus oficial (evita estourar RLIMIT_AS com threads de BLAS)
        env["OPENBLAS_NUM_THREADS"] = "1"

        t0 = time.perf_counter()
        proc = subprocess.Popen(
            [sys.executable, "-I", str(runner_path), str(limit_bytes), str(target_path)],
            cwd=str(tmp_path),
            stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            start_new_session=(sys.platform != "win32"),
        )

        timed_out = False
        try:
            stdout, stderr = proc.communicate(input=stdin_data, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_process_tree(proc)
            try:
                stdout, stderr = proc.communicate(timeout=1.0)
            except Exception:
                stdout, stderr = "", "TimeoutExpired"

        elapsed = time.perf_counter() - t0
        exit_code = proc.returncode if proc.returncode is not None else -1

        mem_exceeded = (
            "MemoryError" in (stderr or "")
            or exit_code in (-9, 137)  # SIGKILL from OOM
            or (sys.platform == "win32" and exit_code in (-1073741801, 3221225495))  # STATUS_NO_MEMORY
        )

        error_type: str | None = None
        if timed_out:
            error_type = "TimeoutError"
        elif mem_exceeded:
            error_type = "MemoryError"
        elif exit_code != 0:
            last_lines = [ln.strip() for ln in (stderr or "").splitlines() if ln.strip()]
            error_type = last_lines[-1].split(":", 1)[0] if last_lines else f"ExitCode({exit_code})"

        return SandboxResult(
            passed=(not timed_out) and (not mem_exceeded) and (exit_code == 0),
            timed_out=timed_out,
            memory_exceeded=mem_exceeded,
            exit_code=exit_code,
            stdout=stdout or "",
            stderr=stderr or "",
            elapsed_s=elapsed,
            error_type=error_type,
        )


# --- Execução de testes de entrada/saída no estilo do LiveCodeBench oficial ----------------------
# Referência: lcb_runner/evaluation/testing_util.py e compute_code_generation_metrics.py
# (https://github.com/LiveCodeBench/LiveCodeBench). Um único subprocesso por problema executa os
# testes em sequência e PARA no primeiro que falha; o tempo por teste é `timeout_s` (oficial: 6 s)
# e há um limite global por problema (oficial: (timeout + 1) * n_testes + 5).

# Mesmo prelúdio de imports do oficial (permite soluções que esquecem `import`s comuns).
_LCB_IMPORT_STRING = (
    "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\n"
    "from heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\n"
    "from random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\n"
    "from operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\n"
    "from builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\n"
    "import collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\n"
    "import statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\n"
    "import json\nsys.setrecursionlimit(50000)\n"
)

# Teto para o limite global por problema (evita que dezenas de testes lentos explodam o tempo total).
LCB_MAX_TOTAL_TIMEOUT_S = 300.0

_IO_RUNNER_TEMPLATE = r"""import contextlib, faulthandler, io, json, os, signal, sys
from decimal import Decimal

SPEC = json.load(open(sys.argv[1], encoding="utf-8"))
RESULT_PATH = sys.argv[2]
STDIN_PATH = sys.argv[3]
TIMEOUT = float(SPEC["timeout"])


class _Timeout(BaseException):  # BaseException: `except Exception` do usuário não engole o timeout
    pass


def _on_alarm(signum, frame):
    raise _Timeout()


_HAS_ALARM = hasattr(signal, "setitimer")
if _HAS_ALARM:
    signal.signal(signal.SIGALRM, _on_alarm)


def _arm():
    if _HAS_ALARM:
        signal.setitimer(signal.ITIMER_REAL, TIMEOUT)


def _disarm():
    if _HAS_ALARM:
        signal.setitimer(signal.ITIMER_REAL, 0)


def _limit_memory(limit_bytes):
    if limit_bytes <= 0:
        return
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    except Exception:
        pass


def _finish(status, **extra):
    extra["status"] = status
    with open(RESULT_PATH, "w", encoding="utf-8") as fh:
        json.dump(extra, fh)
    sys.stdout.flush()
    os._exit(0)


def _trunc(x, n=300):
    x = x if isinstance(x, str) else repr(x)
    return x if len(x) <= n else x[: n // 2] + "...(truncado)..." + x[-n // 2:]


def _set_stdin(data):
    # stdin real (fd 0) por teste: funciona com sys.stdin, input(), open(0), os.read(0) etc.
    fd = os.open(STDIN_PATH, os.O_RDWR | os.O_CREAT | os.O_TRUNC)
    os.write(fd, data.encode("utf-8"))
    os.lseek(fd, 0, 0)
    os.dup2(fd, 0)
    os.close(fd)
    sys.stdin = io.TextIOWrapper(io.BufferedReader(io.FileIO(0, closefd=False)), encoding="utf-8")


def _stripped_lines(val):
    return [ln.strip() for ln in val.strip().split("\n")]


def _decimals(line):
    try:
        return True, [Decimal(e) for e in line.split()]
    except Exception:
        return False, []


def _outputs_match(pred, expected):
    # Mesma regra do oficial (grade_stdio): linhas com strip; igualdade exata ou, se ambas forem
    # numéricas, igualdade em Decimal; número de linhas precisa coincidir.
    pl, el = _stripped_lines(pred), _stripped_lines(expected)
    if len(pl) != len(el):
        return False
    for a, b in zip(pl, el):
        if a == b:
            continue
        ok_a, da = _decimals(a)
        ok_b, db = _decimals(b)
        if not (ok_a and ok_b and da == db):
            return False
    return True


_limit_memory(int(SPEC["max_mem_bytes"]))
faulthandler.disable()
code = SPEC["import_string"] + "\n\n" + SPEC["code"]
fn_name = SPEC.get("fn_name")
inputs, outputs = SPEC["inputs"], SPEC["outputs"]

try:
    code_obj = compile(code, "<solucao>", "exec")
except BaseException as exc:
    _finish("CompileError", message=_trunc(repr(exc)))

if fn_name:
    # Estilo funcional / LeetCode: uma chamada por teste; entradas são JSON (um argumento por linha).
    ns = {"__name__": "tmp_sol"}
    sink = io.StringIO()
    try:
        _arm()
        with contextlib.redirect_stdout(sink):
            exec(code_obj, ns)
        _disarm()
        sol = ns["Solution"]() if "class Solution" in SPEC["code"] else None
        method = getattr(sol, fn_name) if sol is not None else ns.get(fn_name)
        assert method is not None, "funcao %r nao encontrada" % fn_name
    except _Timeout:
        _finish("TimeLimit", failed_test=-1, message="timeout ao carregar a solucao")
    except BaseException as exc:
        _disarm()
        _finish("RuntimeError", failed_test=-1, message=_trunc(repr(exc)))
    for idx, (raw_in, raw_out) in enumerate(zip(inputs, outputs)):
        try:
            args = [json.loads(line) for line in raw_in.split("\n")]
            expected = json.loads(raw_out)
            _arm()
            with contextlib.redirect_stdout(sink):
                pred = method(*args)
            _disarm()
        except _Timeout:
            _finish("TimeLimit", failed_test=idx, message="Time Limit Exceeded")
        except BaseException as exc:
            _disarm()
            _finish("RuntimeError", failed_test=idx, message=_trunc(repr(exc)))
        if isinstance(pred, tuple):  # oficial: não penaliza tuplas no lugar de listas
            pred = list(pred)
        if pred != expected:
            _finish("WrongAnswer", failed_test=idx,
                    message="esperado %s, obtido %s" % (_trunc(expected), _trunc(pred)))
else:
    # Estilo stdin/stdout (Codeforces / AtCoder).
    for idx, (raw_in, raw_out) in enumerate(zip(inputs, outputs)):
        out = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", write_through=True)
        try:
            _set_stdin(raw_in)
            _arm()
            with contextlib.redirect_stdout(out):
                try:
                    exec(code_obj, {"__name__": "__main__"})
                except SystemExit:
                    pass
            _disarm()
        except _Timeout:
            _finish("TimeLimit", failed_test=idx, message="Time Limit Exceeded")
        except BaseException as exc:
            _disarm()
            _finish("RuntimeError", failed_test=idx, message=_trunc(repr(exc)))
        pred = out.buffer.getvalue().decode("utf-8", errors="replace")
        if not _outputs_match(pred, raw_out):
            _finish("WrongAnswer", failed_test=idx,
                    message="esperado %s, obtido %s" % (_trunc(raw_out), _trunc(pred)))

_finish("Passed", n_tests=len(inputs))
"""


def run_io_tests_in_sandbox(
    code: str,
    inputs: list[str],
    outputs: list[str],
    *,
    fn_name: str | None = None,
    timeout_s: float = 6.0,
    total_timeout_s: float | None = None,
    max_memory_mb: int = 4096,
) -> SandboxResult:
    """Roda `code` contra pares (entrada, saída esperada) num único subprocesso, parando no
    primeiro teste que falha (como o LiveCodeBench oficial).

    - `fn_name=None`: modo stdin/stdout. Saída comparada linha a linha após `strip` (e igualdade
      numérica em `Decimal` quando ambas as linhas forem números), igual ao oficial.
    - `fn_name="metodo"`: modo funcional/LeetCode. Entradas são JSON (um argumento por linha),
      chama `Solution().metodo(*args)` (ou a função de módulo) e compara com `==`.
    - `timeout_s`: limite POR TESTE (oficial: 6 s). `total_timeout_s`: limite global do problema;
      padrão = min((timeout_s + 1) * n_testes + 5, LCB_MAX_TOTAL_TIMEOUT_S).
    """
    n_tests = len(inputs)
    if total_timeout_s is None:
        total_timeout_s = min((timeout_s + 1.0) * n_tests + 5.0, LCB_MAX_TOTAL_TIMEOUT_S)
    limit_bytes = int(max_memory_mb) * 1024 * 1024 if max_memory_mb > 0 else 0

    import json as _json

    with tempfile.TemporaryDirectory(prefix="sandbox_io_") as tmpdir:
        tmp_path = Path(tmpdir)
        runner_path = tmp_path / "_io_runner.py"
        spec_path = tmp_path / "spec.json"
        result_path = tmp_path / "result.json"
        stdin_path = tmp_path / "stdin.txt"
        runner_path.write_text(_IO_RUNNER_TEMPLATE, encoding="utf-8")
        spec_path.write_text(
            _json.dumps(
                {
                    "code": code,
                    "import_string": _LCB_IMPORT_STRING,
                    "inputs": [str(x) for x in inputs],
                    "outputs": [str(x) for x in outputs],
                    "fn_name": fn_name,
                    "timeout": float(timeout_s),
                    "max_mem_bytes": limit_bytes,
                }
            ),
            encoding="utf-8",
        )

        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["OMP_NUM_THREADS"] = "1"

        t0 = time.perf_counter()
        proc = subprocess.Popen(
            [sys.executable, "-I", str(runner_path), str(spec_path), str(result_path), str(stdin_path)],
            cwd=str(tmp_path),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            start_new_session=(sys.platform != "win32"),
        )
        timed_out = False
        try:
            _, stderr = proc.communicate(timeout=total_timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_process_tree(proc)
            try:
                _, stderr = proc.communicate(timeout=1.0)
            except Exception:
                stderr = "TimeoutExpired"
        elapsed = time.perf_counter() - t0
        exit_code = proc.returncode if proc.returncode is not None else -1
        stderr = (stderr or "")[-2000:]

        result: dict[str, Any] | None = None
        if result_path.exists():
            try:
                result = _json.loads(result_path.read_text(encoding="utf-8"))
            except Exception:
                result = None

        def _res(passed: bool, error_type: str | None, msg: str, *, timeout: bool = False, mem: bool = False):
            return SandboxResult(
                passed=passed,
                timed_out=timeout,
                memory_exceeded=mem,
                exit_code=0 if passed else (exit_code if exit_code not in (0, None) else 1),
                stdout="",
                stderr=msg,
                elapsed_s=elapsed,
                error_type=error_type,
            )

        if result is None:
            # Sem resultado: estourou o limite global, foi morto (ex.: OOM) ou o interpretador caiu.
            if timed_out:
                return _res(False, "TimeoutError", f"limite global de {total_timeout_s:.0f}s excedido", timeout=True)
            if "MemoryError" in stderr or exit_code in (-9, 137):
                return _res(False, "MemoryError", stderr or "memória excedida", mem=True)
            return _res(False, "RuntimeError", stderr or f"sem resultado (exit={exit_code})")

        status = str(result.get("status"))
        msg = str(result.get("message", ""))
        if status == "Passed":
            return _res(True, None, "")
        if status == "TimeLimit":
            return _res(False, "TimeoutError", msg, timeout=True)
        if status == "WrongAnswer":
            return _res(False, "WrongAnswer", f"teste {result.get('failed_test')}: {msg}")
        if "MemoryError" in msg:
            return _res(False, "MemoryError", msg, mem=True)
        return _res(False, "RuntimeError" if status == "RuntimeError" else status, msg)


# --- Execução no estilo EvalPlus oficial (HumanEval+ / MBPP+) -------------------------------------
# Referência: https://github.com/evalplus/evalplus/blob/master/evalplus/eval/__init__.py
# (`unsafe_execute`, `untrusted_check`), .../eval/utils.py (`reliability_guard`, `time_limit`),
# .../eval/_special_oracle.py e .../gen/util/__init__.py (`trusted_exec`) e .../evaluate.py
# (`check_correctness`, `get_groundtruth`).
# Teste diferencial: a saída esperada de cada entrada (base_input + plus_input) vem da execução da
# solução canônica ("trusted_exec"); o candidato roda em subprocesso com limite de memória por
# `resource` e limite de tempo por teste = max(1 s, 4 x tempo da solução canônica).

EVALPLUS_DEFAULT_MIN_TIME_LIMIT_S = 1.0       # evalplus.config.DEFAULT_MIN_TIME_LIMIT
EVALPLUS_DEFAULT_GT_TIME_LIMIT_FACTOR = 4.0   # evalplus.config.DEFAULT_GT_TIME_LIMIT_FACTOR
EVALPLUS_TIMEOUT_PER_TASK_S = 60.0            # EVALPLUS_TIMEOUT_PER_TASK (padrão oficial)
EVALPLUS_MAX_MEMORY_MB = 4096                 # EVALPLUS_MAX_MEMORY_BYTES (padrão oficial: 4 GB)
EVALPLUS_GROUNDTRUTH_TIMEOUT_S = 300.0        # o oficial não limita o trusted_exec; aqui há um teto de segurança

# evalplus/eval/_special_oracle.py
MBPP_OUTPUT_NOT_NONE_TASKS = ["check_str", "text_match_three", "text_starta_endb"]
MBPP_OUTPUT_SET_EQ_TASKS = [
    "similar_elements", "find_char_long", "common_in_nested_lists", "extract_singly",
    "larg_nnum", "intersection_array", "find_dissimilar", "Diff",
]

_EVALPLUS_RUNNER_TEMPLATE = r'''import builtins
import faulthandler
import io
import json
import math
import os
import pickle
import signal
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy

try:
    import numpy as np
except Exception:
    np = None

SPEC = pickle.load(open(sys.argv[1], "rb"))
RESULT_PATH = sys.argv[2]
_RESULT_FD = os.open(RESULT_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)  # aberto antes do guard (open fica None)
_write, _exit = os.write, os._exit


def _finish(payload):
    _write(_RESULT_FD, json.dumps(payload).encode("utf-8"))
    _exit(0)


class TimeoutException(BaseException):  # BaseException: `except Exception` do candidato nao engole
    pass


class _WriteOnly(io.StringIO):
    def read(self, *a, **k):
        raise IOError

    def readline(self, *a, **k):
        raise IOError

    def readlines(self, *a, **k):
        raise IOError

    def readable(self, *a, **k):
        return False


class _Quiet:
    def __enter__(self):
        self._sink = _WriteOnly()
        self._cms = [redirect_stdout(self._sink), redirect_stderr(self._sink)]
        for cm in self._cms:
            cm.__enter__()
        self._old_stdin, sys.stdin = sys.stdin, self._sink
        return self

    def __exit__(self, *exc):
        sys.stdin = self._old_stdin
        for cm in reversed(self._cms):
            cm.__exit__(*exc)
        return False


def _on_alarm(signum, frame):
    raise TimeoutException("Timed out!")


def _limit_memory(limit_bytes):
    # evalplus.eval.utils.reliability_guard: RLIMIT_AS e RLIMIT_DATA (o oficial tambem limita RLIMIT_STACK,
    # omitido aqui porque a stack padrao das threads e herdada do valor inicial e nao faz diferenca pratica).
    if limit_bytes <= 0:
        return
    try:
        import resource
        for name in ("RLIMIT_AS", "RLIMIT_DATA"):
            lim = getattr(resource, name, None)
            if lim is not None:
                resource.setrlimit(lim, (limit_bytes, limit_bytes))
    except Exception:
        pass


def _reliability_guard():
    # Copia fiel de evalplus.eval.utils.reliability_guard (desativa operacoes destrutivas). NAO e um sandbox de seguranca.
    faulthandler.disable()
    builtins.exit = None
    builtins.quit = None
    os.environ["OMP_NUM_THREADS"] = "1"
    for name in (
        "kill", "system", "putenv", "remove", "removedirs", "rmdir", "fchdir", "setuid", "fork", "forkpty",
        "killpg", "rename", "renames", "truncate", "replace", "unlink", "fchmod", "fchown", "chmod", "chown",
        "chroot", "lchflags", "lchmod", "lchown", "getcwd", "chdir",
    ):
        setattr(os, name, None)
    builtins.open = None
    import shutil
    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None
    import subprocess
    subprocess.Popen = None
    builtins.help = None
    for mod in ("ipdb", "joblib", "resource", "psutil", "tkinter"):
        sys.modules[mod] = None


def _is_floats(x):
    if isinstance(x, float):
        return True
    if isinstance(x, (list, tuple)) and x:
        return all(isinstance(i, float) for i in x)
    if np is not None and isinstance(x, np.ndarray):
        return x.dtype == np.float64 or x.dtype == np.float32
    return False


def _allclose(a, b, rtol, atol):
    if np is not None:
        return bool(np.allclose(a, b, rtol=rtol, atol=atol))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_allclose(x, y, rtol, atol) for x, y in zip(a, b))
    return abs(a - b) <= atol + rtol * abs(b)


# evalplus/eval/_special_oracle.py
def _surface_Area(base_edge, height):
    slant_height = math.sqrt((base_edge / 2) ** 2 + height ** 2)
    base_area = base_edge ** 2
    lateral_area = 4 * (base_edge * slant_height) / 2
    return round(base_area + lateral_area)


def _digit_distance_nums(num1, num2):
    s1, s2 = str(num1), str(num2)
    n = max(len(s1), len(s2))
    s1, s2 = s1.zfill(n), s2.zfill(n)
    return sum(abs(int(d1) - int(d2)) for d1, d2 in zip(s1, s2))


def _poly(xs, x):
    return sum([coeff * math.pow(x, i) for i, coeff in enumerate(xs)])


def _run_groundtruth():
    # evalplus/gen/util/__init__.py::trusted_exec
    g = {}
    exec(SPEC["code"], g)
    fn = g[SPEC["entry_point"]]
    out = {}
    for key in ("base", "plus"):
        ret, rtime = [], []
        for inp in SPEC[key + "_input"]:
            inp = deepcopy(inp)
            t0 = time.time()
            ret.append(fn(*inp))
            rtime.append(time.time() - t0)
        if SPEC["output_not_none"]:
            ret = [i is not None for i in ret]
        out[key], out[key + "_time"] = ret, rtime
    with open(RESULT_PATH, "wb") as fh:
        pickle.dump(out, fh)
    sys.stdout.flush()
    _exit(0)


def _run_check():
    dataset, entry_point = SPEC["dataset"], SPEC["entry_point"]
    inputs, expected, time_limits = SPEC["inputs"], SPEC["expected"], SPEC["time_limits"]
    atol, fast_check = SPEC["atol"], SPEC["fast_check"]
    code = SPEC["code"]
    details = [False] * len(inputs)
    progress = 0
    _limit_memory(int(SPEC["max_mem_bytes"]))
    _reliability_guard()
    signal.signal(signal.SIGALRM, _on_alarm)
    exec_globals = {}
    try:
        with _Quiet():
            exec(code, exec_globals)
            fn = exec_globals[entry_point]
        for i, inp in enumerate(inputs):
            try:
                signal.setitimer(signal.ITIMER_REAL, time_limits[i])
                try:
                    with _Quiet():
                        out = fn(*inp)
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                exp = expected[i]
                exact_match = out == exp

                # ---------------- special oracles (oficial) ----------------
                if dataset == "mbpp":
                    if "are_equivalent" == entry_point:  # Mbpp/164
                        exact_match = exact_match or True
                    elif "sum_div" == entry_point:  # Mbpp/295
                        exact_match = exact_match or out == 0
                    elif "surface_Area" == entry_point:  # Mbpp/581
                        exact_match = exact_match or abs(out - _surface_Area(*inp)) <= atol
                    elif "digit_distance_nums" == entry_point:  # Mbpp/558
                        exact_match = exact_match or out == _digit_distance_nums(*inp)
                    elif entry_point in SPEC["set_eq_tasks"]:
                        exact_match = set(out) == set(exp)
                    elif entry_point in SPEC["not_none_tasks"]:
                        if isinstance(out, bool):
                            exact_match = out == exp
                        else:
                            exact_match = exp == (out is not None)
                if dataset == "humaneval":
                    if "find_zero" == entry_point:  # HumanEval/32
                        assert abs(_poly(*inp, out)) <= atol
                        details[i] = True
                        progress += 1
                        continue
                # -----------------------------------------------------------

                if atol == 0 and _is_floats(exp):
                    atol = 1e-6  # oficial: forca atol para comparacao de floats
                if not exact_match and atol != 0:
                    assert type(out) == type(exp)
                    if isinstance(exp, (list, tuple)):
                        assert len(out) == len(exp)
                    assert _allclose(out, exp, 1e-07, atol)
                else:
                    assert exact_match
            except BaseException:
                details[i] = False
                progress += 1
                if fast_check:
                    raise
                continue
            details[i] = True
            progress += 1
        stat = "pass"
    except TimeoutException:
        stat = "timeout"
    except BaseException:
        stat = "fail"
    _finish({"stat": stat, "details": details[:progress]})


if SPEC["mode"] == "groundtruth":
    _run_groundtruth()
else:
    _run_check()
'''


def _run_evalplus_runner(
    spec: dict[str, Any], *, timeout_s: float, max_memory_mb: int
) -> tuple[bool, str, dict[str, Any]]:
    """Roda `_EVALPLUS_RUNNER_TEMPLATE` num subprocesso descartável. Retorna `(timed_out, stderr, spec)`;
    o conteúdo do arquivo de resultado (bytes ou None) e o exit code voltam em `spec['_result']` e
    `spec['_exit_code']`, pois o diretório temporário é apagado ao sair."""
    import pickle as _pickle

    limit_bytes = int(max_memory_mb) * 1024 * 1024 if max_memory_mb > 0 else 0
    spec = dict(spec)
    spec["max_mem_bytes"] = limit_bytes
    with tempfile.TemporaryDirectory(prefix="sandbox_ep_") as tmpdir:
        tmp_path = Path(tmpdir)
        runner_path = tmp_path / "_ep_runner.py"
        spec_path = tmp_path / "spec.pkl"
        result_path = tmp_path / "result.bin"
        runner_path.write_text(_EVALPLUS_RUNNER_TEMPLATE, encoding="utf-8")
        with spec_path.open("wb") as fh:
            _pickle.dump(spec, fh)
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        proc = subprocess.Popen(
            [sys.executable, "-I", str(runner_path), str(spec_path), str(result_path)],
            cwd=str(tmp_path),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            start_new_session=(sys.platform != "win32"),
        )
        timed_out = False
        try:
            _, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_process_tree(proc)
            try:
                _, stderr = proc.communicate(timeout=1.0)
            except Exception:
                stderr = "TimeoutExpired"
        spec["_exit_code"] = proc.returncode
        spec["_result"] = result_path.read_bytes() if result_path.exists() else None
        return timed_out, (stderr or "")[-2000:], spec


def compute_evalplus_groundtruth(
    code: str,
    *,
    entry_point: str,
    base_input: list[Any],
    plus_input: list[Any],
    output_not_none: bool = False,
    timeout_s: float = EVALPLUS_GROUNDTRUTH_TIMEOUT_S,
    max_memory_mb: int = EVALPLUS_MAX_MEMORY_MB,
) -> dict[str, list[Any]]:
    """Executa a solução canônica (`prompt + canonical_solution`) em todas as entradas e devolve
    `{"base", "base_time", "plus", "plus_time"}` — equivalente a `get_groundtruth` do EvalPlus oficial."""
    import pickle as _pickle

    timed_out, stderr, spec = _run_evalplus_runner(
        {
            "mode": "groundtruth",
            "code": code,
            "entry_point": entry_point,
            "base_input": base_input,
            "plus_input": plus_input,
            "output_not_none": bool(output_not_none),
        },
        timeout_s=timeout_s,
        max_memory_mb=max_memory_mb,
    )
    raw = spec.get("_result")
    if timed_out or not raw:
        raise RuntimeError(
            f"Falha ao calcular a saída esperada (groundtruth) de '{entry_point}': "
            f"{'timeout' if timed_out else stderr.strip()[-500:] or 'sem resultado'}"
        )
    return _pickle.loads(raw)


def evalplus_untrusted_check(
    dataset: str,
    code: str,
    inputs: list[Any],
    entry_point: str,
    expected: list[Any],
    atol: float,
    ref_time: list[float],
    *,
    fast_check: bool = True,
    min_time_limit: float = EVALPLUS_DEFAULT_MIN_TIME_LIMIT_S,
    gt_time_limit_factor: float = EVALPLUS_DEFAULT_GT_TIME_LIMIT_FACTOR,
    max_memory_mb: int = EVALPLUS_MAX_MEMORY_MB,
) -> tuple[str, list[bool], bool]:
    """Equivalente a `untrusted_check` do EvalPlus. Retorna `(stat, details, memory_exceeded)` com
    `stat` em {"pass", "fail", "timeout"}. O tempo do processo inteiro é
    `min(60, soma dos limites por teste) + 1 (+1 sem fast_check)`, como no oficial."""
    import json as _json

    time_limits = [max(min_time_limit, gt_time_limit_factor * t) for t in ref_time]
    timeout = min(EVALPLUS_TIMEOUT_PER_TASK_S, sum(time_limits)) + 1.0
    if not fast_check:
        timeout += 1.0
    timed_out, stderr, spec = _run_evalplus_runner(
        {
            "mode": "check",
            "dataset": dataset,
            "code": code,
            "entry_point": entry_point,
            "inputs": inputs,
            "expected": expected,
            "time_limits": time_limits,
            "atol": atol,
            "fast_check": fast_check,
            "set_eq_tasks": MBPP_OUTPUT_SET_EQ_TASKS,
            "not_none_tasks": MBPP_OUTPUT_NOT_NONE_TASKS,
        },
        timeout_s=timeout + 1.0,
        max_memory_mb=max_memory_mb,
    )
    raw = spec.get("_result")
    exit_code = spec.get("_exit_code")
    mem = "MemoryError" in stderr or exit_code in (-9, 137)
    if not raw:
        # Sem resultado: estourou o tempo (oficial: stat desconhecido -> TIMEOUT) ou o processo morreu.
        return ("timeout" if timed_out else "fail"), [], mem
    res = _json.loads(raw.decode("utf-8"))
    stat, details = str(res["stat"]), [bool(d) for d in res["details"]]
    if stat == "pass" and (len(details) != len(inputs) or not all(details)):
        stat = "fail"
    return stat, details, mem


def check_evalplus_solution(
    candidate_text_or_code: str,
    *,
    dataset: str,
    entry_point: str,
    base_input: list[Any],
    plus_input: list[Any],
    groundtruth: dict[str, list[Any]],
    atol: float = 0.0,
    base_only: bool = False,
    min_time_limit: float = EVALPLUS_DEFAULT_MIN_TIME_LIMIT_S,
    gt_time_limit_factor: float = EVALPLUS_DEFAULT_GT_TIME_LIMIT_FACTOR,
    max_memory_mb: int = EVALPLUS_MAX_MEMORY_MB,
) -> SandboxResult:
    """Avalia um candidato como `check_correctness` do EvalPlus: passa só se TODOS os testes base E plus
    passam. Para economizar tempo, os testes plus são pulados quando os base já falharam (o resultado
    final, pass/fail, é idêntico). `dataset` é "humaneval" ou "mbpp"."""
    code = extract_python_code(candidate_text_or_code)
    t0 = time.perf_counter()
    if not code.strip():
        return SandboxResult(False, False, False, 1, "", "Empty solution code", 0.0, "EmptyCode")

    stages = [("base", base_input, groundtruth["base"], groundtruth["base_time"])]
    if not base_only:
        stages.append(("plus", plus_input, groundtruth["plus"], groundtruth["plus_time"]))
    for name, inputs, expected, ref_time in stages:
        stat, details, mem = evalplus_untrusted_check(
            dataset, code, inputs, entry_point, expected, atol, ref_time,
            min_time_limit=min_time_limit, gt_time_limit_factor=gt_time_limit_factor,
            max_memory_mb=max_memory_mb,
        )
        if stat != "pass":
            err = "TimeoutError" if stat == "timeout" else ("MemoryError" if mem else "WrongAnswer")
            return SandboxResult(
                passed=False,
                timed_out=(stat == "timeout"),
                memory_exceeded=mem,
                exit_code=1,
                stdout="",
                stderr=f"falhou nos testes {name} (teste {max(len(details) - 1, 0)} de {len(inputs)})",
                elapsed_s=time.perf_counter() - t0,
                error_type=err,
            )
    return SandboxResult(True, False, False, 0, "", "", time.perf_counter() - t0, None)


def _normalize_output(text: str) -> str:
    """Normalize trailing whitespace per line and trailing newlines for competitive programming I/O."""
    lines = [line.rstrip() for line in (text or "").strip().splitlines()]
    return "\n".join(lines)


def check_solution(
    candidate_text_or_code: str,
    *,
    test_code: str | None = None,
    entry_point: str | None = None,
    io_examples: list[dict[str, str]] | None = None,
    timeout_s: float = 4.0,
    max_memory_mb: int = 512,
    fn_name: str | None = None,
) -> SandboxResult:
    """Verify a candidate solution against either:
    - Unit test assertions (`test_code`, optionally calling `check(entry_point)`), or
    - Competitive programming input/output pairs (`io_examples = [{"input": ..., "output": ...}, ...]`).

    Com `io_examples`, todos os testes rodam num único subprocesso, param no primeiro que falha e
    seguem o LiveCodeBench oficial (ver `run_io_tests_in_sandbox`). `fn_name` ativa o modo funcional
    (LeetCode); `timeout_s` passa a ser o limite por teste e `max_memory_mb` o RLIMIT_AS do processo.
    """
    code = extract_python_code(candidate_text_or_code)
    if not code.strip():
        return SandboxResult(
            passed=False,
            timed_out=False,
            memory_exceeded=False,
            exit_code=1,
            stdout="",
            stderr="Empty solution code",
            elapsed_s=0.0,
            error_type="EmptyCode",
        )

    if io_examples:
        return run_io_tests_in_sandbox(
            code,
            [str(c.get("input", "")) for c in io_examples],
            [str(c.get("output", "")) for c in io_examples],
            fn_name=fn_name,
            timeout_s=timeout_s,
            max_memory_mb=max_memory_mb,
        )

    full_script = code
    if test_code:
        full_script = f"{code}\n\n{test_code}\n"
        if entry_point and f"check({entry_point})" not in test_code and "def check(" in test_code:
            full_script += f"\ncheck({entry_point})\n"

    return run_code_in_sandbox(
        full_script,
        timeout_s=timeout_s,
        max_memory_mb=max_memory_mb,
    )
