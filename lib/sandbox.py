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
import tracemalloc

MAX_MEM_BYTES = int(sys.argv[1])
TARGET_SCRIPT = sys.argv[2]

def _apply_memory_limit(limit_bytes: int) -> None:
    if limit_bytes <= 0:
        return
    # 1. POSIX RLIMIT_AS / RLIMIT_DATA (Linux / Colab)
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        new_hard = limit_bytes if hard == resource.RLIM_INFINITY else min(hard, limit_bytes)
        resource.setrlimit(resource.RLIMIT_AS, (new_hard, new_hard))
    except Exception:
        pass

    # 2. Cross-platform Python allocator guard via tracemalloc
    tracemalloc.start()

    def _mem_trace(frame, event, arg):
        if limit_bytes > 0:
            _, peak = tracemalloc.get_traced_memory()
            if peak > limit_bytes:
                raise MemoryError(f"Memory limit exceeded: peak {peak} bytes > {limit_bytes} bytes")
        return _mem_trace

    sys.settrace(_mem_trace)

_apply_memory_limit(MAX_MEM_BYTES)

with open(TARGET_SCRIPT, "rb") as _f:
    _code_bytes = _f.read()

_globals = {
    "__name__": "__main__",
    "__file__": TARGET_SCRIPT,
    "__builtins__": __builtins__,
}
try:
    exec(compile(_code_bytes, TARGET_SCRIPT, "exec"), _globals)
finally:
    if MAX_MEM_BYTES > 0 and tracemalloc.is_tracing():
        _, _peak = tracemalloc.get_traced_memory()
        if _peak > MAX_MEM_BYTES:
            raise MemoryError(f"Memory limit exceeded: peak {_peak} bytes > {MAX_MEM_BYTES} bytes")
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
) -> SandboxResult:
    """Verify a candidate solution against either:
    - Unit test assertions (`test_code`, optionally calling `check(entry_point)`), or
    - Competitive programming input/output pairs (`io_examples = [{"input": ..., "output": ...}, ...]`).
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
        total_elapsed = 0.0
        for idx, case in enumerate(io_examples):
            inp = str(case.get("input", ""))
            expected = _normalize_output(str(case.get("output", "")))
            res = run_code_in_sandbox(
                code,
                stdin_data=inp,
                timeout_s=timeout_s,
                max_memory_mb=max_memory_mb,
            )
            total_elapsed += res.elapsed_s
            if not res.passed:
                res.elapsed_s = total_elapsed
                return res
            actual = _normalize_output(res.stdout)
            if actual != expected:
                return SandboxResult(
                    passed=False,
                    timed_out=False,
                    memory_exceeded=False,
                    exit_code=1,
                    stdout=res.stdout,
                    stderr=f"Wrong Answer on test {idx}: expected {expected!r}, got {actual!r}",
                    elapsed_s=total_elapsed,
                    error_type="WrongAnswer",
                )
        return SandboxResult(
            passed=True,
            timed_out=False,
            memory_exceeded=False,
            exit_code=0,
            stdout="",
            stderr="",
            elapsed_s=total_elapsed,
            error_type=None,
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
