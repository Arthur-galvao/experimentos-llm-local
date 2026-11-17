"""Smoke longo: poucas sequências compridas para medir tok/s no regime da F1a com tempo limitado.

Reusa `cmd_smoke` de scripts/t4_validate.py sem alterá-lo: só troca, em tempo de execução,
quantas sequências e quantos tokens ele gera. Grava `smoke_int8.json` no mesmo lugar que o smoke
normal; o lab_bench.sh renomeia para `smoke_int8_long.json`.

  python lab/long_smoke.py --repo-root ~/dory-lab/repo --n-seqs 4 --max-tokens 4096 [--enforce-eager]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--n-seqs", type=int, default=4, help="múltiplo de 2 (o smoke usa 2 prompts)")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--enforce-eager", action="store_true")
    a = ap.parse_args()

    repo = a.repo_root.resolve()
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location("t4_validate", repo / "scripts" / "t4_validate.py")
    t4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t4)

    t4.SMOKE_N_SEQS = a.n_seqs  # lidos por cmd_smoke na hora da chamada
    t4.SMOKE_MAX_TOKENS = a.max_tokens
    t4.cmd_smoke(argparse.Namespace(variant="int8", n_problems=32, attn_backend=None, enforce_eager=a.enforce_eager))


if __name__ == "__main__":  # o vLLM usa spawn e reimporta este arquivo
    main()
