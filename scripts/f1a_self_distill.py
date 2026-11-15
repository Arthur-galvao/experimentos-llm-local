"""F1a: autodestilação em cascata do Dory L2. A lógica e as opções estão em `lib/f1a.py`.

  python scripts/f1a_self_distill.py --stage nothink|efficient|full [--n-problems 500]
  python scripts/f1a_self_distill.py --prepare-only | --build-only
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from lib.f1a import main as f1a_main

    f1a_main()


if __name__ == "__main__":
    main()
