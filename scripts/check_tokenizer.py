"""Compare HF tokenizer vs vLLM tokenizer on 20 diverse texts (code, English, spaces, newlines)
and apply `fix_mistral_regex=True` if needed so both produce identical token IDs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from lib.dory import DORY_HF_REPO, TOKENIZER_TEST_TEXTS, verify_tokenizer_equivalence
    from lib.io import DEFAULT_DRIVE_ROOT, ensure_model_files, setup_drive_and_cache

    parser = argparse.ArgumentParser(description="Verifica equivalência de tokenização HF vs vLLM em 20 textos.")
    parser.add_argument("--model", default=DORY_HF_REPO, help="Repositório ou caminho do modelo.")
    args = parser.parse_args()

    setup_drive_and_cache(drive_root=DEFAULT_DRIVE_ROOT, local_fallback_root=repo_root)
    ensure_model_files(args.model)

    report = verify_tokenizer_equivalence(args.model, texts=TOKENIZER_TEST_TEXTS)
    print(
        f"[Tokenizer Check] model={args.model} | textos={report['n_texts']} | "
        f"divergências iniciais={report['divergences_initial']} | "
        f"divergências finais={report['divergences_final']} | "
        f"fix_mistral_regex={report['fix_mistral_regex']} | "
        f"matched={report['matched']}"
    )
    if not report["matched"]:
        for d in report["details"][:5]:
            print(f"  [texto #{d['text_idx']}] HF={d['hf_ids'][:12]} vs vLLM={d['vllm_ids'][:12]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
