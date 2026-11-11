"""F0 Preparation Script: Hardware summary, local HF cache (`/root/.cache/huggingface`),
`ensure_model_files`, tokenizer check, and fixed 200-problem `data/lcb_quick.json` split.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from lib.dory import DORY_HF_REPO, verify_tokenizer_equivalence
    from lib.evals import build_lcb_quick_split
    from lib.hw import get_hw_config, print_hardware_summary
    from lib.io import DEFAULT_DRIVE_ROOT, ensure_model_files, setup_drive_and_cache

    print_hardware_summary()
    eval_cfg = get_hw_config("eval")
    print("Configuração de avaliação para o perfil ativo:", eval_cfg)

    paths = setup_drive_and_cache(drive_root=DEFAULT_DRIVE_ROOT, local_fallback_root=repo_root)
    for k, v in paths.items():
        print(f"{k:12s}: {v}")

    print(f"\nValidando arquivos JSON do modelo ({DORY_HF_REPO})...")
    verified = ensure_model_files(DORY_HF_REPO)
    for fname, p in verified.items():
        print(f"  [OK] {fname}: {p}")

    tok_report = verify_tokenizer_equivalence(DORY_HF_REPO)
    print(
        f"[Tokenizer] {tok_report['n_texts']} textos | "
        f"matched={tok_report['matched']} | fix_mistral_regex={tok_report['fix_mistral_regex']}"
    )

    lcb_quick_path = paths["data"] / "lcb_quick.json"
    items = build_lcb_quick_split(output_path=lcb_quick_path, n_problems=200)
    local_lcb_path = repo_root / "data" / "lcb_quick.json"
    if local_lcb_path.resolve() != lcb_quick_path.resolve():
        local_lcb_path.parent.mkdir(parents=True, exist_ok=True)
        local_lcb_path.write_text(lcb_quick_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"Split LCB Quick pronto: {len(items)} problemas.")
    if items:
        print("Mais recente:", items[0].get("question_id"), items[0].get("contest_date"))
        print("200º mais recente:", items[-1].get("question_id"), items[-1].get("contest_date"))


if __name__ == "__main__":
    main()
