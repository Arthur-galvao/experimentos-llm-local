"""F0 HF Validation Script: Loads Dory-4B in HuggingFace, validates dynamic loop switching
(`n_recurrent_loops` in {1, 2, 4}), and prints the Logit Lens per loop report.
Runs in an isolated process so 100% of GPU VRAM is freed upon termination.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import torch
    from lib.dory import DORY_HF_REPO, load_dory, logit_lens_per_loop, set_dory_loops
    from lib.gen import format_chat_prompt, generate
    from lib.io import DEFAULT_DRIVE_ROOT, ensure_model_files, setup_drive_and_cache

    setup_drive_and_cache(drive_root=DEFAULT_DRIVE_ROOT, local_fallback_root=repo_root)
    ensure_model_files(DORY_HF_REPO)

    dory_model, dory_tokenizer = load_dory(
        DORY_HF_REPO,
        n_loops=2,
        kv_cache_mode="per_loop",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    validation_prompt = format_chat_prompt(
        dory_tokenizer,
        "Write a Python function `fibonacci(n)` that returns the n-th Fibonacci number (0-indexed).",
        enable_thinking=True,
    )

    for k_loops in (1, 2, 4):
        set_dory_loops(dory_model, k_loops, kv_cache_mode="per_loop")
        out_k = generate(
            dory_model,
            [validation_prompt],
            max_new_tokens=256,
            temperature=0.0,
            n_loops=k_loops,
            tokenizer=dory_tokenizer,
            batch_size=1,
        )[0]
        print(
            f"[n_loops={k_loops}] latency={out_k.latency_s:.3f}s | "
            f"cot_tokens={out_k.n_cot_tokens} | total_tokens={out_k.n_total_tokens}"
        )
        print("  Trecho inicial:", repr(out_k.text[:140]))

    lens_report = logit_lens_per_loop(
        dory_model,
        dory_tokenizer,
        validation_prompt,
        n_loops=4,
        top_k=5,
        last_n_positions=5,
        apply_output_block=True,  # estados intermediários só fazem sentido após as Output layers
    )
    print("\n[Logit lens] Estado de cada loop passado pelas Output layers antes do lm_head.")
    for loop_info in lens_report:
        print(f"\n=== Loop Recorrente #{loop_info['loop_idx']} ===")
        for pos in loop_info["positions"]:
            top_pairs = [
                f"{repr(tok)} ({prob:.3f})"
                for tok, prob in zip(pos["top_tokens"], pos["top_probs"])
            ]
            print(
                f"  pos={pos['position_idx']:4d} | in={repr(pos['input_token']):12s} -> "
                + ", ".join(top_pairs)
            )


if __name__ == "__main__":
    main()
