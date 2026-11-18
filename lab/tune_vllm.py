"""Lab perf tuning for Dory INT8 on vLLM: one config per run, writes ~/dory-lab/tune/<label>.json.

Measures only throughput and memory (no dataset generation). Same workload for every config so
results are comparable: N sequences x M tokens with ignore_eos, i.e. the long F1a regime.

  python lab/tune_vllm.py --label g_mml8704 --max-model-len 8704 [--eager] [--vram-gib 7.3]
         [--gpu-util 0.95] [--capture-sizes 1,2,4,8] [--max-num-seqs 16] [--cudagraph-mode PIECEWISE]
         [--n-seqs 8] [--max-tokens 4096]
"""
from __future__ import annotations
import argparse, importlib.util, json, logging, os, socket, subprocess, sys, threading, time, traceback
from pathlib import Path

def vram_sampler(stop, peak):
    while not stop.is_set():
        try:
            o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5).stdout.split(",")
            peak["mem"] = max(peak["mem"], int(o[0])); peak["util_sum"] += int(o[1]); peak["n"] += 1
        except Exception:
            pass
        stop.wait(1.0)

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True)
    ap.add_argument("--repo-root", type=Path, default=Path.home() / "dory-lab" / "repo")
    ap.add_argument("--out-dir", type=Path, default=Path.home() / "dory-lab" / "tune")
    ap.add_argument("--max-model-len", type=int, default=8704)
    ap.add_argument("--eager", action="store_true", help="enforce_eager (sem CUDA graphs)")
    ap.add_argument("--vram-gib", type=float, default=None, help="LAB_VRAM_GIB (vira gpu_memory_utilization, teto 0.95)")
    ap.add_argument("--gpu-util", type=float, default=None, help="gpu_memory_utilization direto (ignora o teto)")
    ap.add_argument("--capture-sizes", default=None, help="cudagraph_capture_sizes, ex.: 1,2,4,8")
    ap.add_argument("--cudagraph-mode", default=None, help="ex.: PIECEWISE, FULL_DECODE_ONLY, FULL_AND_PIECEWISE")
    ap.add_argument("--max-num-seqs", type=int, default=None)
    ap.add_argument("--max-num-batched-tokens", type=int, default=None)
    ap.add_argument("--kv-cache-dtype", default=None, help="ex.: fp8_e5m2, fp8_e4m3 (muda a numérica: precisa revalidar)")
    ap.add_argument("--attn-backend", default=None, help="ex.: TRITON_ATTN, FLEX_ATTENTION")
    ap.add_argument("--kv-mode", choices=["per_loop", "last_loop"], default="per_loop", help="recurrent_kv_cache_mode do Dory")
    ap.add_argument("--loops", type=int, default=2)
    ap.add_argument("--n-seqs", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4096)
    a = ap.parse_args()

    if a.vram_gib is not None:
        os.environ["LAB_VRAM_GIB"] = str(a.vram_gib)
    repo = a.repo_root.resolve()
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location("t4_validate", repo / "scripts" / "t4_validate.py")
    t4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t4)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = a.out_dir / f"{a.label}.json"
    rep = {"label": a.label, "host": socket.gethostname(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
           "n_seqs": a.n_seqs, "max_tokens": a.max_tokens, "ignore_eos": True,
           "config": {k: v for k, v in vars(a).items() if k not in ("repo_root", "out_dir", "label")}}
    stop, peak = threading.Event(), {"mem": 0, "util_sum": 0, "n": 0}
    threading.Thread(target=vram_sampler, args=(stop, peak), daemon=True).start()
    try:
        t4._require_dory_patch()
        from vllm import LLM, SamplingParams
        from lib.dory import DEFAULT_EQUIVALENCE_PROMPTS, DORY_HF_REPO, load_dory_tokenizer, official_sampling

        kw = t4._build_llm_kwargs("int8", max_model_len=a.max_model_len, n_loops=a.loops, enforce_eager=a.eager,
                                     attn_backend=a.attn_backend)
        kw["hf_overrides"] = {**kw["hf_overrides"], "recurrent_kv_cache_mode": a.kv_mode}
        if a.gpu_util is not None: kw["gpu_memory_utilization"] = a.gpu_util
        if a.max_num_seqs is not None: kw["max_num_seqs"] = a.max_num_seqs
        if a.max_num_batched_tokens is not None: kw["max_num_batched_tokens"] = a.max_num_batched_tokens
        if a.kv_cache_dtype: kw["kv_cache_dtype"] = a.kv_cache_dtype
        cc = {}
        if a.capture_sizes: cc["cudagraph_capture_sizes"] = [int(x) for x in a.capture_sizes.split(",")]
        if a.cudagraph_mode: cc["cudagraph_mode"] = a.cudagraph_mode
        if cc and not a.eager: kw["compilation_config"] = cc
        rep["llm_kwargs"] = kw
        print(f"[tune] {a.label}: {kw}", flush=True)
        rep["attention_backend_requested"] = a.attn_backend

        tok = load_dory_tokenizer(DORY_HF_REPO)
        prompts = t4._formatted_prompts(tok, list(DEFAULT_EQUIVALENCE_PROMPTS[:2]))
        s = official_sampling(DORY_HF_REPO)
        sp = SamplingParams(n=max(1, a.n_seqs // len(prompts)), temperature=float(s["temperature"]), top_p=float(s["top_p"]),
                            max_tokens=a.max_tokens, ignore_eos=True, seed=42)

        lg = logging.getLogger("vllm"); col = t4._LogCollector(); lg.addHandler(col)
        try:
            t0 = time.perf_counter(); llm = LLM(**kw); load_s = time.perf_counter() - t0
        finally:
            lg.removeHandler(col)
        kv = t4._extract_kv_capacity(llm, col.lines, a.max_model_len)
        rep["attention_backend"] = t4._detect_attn_backend(llm, a.attn_backend)
        idle_mem = peak["mem"]
        peak.update(util_sum=0, n=0)
        t1 = time.perf_counter(); outs = llm.generate(prompts, sp); gen_s = max(1e-3, time.perf_counter() - t1)
        n_tok = sum(len(c.token_ids) for r in outs for c in r.outputs)
        prompt_len = max(len(r.prompt_token_ids) for r in outs)
        kv_tokens = kv.get("kv_cache_size_tokens")
        rep.update(status="OK", load_s=round(load_s, 1), gen_s=round(gen_s, 1), tokens_generated=n_tok,
                   tok_s=round(n_tok / gen_s, 1), per_seq_tok_s=round(n_tok / gen_s / a.n_seqs, 2), **kv,
                   # quantas sequências do tamanho deste teste cabem juntas no KV
                   seqs_fit_kv=round(kv_tokens / (prompt_len + a.max_tokens), 2) if kv_tokens else None,
                   seqs_fit_f1a_8192=round(kv_tokens / (prompt_len + 8192), 2) if kv_tokens else None,
                   peak_vram_mib=peak["mem"], vram_after_load_mib=idle_mem,
                   gpu_util_mean=round(peak["util_sum"] / max(1, peak["n"]), 1))
    except BaseException as e:  # noqa: BLE001
        rep.update(status="FALHOU", error=f"{type(e).__name__}: {e}"[:600], traceback=traceback.format_exc()[-3000:])
    finally:
        stop.set()
        rep["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({k: rep.get(k) for k in ("label", "status", "tok_s", "kv_cache_size_tokens", "seqs_fit_f1a_8192", "peak_vram_mib", "error")}, ensure_ascii=False), flush=True)
        os._exit(0 if rep.get("status") == "OK" else 1)

if __name__ == "__main__":  # vLLM usa spawn e reimporta este arquivo
    main()
