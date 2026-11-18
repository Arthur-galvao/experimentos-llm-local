#!/usr/bin/env python3
"""Lab status collector: prints one JSON snapshot (default) or a live terminal view (--tui)."""
import glob, json, os, re, socket, subprocess, sys, time
H = os.path.expanduser("~/dory-lab")
def sh(c):
    try: return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=8).stdout
    except Exception: return ""
def gpu():
    q = sh("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits").strip().split(",")
    try: return dict(util=int(q[0]), mem_used=int(q[1]), mem_total=int(q[2]), temp=int(q[3]), power=float(q[4]))
    except Exception: return {}
def tail(path, n=6000):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2); s = f.tell(); f.seek(max(0, s - n)); return f.read().decode("utf-8", "replace")
    except Exception: return ""
def snapshot():
    log = max(glob.glob(H + "/*_run.log"), key=os.path.getmtime, default="")
    t = tail(log, 200000) if log else ""
    steps = re.findall(r"^=== \[([^\]]+)\] (.*?) ===", t, re.M)
    lines = [l for l in t.replace("\r", "\n").split("\n") if l.strip()]
    prog = next((l for l in reversed(lines) if "Processed prompts" in l or "%|" in l), "")
    m = re.search(r"output: ([\d.]+) toks/s", prog)
    results = {}
    for f in glob.glob(H + "/drive/results/lab_validation/*/*.json") + glob.glob(H + "/tune/*.json"):
        try:
            d = json.load(open(f)); k = os.path.basename(f)[:-5]
            results[k] = {x: d.get(x) for x in ("tok_s", "kv_cache_size_tokens", "max_concurrency", "load_s", "n_seqs", "max_tokens", "tokens_generated", "gen_s", "label", "status", "error", "seqs_fit_f1a_8192", "peak_vram_mib", "gpu_util_mean", "per_seq_tok_s") if x in d}
        except Exception: pass
    procs = sh("pgrep -fa 'lab_bench|lab_setup|long_smoke|t4_validate|tune_vllm' | grep -v pgrep").strip().splitlines()
    return dict(host=socket.gethostname(), ts=time.time(), gpu=gpu(), log=os.path.basename(log),
                log_age_s=round(time.time() - os.path.getmtime(log)) if log else None,
                step=" ".join(steps[-1]) if steps else "", progress=prog[-220:],
                live_out_tok_s=float(m.group(1)) if m else None, running=bool(procs), procs=[p[:160] for p in procs][:4],
                results=results, last_lines=lines[-6:])
if "--tui" in sys.argv:
    while True:
        s = snapshot(); g = s["gpu"]
        os.system("clear")
        bar = lambda v, mx, w=40: "#" * int(w * v / max(mx, 1)) + "." * (w - int(w * v / max(mx, 1)))
        st = "\033[32mRODANDO\033[0m" if s["running"] else "\033[33mOCIOSO\033[0m"
        print("\033[1m DORY LAB  %s  %s\033[0m   %s\n" % (s["host"], time.strftime("%H:%M:%S"), st))
        if g:
            print(f" GPU  {g['util']:3d}%  [{bar(g['util'],100)}]")
            print(f" VRAM {g['mem_used']:5d}/{g['mem_total']} MiB [{bar(g['mem_used'],g['mem_total'])}]")
            print(f" {g['temp']} C   {g['power']:.0f} W\n")
        print(f" Etapa: {s['step']}")
        if s["live_out_tok_s"] is not None: print(f" Agora: {s['live_out_tok_s']} tok/s (saida)")
        print(f" {s['progress'][-110:]}\n")
        print(" Resultados:")
        for k, v in sorted(s["results"].items()):
            if "tok_s" in v: print(f"  {k:28s} {v.get('tok_s')!s:>7} tok/s   KV {v.get('kv_cache_size_tokens')!s:>6}   conc {v.get('max_concurrency')}")
        time.sleep(3)
else:
    print(json.dumps(snapshot()))
