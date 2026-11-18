"""Provision new lab PCs from the chefe (lab/chefe/): kit + artifacts (pulled over LAN from a seed PC) + setup + status window.

  python provision.py 9.05=<ip> 9.07=<ip>

Each new PC needs: Ubuntu booted, user ceitelabinfo logged in, sshd running, and the chefe public key
in ~/.ssh/authorized_keys. Artifacts are pulled from the seed PCs' HTTP share (not from Google Drive).
Progress goes to provision_<name>.log next to this file; on success the PC is added to hosts.json.
"""
import concurrent.futures as cf, json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
KEY = os.path.expanduser(r"~\.ssh\id_ed25519_lab")
SSH = r"C:\Windows\System32\OpenSSH\ssh.exe"; SCP = r"C:\Windows\System32\OpenSSH\scp.exe"
REPO_LOCAL = os.path.abspath(os.path.join(HERE, "..", ".."))  # repo root (this file lives in lab/chefe/)
# seeds = PCs already set up (they serve ~/dory-lab/src over HTTP); first 3 entries of hosts.json
SEEDS = list(json.load(open(os.path.join(HERE, "hosts.json"), encoding="utf-8")).values())[:3]
PORT = 8800
NOWIN = 0x08000000

def ssh(ip, cmd, t=120):
    p = subprocess.run([SSH, "-n", "-i", KEY, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=8",
                        "ceitelabinfo@" + ip, cmd], capture_output=True, timeout=t, stdin=subprocess.DEVNULL, creationflags=NOWIN)
    return p.returncode, (p.stdout + p.stderr).decode("utf-8", "replace")

def scp(ip, local, remote):
    p = subprocess.run([SCP, "-i", KEY, "-o", "BatchMode=yes", local, f"ceitelabinfo@{ip}:{remote}"], capture_output=True,
                       timeout=120, stdin=subprocess.DEVNULL, creationflags=NOWIN)
    if p.returncode: raise RuntimeError(p.stderr.decode("utf-8", "replace"))

def ensure_seed(seed):
    # idempotent: start a read-only HTTP share of ~/dory-lab/src on the seed if not running.
    # Fire-and-forget Popen: Windows OpenSSH keeps the session open while a backgrounded child lives.
    rc, o = ssh(seed, f"ss -ltn | grep -q ':{PORT} ' && echo UP", 20)
    if "UP" in o: return
    subprocess.Popen([SSH, "-n", "-i", KEY, "-o", "BatchMode=yes", "ceitelabinfo@" + seed,
                      f"cd ~/dory-lab/src && setsid nohup python3 -m http.server {PORT} > ~/dory-lab/seed_http.log 2>&1 < /dev/null &"],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=NOWIN)
    time.sleep(4)

def provision(name, ip, seed):
    log = open(os.path.join(HERE, f"provision_{name}.log"), "a", encoding="utf-8")
    def say(m):
        line = f"[{time.strftime('%H:%M:%S')}] {name} {m}"; print(line, flush=True); log.write(line + "\n"); log.flush()
    rc, o = ssh(ip, "hostname; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader", 30)
    if rc: say("SEM ACESSO SSH: " + o.strip()[:200]); return False
    say("conectado: " + o.strip().replace("\n", " | "))
    rc, o = ssh(ip, "mkdir -p ~/dory-lab/src ~/dory-lab/tune && cd ~/dory-lab && rm -rf repo && "
                    "curl -sL https://codeload.github.com/Arthur-galvao/experimentos-llm-local/tar.gz/refs/heads/main | tar xz && "
                    "mv experimentos-llm-local-main repo && echo kit-ok", 180)
    if "kit-ok" not in o: say("falha ao baixar kit: " + o[-300:]); return False
    for f, dst in [(os.path.join(REPO_LOCAL, "lab", "tune_vllm.py"), "dory-lab/repo/lab/tune_vllm.py"),
                   (os.path.join(REPO_LOCAL, "lab", "tune_queue.sh"), "dory-lab/repo/lab/tune_queue.sh"),
                   (os.path.join(HERE, "lab_status.py"), "dory-lab/lab_status.py"),
                   (os.path.join(HERE, "show_status.sh"), "dory-lab/show_status.sh")]:
        scp(ip, f, dst)
    rc, o = ssh(ip, "cd ~/dory-lab/repo && bash lab/lab_check.sh 2>&1 | grep -E 'BLOQUEIO|Resumo'", 300)
    say("lab_check: " + o.strip().replace("\n", " | ")[:300])
    if "BLOQUEIO]" in o: say("PARADO: lab_check acusou bloqueio"); return False
    ensure_seed(seed)
    say(f"baixando artefatos do seed {seed} (~8,5 GiB pela LAN)...")
    rc, o = ssh(ip, f"cd ~/dory-lab/src && wget -q -r -np -nH -e robots=off -R 'index.html*' http://{seed}:{PORT}/coder-comprimido/ ; "
                    "find coder-comprimido -type f | wc -l; du -sh coder-comprimido", 3600)
    say("artefatos: " + o.strip().replace("\n", " | "))
    say("rodando lab_setup.sh (~5 min)...")
    rc, o = ssh(ip, "cd ~/dory-lab/repo && bash lab/lab_setup.sh ~/dory-lab/src/coder-comprimido > ~/dory-lab/setup_run.log 2>&1; "
                    "tail -n 4 ~/dory-lab/setup_run.log; test -f ~/dory-lab/setup_ok.json && echo SETUP_OK", 3600)
    if "SETUP_OK" not in o: say("SETUP FALHOU: " + o[-500:]); return False
    say("setup OK")
    rc, o = ssh(ip, "bash ~/dory-lab/show_status.sh", 30); say("janela de status: " + o.strip())
    hf = os.path.join(HERE, "hosts.json")
    hosts = json.load(open(hf, encoding="utf-8")); hosts[name] = ip
    json.dump(dict(sorted(hosts.items())), open(hf, "w", encoding="utf-8"), indent=2)
    say("adicionado ao dashboard (hosts.json)")
    return True

if __name__ == "__main__":
    targets = [a.split("=", 1) for a in sys.argv[1:]]
    if not targets: sys.exit(__doc__)
    with cf.ThreadPoolExecutor(len(targets)) as ex:
        res = list(ex.map(lambda t: provision(t[1][0], t[1][1], SEEDS[t[0] % len(SEEDS)]), enumerate(targets)))
    print("RESUMO:", dict(zip([t[0] for t in targets], res)))
