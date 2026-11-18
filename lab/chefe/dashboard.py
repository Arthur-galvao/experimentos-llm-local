"""Dory lab dashboard (runs on the chefe PC). Polls each lab PC over SSH and serves a live page."""
import json, os, subprocess, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
HOSTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hosts.json")
def load_hosts():
    with open(HOSTS_FILE, encoding="utf-8") as f:
        return json.load(f)
HOSTS = load_hosts()
KEY = os.path.expanduser(r"~\.ssh\id_ed25519_lab")
SSH = r"C:\Windows\System32\OpenSSH\ssh.exe"
PORT = int(os.environ.get("DASH_PORT", "8765"))
STATE = {}
HIST = {}
STARTED = set()
def poll(name, ip):
    while True:
        try:
            p = subprocess.run([SSH, "-n", "-i", KEY, "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "ceitelabinfo@" + ip,
                                "python3 ~/dory-lab/lab_status.py"], capture_output=True, timeout=25, stdin=subprocess.DEVNULL,
                               creationflags=0x08000000)  # CREATE_NO_WINDOW: no console flash per poll
            d = json.loads(p.stdout.decode("utf-8", "replace"))
            d["ok"] = True; d["ip"] = ip; STATE[name] = d
            g = d.get("gpu") or {}
            HIST[name].append([d["ts"], g.get("util"), g.get("mem_used"), d.get("live_out_tok_s")])
            del HIST[name][:-600]
        except Exception as e:
            STATE[name] = {**STATE[name], "ok": False, "err": str(e)[:200]}
        time.sleep(4)
HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
def PAGE():
    with open(HTML_PATH, encoding="utf-8") as f:
        return f.read()
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path.startswith("/api"):
            body = json.dumps({"state": dict(sorted(STATE.items())), "hist": HIST}).encode(); ct = "application/json"
        else:
            body = PAGE().encode("utf-8"); ct = "text/html; charset=utf-8"
        self.send_response(200); self.send_header("Content-Type", ct); self.send_header("Cache-Control", "no-store")
        self.end_headers(); self.wfile.write(body)
def watch_hosts():
    # picks up PCs added to hosts.json without restarting the dashboard
    while True:
        try:
            for n, ip in load_hosts().items():
                if n not in STARTED:
                    STARTED.add(n); STATE[n] = {"ok": False, "err": "aguardando"}; HIST[n] = []
                    threading.Thread(target=poll, args=(n, ip), daemon=True).start()
        except Exception:
            pass
        time.sleep(10)
threading.Thread(target=watch_hosts, daemon=True).start()
ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
