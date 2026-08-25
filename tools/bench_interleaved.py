#!/usr/bin/env python3
"""Interleaved paired benchmark: multiple rounds, per-fixture medians.
Env: BIN/LIB, ROUNDS (default 3), ONLY (csv filter), N_MAX base."""
import json, subprocess, sys, time, urllib.request, os, signal, statistics

PORT = 5931
BASE = f"http://127.0.0.1:{PORT}"
M = "/home/luke/models"
BIN = os.environ.get("BIN", "/home/luke/llm/cuda-dflash2-gemm/bin/llama-server")
LIB = os.environ.get("LIB", "/home/luke/llm/cuda-dflash2-gemm/lib")
ROUNDS = int(os.environ.get("ROUNDS", "3"))

CODE_MODULE = '''import hashlib\nclass Ledger:\n    def __init__(self):\n        self.entries = []\n    def add(self, account, amount):\n        self.entries.append((account, amount))\n    def balance(self, account):\n        return sum(a for acc, a in self.entries if acc == account)\n    def audit(self):\n        return hashlib.sha256(repr(self.entries).encode()).hexdigest()\ndef transfer(ledger, src, dst, amount):\n    assert ledger.balance(src) >= amount, "insufficient funds"\n    ledger.add(src, -amount)\n    ledger.add(dst, amount)\n    return ledger\n''' * 6

PROMPTS = [
    ("essay", "Write a detailed technical essay about the history of GPU computing, covering CUDA, tensor cores, and modern inference optimizations. Be thorough and specific."),
    ("code", "Write a complete Python implementation of a red-black tree with insert, delete, and validation methods. Include detailed comments explaining each rotation case."),
    ("math", "Solve step by step: A factory produces widgets where the cost function is C(x) = 0.01x^3 - 3x^2 + 500x + 2000 and revenue R(x) = 250x - 0.5x^2. Find the production level x that maximizes profit, show all calculus work, and compute maximum profit."),
    ("edit-code", f"Here is a Python module:\n\n{CODE_MODULE}\n\nRename every occurrence of 'entries' to 'records', rename 'audit' to 'checksum', and add a one-line docstring to every class and function. Output the complete updated module verbatim."),
    ("copy-table", "Reformat this data as a markdown table, preserving every row and value exactly:\n\n" + "\n".join(f"region-{i}, latency={7+i%13}ms, throughput={120+i*17} tok/s, errors={i%4}" for i in range(60)) + "\n\nOutput only the table."),
    ("doc-extend", "Summarize these release notes as a bulleted changelog, keeping every version number and date exactly:\n\n" + "\n".join(f"- v2.{i}.0 (2026-0{i%9+1}-1{i}): scheduler fix #{i}, KV cache optimization, new sampler preset {i}" for i in range(40))),
]

CONFIGS = [
    ("BASE-n8",   {}, {}),
    ("TAIL7",     {"SPEC_TAIL_N": "7"}, {"CFG_N_MAX": "14"}),
    ("GAP1",      {"SPEC_TRUNC_GAP": "1.0"}, {}),
    ("BOTH",      {"SPEC_TAIL_N": "7", "SPEC_TRUNC_GAP": "1.0"}, {"CFG_N_MAX": "14"}),
]
if os.environ.get("ONLY"):
    want = os.environ["ONLY"].split(",")
    CONFIGS = [c for c in CONFIGS if c[0] in want]

def wait_health(timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"{BASE}/health", timeout=2) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return time.time() - t0
        except Exception:
            time.sleep(1)
    raise RuntimeError("server not healthy")

def start(cenv):
    envd = dict(os.environ); envd["LD_LIBRARY_PATH"] = LIB; envd.update(cenv)
    n_max = cenv.get("CFG_N_MAX", "8")
    args = [BIN, "--host", "127.0.0.1", "--port", str(PORT),
            "-fa", "on", "-ngl", "999", "--no-mmap", "--jinja",
            "-ub", "1024", "-b", "1024", "-c", "131072",
            "-m", f"{M}/Qwen3.8-27B-GGUF-IQ4XS/Qwen3.8-27B-UD-IQ4_XS.gguf",
            "--mmproj", f"{M}/Qwen3.8-27B-GGUF/mmproj-F16.gguf",
            "--spec-type", "draft-dflash",
            "--model-draft", f"{M}/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf",
            "--spec-draft-n-max", n_max, "-bs", "--cache-reuse", "256",
            "--reasoning-effort", "medium"]
    logf = open(f"/home/luke/github/afap-qwen3.8/logs/il-{int(time.time()*1000)%10**9}.log", "w")
    proc = subprocess.Popen(args, stdout=logf, stderr=subprocess.STDOUT, preexec_fn=os.setsid, env=envd)
    return proc, logf

def stop(proc, logf):
    try: os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception: pass
    try: proc.wait(timeout=20)
    except Exception: os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    logf.close()

def chat(prompt, max_tokens=384):
    body = {"model": "x", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "stream": False, "temperature": 1.0,
            "top_k": 20, "top_p": 0.95, "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{BASE}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())

if __name__ == "__main__":
    acc = {c[0]: {f: [] for f, _ in PROMPTS} for c in CONFIGS}
    accn = {c[0]: {f: [] for f, _ in PROMPTS} for c in CONFIGS}
    for rnd in range(ROUNDS):
        for label, cenv, _n in CONFIGS:
            proc, logf = start(cenv)
            try:
                t = wait_health()
                print(f"[r{rnd} {label}] loaded {t:.0f}s", flush=True)
                time.sleep(1)
                for name, p in PROMPTS:
                    d = chat(p)
                    tm = d["timings"]
                    if d["choices"][0].get("finish_reason") == "length":
                        acc[label][name].append(tm["predicted_per_second"])
                        accn[label][name].append(
                            (tm.get("draft_n_accepted") or 0) / max(1, tm.get("draft_n") or 1))
            finally:
                stop(proc, logf)
    out = {}
    for label in acc:
        rows = {}
        means = []
        for name, _ in PROMPTS:
            v = acc[label][name]
            med = statistics.median(v) if v else float("nan")
            an = statistics.median(accn[label][name]) if accn[label][name] else float("nan")
            rows[name] = {"median_tps": round(med, 1), "runs": [round(x, 1) for x in v],
                          "median_accept": round(an, 3)}
        for name, _ in PROMPTS:
            means.append(rows[name]["median_tps"])
        rows["MEAN"] = round(sum(means) / max(1, len(means)), 1)
        out[label] = rows
        print(f"[{label}] MEAN(median-of-{ROUNDS})={rows['MEAN']}", flush=True)
        for name, _ in PROMPTS:
            print(f"    {name}: {rows[name]['median_tps']} t/s acc={rows[name]['median_accept']} runs={rows[name]['runs']}", flush=True)
    open("/home/luke/github/afap-qwen3.8/logs/interleaved-results.json", "w").write(json.dumps(out, indent=1))
    print("DONE")
