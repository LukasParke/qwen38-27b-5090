#!/usr/bin/env python3
"""Cascade/block-size benchmark. Env: N_MAX (draft size, default 7), ONLY (csv of config labels)."""
import json, subprocess, sys, time, urllib.request, os, signal

PORT = 5931
BASE = f"http://127.0.0.1:{PORT}"
M = "/home/luke/models"
N_MAX = os.environ.get("N_MAX", "7")
BIN = os.environ.get("BIN", "/home/luke/llm/cuda-dflash2-gemm/bin/llama-server")
LIB = os.environ.get("LIB", "/home/luke/llm/cuda-dflash2-gemm/lib")

CODE_MODULE = '''import hashlib\nclass Ledger:\n    def __init__(self):\n        self.entries = []\n    def add(self, account, amount):\n        self.entries.append((account, amount))\n    def balance(self, account):\n        return sum(a for acc, a in self.entries if acc == account)\n    def audit(self):\n        return hashlib.sha256(repr(self.entries).encode()).hexdigest()\ndef transfer(ledger, src, dst, amount):\n    assert ledger.balance(src) >= amount, "insufficient funds"\n    ledger.add(src, -amount)\n    ledger.add(dst, amount)\n    return ledger\n''' * 6

PROMPTS = [
    ("essay", "Write a detailed technical essay about the history of GPU computing, covering CUDA, tensor cores, and modern inference optimizations. Be thorough and specific."),
    ("code", "Write a complete Python implementation of a red-black tree with insert, delete, and validation methods. Include detailed comments explaining each rotation case."),
    ("math", "Solve step by step: A factory produces widgets where the cost function is C(x) = 0.01x^3 - 3x^2 + 500x + 2000 and revenue R(x) = 250x - 0.5x^2. Find the production level x that maximizes profit, show all calculus work, and compute maximum profit."),
    ("edit-code", f"Here is a Python module:\n\n{CODE_MODULE}\n\nRename every occurrence of 'entries' to 'records', rename 'audit' to 'checksum', and add a one-line docstring to every class and function. Output the complete updated module verbatim."),
    ("copy-table", "Reformat this data as a markdown table, preserving every row and value exactly:\n\n" + "\n".join(f"region-{i}, latency={7+i%13}ms, throughput={120+i*17} tok/s, errors={i%4}" for i in range(60)) + "\n\nOutput only the table."),
    ("doc-extend", "Summarize these release notes as a bulleted changelog, keeping every version number and date exactly:\n\n" + "\n".join(f"- v2.{i}.0 (2026-0{i%9+1}-1{i}): scheduler fix #{i}, KV cache optimization, new sampler preset {i}" for i in range(40))),
]

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

def start(llama, lib, env_extra, extra_args):
    envd = dict(os.environ); envd["LD_LIBRARY_PATH"] = lib
    if env_extra: envd.update(env_extra)
    args = [llama, "--host", "127.0.0.1", "--port", str(PORT),
            "-fa", "on", "-ngl", "999", "--no-mmap", "--jinja",
            "-ub", "1024", "-b", "1024", "-c", "131072",
            "-m", f"{M}/Qwen3.8-27B-GGUF-IQ4XS/Qwen3.8-27B-UD-IQ4_XS.gguf",
            "--mmproj", f"{M}/Qwen3.8-27B-GGUF/mmproj-F16.gguf",
            "--spec-type", "draft-dflash",
            "--model-draft", os.environ.get("DRAFT", f"{M}/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"),
            "--spec-draft-n-max", os.environ.get("CFG_N_MAX", N_MAX), "-bs", "--cache-reuse", "256",
            "--reasoning-effort", "medium"] + extra_args
    logf = open(f"/home/luke/github/afap-qwen3.8/logs/cascade-{int(time.time())}.log", "w")
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

CONFIGS = [
    (name, extra, BIN, LIB, cenv) for name, extra, cenv in [
        ("F0-parity",       [], {}),
        ("F1-tail7",        [], {"SPEC_TAIL_N": "7"}),
        ("F2-gap0.5",       [], {"SPEC_TRUNC_GAP": "0.5"}),
        ("F3-gap1.0",       [], {"SPEC_TRUNC_GAP": "1.0"}),
        ("F4-tail+gap0.5",  [], {"SPEC_TAIL_N": "7", "SPEC_TRUNC_GAP": "0.5"}),
    ]
]
CONFIGS = [(n, e, BIN, LIB, {**{"N_MAX_HINT": "1"}, **ce}) for n, e, ce in [(c[0], c[1], c[4]) for c in CONFIGS]]

if os.environ.get("ONLY"):
    _want = os.environ["ONLY"].split(",")
    CONFIGS = [c for c in CONFIGS if c[0] in _want]

if __name__ == "__main__":
    results = {}
    for label, extra, llama, lib, cenv in CONFIGS:
        proc, logf = start(llama, lib, {"GGML_CUDA_MMVQ_MAX_BATCH": os.environ.get("MMVQ", "1"),
                                            **({"GGML_CUDA_PDL": "1"} if os.environ.get("PDL") else {})}, extra)
        try:
            t = wait_health()
            print(f"[{label}] cfg={cenv} n_max={os.environ.get('CFG_N_MAX', N_MAX)} loaded in {t:.0f}s", flush=True)
            time.sleep(1)
            rows = []
            for name, p in PROMPTS:
                d = chat(p)
                tm = d["timings"]; u = d.get("usage", {})
                rows.append({"fixture": name,
                             "pps": round(tm["predicted_per_second"], 1),
                             "prompt_pps": round(tm["prompt_per_second"], 1),
                             "gen": u.get("completion_tokens"),
                             "draft_n": tm.get("draft_n"), "acc": tm.get("draft_n_accepted"),
                             "finish": d["choices"][0].get("finish_reason")})
                print(f"    {name}: {rows[-1]['pps']} t/s finish={rows[-1]['finish']}", flush=True)
            ok = [r for r in rows if r["finish"] == "length"]
            mean = sum(r["pps"] for r in ok) / max(1, len(ok))
            draft = sum(r["draft_n"] or 0 for r in rows); acc = sum(r["acc"] or 0 for r in rows)
            print(f"[{label}] MEAN={mean:.1f} accept={acc/max(1,draft):.3f}", flush=True)
            results[label] = {"mean": round(mean, 1), "accept": round(acc / max(1, draft), 3), "detail": rows}
        finally:
            stop(proc, logf)
    open(f"/home/luke/github/afap-qwen3.8/logs/cascade-results-{N_MAX}.json", "w").write(json.dumps(results, indent=1))
    print("DONE " + json.dumps({k: v["mean"] for k, v in results.items()}))
