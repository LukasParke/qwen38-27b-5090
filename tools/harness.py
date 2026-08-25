#!/usr/bin/env python3
"""llama-server benchmark harness: start server with args, run prompts, report timings."""
import json, subprocess, sys, time, urllib.request, os, signal

PORT = 5931
BASE = f"http://127.0.0.1:{PORT}"
LLAMA = "/home/luke/llm/cuda-dflash2-gemm/bin/llama-server"
LIB = "/home/luke/llm/cuda-dflash2-gemm/lib"
M = "/home/luke/models"

PROMPTS = [
    ("essay", "Write a detailed technical essay about the history of GPU computing, covering CUDA, tensor cores, and modern inference optimizations. Be thorough and specific."),
    ("code", "Write a complete Python implementation of a red-black tree with insert, delete, and validation methods. Include detailed comments explaining each rotation case."),
    ("math", "Solve step by step: A factory produces widgets where the cost function is C(x) = 0.01x^3 - 3x^2 + 500x + 2000 and revenue R(x) = 250x - 0.5x^2. Find the production level x that maximizes profit, show all calculus work, and compute maximum profit."),
]

def wait_health(timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"{BASE}/health", timeout=2) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return time.time() - t0
        except Exception:
            time.sleep(1)
    raise RuntimeError("server did not become healthy")

def start(extra_args, ctx=8192, port=PORT, model=None, lib=None, env_extra=None, llama=None):
    envd = dict(os.environ)
    envd["LD_LIBRARY_PATH"] = lib or LIB
    if env_extra:
        envd.update(env_extra)
    args = [
        llama or LLAMA,
        "--host", "127.0.0.1", "--port", str(port),
        "-fa", "on", "-ngl", "999", "--no-mmap", "--jinja",
        "-ub", "1024", "-b", "1024", "--parallel", "1",
        "-m", model or f"{M}/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf",
        "-c", str(ctx),
    ] + extra_args
    logf = open(f"/home/luke/github/afap-qwen3.8/logs/server-{int(time.time())}.log", "w")
    proc = subprocess.Popen(args, stdout=logf, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid, env=envd)
    return proc, logf

def stop(proc, logf):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        pass
    try:
        proc.wait(timeout=15)
    except Exception:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    logf.close()

def chat(prompt, max_tokens=384, temperature=1.0, top_k=20, top_p=0.95, extra_body=None, timeout=300):
    body = {"model": "x", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "stream": False,
            "temperature": temperature, "top_k": top_k, "top_p": top_p}
    if extra_body:
        body.update(extra_body)
    req = urllib.request.Request(f"{BASE}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def bench(label, rounds=2, max_tokens=384, no_think=True, extra_body=None):
    results = []
    eb = dict(extra_body or {})
    if no_think:
        eb.setdefault("chat_template_kwargs", {}).setdefault("enable_thinking", False)
    for name, p in PROMPTS:
        d = chat(p, max_tokens=max_tokens, extra_body=eb)
        t = d["timings"]
        u = d.get("usage", {})
        results.append({"prompt": name, "pps": t["predicted_per_second"],
                        "prompt_pps": t["prompt_per_second"],
                        "gen": u.get("completion_tokens"),
                        "draft_n": t.get("draft_n"), "acc": t.get("draft_n_accepted"),
                        "finish": d["choices"][0].get("finish_reason")})
    gen_ok = [r for r in results if r["finish"] == "length"]
    mean_pps = sum(r["pps"] for r in gen_ok) / max(1, len(gen_ok))
    draft = sum(r["draft_n"] or 0 for r in results)
    acc = sum(r["acc"] or 0 for r in results)
    toks = sum(r["gen"] or 0 for r in results)
    print(f"[{label}] mean_pps={mean_pps:.1f} accept={acc/max(1,draft):.3f}")
    for r in results:
        print(f"    {r['prompt']}: {r['pps']:.1f} t/s finish={r['finish']} gen={r['gen']}")
    return {"label": label, "mean_pps": mean_pps, "accept_ratio": acc / max(1, draft),
            "detail": results}

if __name__ == "__main__":
    cfg = json.loads(sys.argv[1])
    proc, logf = start(cfg["args"], ctx=cfg.get("ctx", 8192),
                       model=cfg.get("model"), lib=cfg.get("lib"),
                       env_extra=cfg.get("env"), llama=cfg.get("llama"))
    try:
        t = wait_health()
        print(f"loaded in {t:.0f}s")
        time.sleep(2)
        r = bench(cfg["label"], rounds=cfg.get("rounds", 1),
                  max_tokens=cfg.get("max_tokens", 384),
                  no_think=cfg.get("no_think", True),
                  extra_body=cfg.get("extra_body"))
        print("RESULT_JSON=" + json.dumps(r))
    finally:
        stop(proc, logf)
