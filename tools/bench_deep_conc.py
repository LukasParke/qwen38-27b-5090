#!/usr/bin/env python3
"""Deep-context and concurrency benchmarks."""
import json, sys, time, urllib.request, threading

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 5931
BASE = f"http://127.0.0.1:{PORT}"

def chat(prompt, max_tokens=256, tag=""):
    body = {"model": "x", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "stream": False, "temperature": 1.0,
            "top_k": 20, "top_p": 0.95,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{BASE}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1200) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    t = d["timings"]
    print(f"  {tag}: prompt={t['prompt_n']} decode={t['predicted_n']} "
          f"pps={t['predicted_per_second']:.1f} ttft={t['prompt_ms']/1000:.2f}s wall={dt:.1f}s")
    return t

def deep_ctx_test(fill=30000):
    filler = ("The quick brown fox jumps over the lazy dog near the riverbank where "
              "willows sway gently in the afternoon breeze. ") * (fill // 20)
    prompt = f"Here is an archive of notes:\n\n{filler}\n\nIgnore the notes above. Write a 200-word story about a lighthouse keeper."
    print(f"[deep-ctx ~{len(filler.split())} words]")
    chat(prompt, max_tokens=256, tag="deep")

def conc_test(n=4):
    prompts = [
        "Write a detailed essay about the history of computing.",
        "Explain quantum entanglement in depth with examples.",
        "Write a Python web server with threading from scratch, well commented.",
        "Describe the process of making sourdough bread step by step in detail.",
    ]
    results = []
    lock = threading.Lock()
    def run(i):
        t = chat(prompts[i % len(prompts)], max_tokens=384, tag=f"c{i}")
        with lock:
            results.append(t)
    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    t0 = time.time()
    for th in threads: th.start()
    for th in threads: th.join()
    wall = time.time() - t0
    tot = sum(t["predicted_n"] for t in results)
    print(f"[conc-{n}] aggregate={tot/wall:.1f} t/s over {wall:.1f}s, "
          f"per-stream avg={sum(t['predicted_per_second'] for t in results)/n:.1f}")

if __name__ == "__main__":
    mode = sys.argv[2] if len(sys.argv) > 2 else "all"
    if mode in ("all", "deep"): deep_ctx_test()
    if mode in ("all", "conc"):
        for n in (2, 4):
            conc_test(n)
