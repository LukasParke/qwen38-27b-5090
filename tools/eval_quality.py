#!/usr/bin/env python3
"""Quant-quality parity eval: 25 verifiable questions, greedy decode, exact-match scoring."""
import json, sys, urllib.request

PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 5931
BASE = f"http://127.0.0.1:{PORT}"

QS = [
    ("What is 17 * 23 - 45? Answer with just the number.", "346"),
    ("A train travels 120 km in 90 minutes. What is its average speed in km/h? Answer with just the number.", "80"),
    ("If x + 7 = 19, what is x^2? Answer with just the number.", "144"),
    ("What is the capital of Australia? Answer with just the city name.", "Canberra"),
    ("How many continents are there? Answer with just the number.", "7"),
    ("What is the square root of 1764? Answer with just the number.", "42"),
    ("Simplify: (3^4)*(3^2)/3^5. Answer with just the number.", "27"),
    ("What year did the Berlin Wall fall? Answer with just the year.", "1989"),
    ("A shirt costs $40 and is discounted 25%. What is the final price in dollars? Answer with just the number.", "30"),
    ("What is 15% of 240? Answer with just the number.", "36"),
    ("Write the next prime after 97. Answer with just the number.", "101"),
    ("How many bits are in a byte? Answer with just the number.", "8"),
    ("What is the chemical symbol for gold? Answer with just the symbol.", "Au"),
    ("If a rectangle has area 84 and one side 7, what is the other side? Answer with just the number.", "12"),
    ("What is 2^10? Answer with just the number.", "1024"),
    ("Name the largest planet in our solar system. Answer with just the name.", "Jupiter"),
    ("What is the sum of interior angles of a hexagon in degrees? Answer with just the number.", "720"),
    ("Convert 100 degrees Fahrenheit to Celsius rounded to nearest integer. Answer with just the number.", "38"),
    ("How many players are on a standard soccer team on the field? Answer with just the number.", "11"),
    ("What is gcd(48, 180)? Answer with just the number.", "12"),
    ("A car depreciates 20% per year. After 2 years what fraction of original value remains, as percent? Answer with just the number.", "64"),
    ("What is the derivative of x^3 evaluated at x=2? Answer with just the number.", "12"),
    ("How many strings does a standard guitar have? Answer with just the number.", "6"),
    ("What is log2(8192)? Answer with just the number.", "13"),
    ("In binary, what is decimal 45? Answer with just the binary digits.", "101101"),
]

def ask(q):
    body = {"model": "x", "messages": [{"role": "user", "content": q}],
            "max_tokens": 512, "stream": False, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{BASE}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())

def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "eval"
    correct = 0
    details = []
    tps_all = []
    for q, want in QS:
        try:
            d = ask(q)
        except Exception as e:
            print(f"  ERROR {e}")
            continue
        txt = d["choices"][0]["message"]["content"].strip()
        got = txt.split()[0].rstrip(".,;:") if txt else ""
        ok = want.lower() == got.lower()
        # fallback: substring match
        if not ok:
            ok = want.lower() in txt.lower()
        correct += ok
        tps_all.append(d["timings"]["predicted_per_second"])
        details.append({"q": q[:60], "want": want, "got": got, "ok": ok})
        print(f"  {'OK ' if ok else 'MISS'} want={want:>8} got={got[:24]:<24} | {q[:50]}")
    n = len(QS)
    print(f"[{label}] {correct}/{n} = {100*correct/n:.0f}%  mean_tps={sum(tps_all)/max(1,len(tps_all)):.1f}")
    json.dump({"label": label, "score": correct, "n": n, "details": details},
              open(f"/home/luke/github/afap-qwen3.8/logs/eval-{label}.json", "w"), indent=1)

if __name__ == "__main__":
    main()
