#!/usr/bin/env python3
"""Scan captured records for the >=2-consecutive-supervised-token predicate (at train max_len)."""
import torch, glob, os, json, sys
from concurrent.futures import ProcessPoolExecutor

D = "/home/luke/github/afap-qwen3.8/draft-training/cache/hidden_states/qwen38-27b-dspark-b16"
MAXLEN = 1024

def ok(path):
    try:
        d = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        m = d["loss_mask"].view(-1)[:MAXLEN].bool()
        return bool((m[:-1] & m[1:]).any())
    except Exception as e:
        return f"ERR {e}"

if __name__ == "__main__":
    files = sorted(glob.glob(f"{D}/rows_*/*.ckpt"))
    print("total", len(files), flush=True)
    bad = []
    with ProcessPoolExecutor(max_workers=24) as ex:
        for i, (f, r) in enumerate(zip(files, ex.map(ok, files, chunksize=8))):
            if r is not True:
                bad.append((f, r))
            if i % 2000 == 0:
                print(i, "scanned", flush=True)
    print("bad:", len(bad))
    for f, r in bad[:10]:
        print(os.path.basename(f), r)
    json.dump([f for f, _ in bad], open("/home/luke/github/afap-qwen3.8/draft-training/bad_records.json", "w"))
