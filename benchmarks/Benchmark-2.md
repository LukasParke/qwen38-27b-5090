# Benchmark-2 — Speculative decoding optimization + kernel patch (2026-08-21)

Continues Benchmark-1. All runs: patched fork build 570 (`~/llm/cuda-dflash2-gemm`, adds
`GGML_CUDA_MMVQ_MAX_BATCH` runtime knob), ctx 8192, temp 1.0/top-k 20/top-p 0.95, thinking off,
3-prompt mean (essay/code/math), server-reported `predicted_per_second`.

## Kernel patch (the fork change)

`ggml/src/ggml-cuda/mmvq.cu`: MMVQ dispatched for quantized matmuls with ≤8 token rows; spec-decode
verify batches are ≤9 rows ⇒ weights re-streamed per drafted token. Added runtime env
`GGML_CUDA_MMVQ_MAX_BATCH` (default 8 = stock; `1` routes all multi-token batches to MMQ/GEMM,
which streams weights once per step). Rebuilt to separate prefix; production binary untouched.

| config | stock (MMVQ≤8) | GEMM forced (=1) |
|---|---|---|
| Q4_K_XL + DFlash2-Q8 n7 | 169.5 | 188.2 (+11%) |

## Draft mechanism comparison (all with GEMM=1)

| config | t/s | accept |
|---|---|---|
| nospec | 75.4 | — |
| draft-mtp n3 (built-in head) | 139.0 | 0.676 |
| draft-dspark n7 (local Q8) | 115.4 | 0.255 |
| draft-dflash DFlash2-Q4 n7 | 166.1→188.2 | 0.42 |
| draft-dflash DFlash2-Q8 n7 | **169.5→192.9 w/ f16 draft-KV** | 0.44–0.45 |
| draft-dflash DFlash2-BF16 n7 | 205.7 | 0.505 |

Findings: DSpark draft is rope-mismatched (trained w/ YaRN 32× @8k orig ctx) — dead end here.
Draft precision barely matters (BF16's higher acceptance eaten by 3.9 GB draft pass).
f16 draft KV > q8_0 draft KV. MTP head loses to DFlash2 (matches published accept 4.28 vs 4.80).

## Main-quant comparison (DFlash2-Q4 n7, GEMM=1, -bs backend sampling)

| main quant | size | t/s |
|---|---|---|
| UD-Q4_K_XL | 17.9 GB | 210.9 |
| **UD-IQ4_XS** | 14.3 GB | **210.9–211.4** (best; also highest acceptance 0.482) |
| NVFP4-MTP-VERY-LOW (3rd party) | 14.9 GB | 207.6, high per-prompt variance — rejected |

## Quality parity (25 verifiable questions, greedy, exact-match)

- IQ4_XS: **24/25 (96%)**
- Q4_K_XL: **24/25 (96%)** — identical; intelligence maintained.
- Speculative decoding itself is lossless by construction (rejection sampling preserves target
  distribution) — quant choice is the only quality lever.

## Context depth & KV precision (IQ4_XS + DFlash2-Q4)

| KV config | fresh | ~26–35k depth | verdict |
|---|---|---|---|
| f16 KV | 211 | 90 (story task) | fastest |
| q8_0 KV | — | 84.6 (same task) | ~6% slower at depth — rejected |
| q4_0 KV | — | (user data: −46%) | rejected |

Depth itself is NOT the killer — task type is (acceptance: technical 0.44–0.48, creative prose
0.20, Q&A short-form low). Spec decode is depth-stable: 102 t/s @165 tokens ≈ 103 t/s @22k depth
(same story prompt). No-spec decode is depth-flat 75–88.

## Max native context (no YaRN)

262144 ctx boots on IQ4_XS with q8_0 target KV + q4_0 draft KV: **27.4 GB VRAM** (~5 GB headroom),
decode 130 fresh / 109 @60k / 67 @83k depth, acceptance stable 0.32–0.38. Tradeoff: q8_0 KV costs
~6% at depth vs f16, but f16 KV @262k (16 GB) + draft KV does not fit 32 GB.
Two profiles delivered: **speed** (131072, f16 KV) and **max-context** (262144, q8_0 KV).

## Concurrency (IQ4_XS + DFlash2-Q4, GEMM=1, -bs)

| slots | aggregate t/s | per-stream |
|---|---|---|
| 1 | 211 | 211 |
| 2 (np4) | 195 | 117 |
| 4 (np4) | 285.6 | 98 |

Sublinear (draft pass batches per-slot blocks; CPU sampling). Backend sampling (`-bs`) required
for slot scaling. Aggregate rises with concurrency as required, but single-stream remains the
per-request figure.

## Cumulative progress

75.4 (stock nospec) → 169.5 (DFlash2) → 192.9 (GEMM patch + f16 draft KV) → **211.4 t/s**
single-stream (2.8×). Published SGLang 5090 cookbook for this model: 206.1 t/s — exceeded.

## Path to 500 (analysis)

Verify cost ≈ one weight pass (~12 ms @ IQ4_XS) ⇒ t/s ≈ 85 steps/s × accepted-tok/cycle.
500 t/s needs τ ≈ 6/cycle ⇒ acceptance ratio ≈ 0.85 at block 7 — no public drafter achieves this
on this target (best published mean accept: DFlash2 4.80 incl. bonus). Remaining lever: train a
custom larger-block diffusion drafter (DeepSpec / DSpark-lineage, block 16–32 + Markov + confidence
heads) — under investigation (DeepSpecScout). Physics: even Q1 quant (6.7 GB) at τ=3.3 caps ~370 t/s;
τ is the binding constraint, not weights.
