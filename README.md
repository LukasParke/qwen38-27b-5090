# Qwen3.8-27B on one RTX 5090 — 211–293 tok/s, lossless, fully documented

Serving configuration, benchmarks, and tooling for running **Qwen3.8-27B** on a single
RTX 5090 with speculative decoding: **211–293 tok/s single-stream** (workload-dependent),
**285 tok/s aggregate @ 4 concurrent streams**, lossless (rejection-sampled), quality-verified,
131k native context — via a patched llama.cpp fork with DFlash2 block-diffusion drafting.

Everything here was measured, not estimated. Full evidence chain in [`benchmarks/`](benchmarks/).

## Headline results (RTX 5090, greedy + T=1.0 verified)

| Workload | tok/s | notes |
|---|---:|---|
| math / structured reasoning | **293–320** | acceptance 0.79–0.84 |
| code generation | **212–258** | |
| verbatim document reproduction (25k ctx) | **up to ~380** | context-fill verify blocks |
| prose / open-ended | 114–127 | acceptance-bound (drafter oracle) |
| aggregate @4 streams | **285** | `--parallel 4` profile |

Stack: llama.cpp fork (`5ecbe1ac1`, PR #27342 lineage) + IQ4_XS quant + **DFlash2 block-diffusion
drafting** (`--spec-type draft-dflash`, n7) + f16 KV @131k + backend sampling + an MMVQ→MMQ
dispatch patch (+11%). All speculation is lossless by construction; IQ4_XS vs Q4_K_XL parity
verified 24/25 on a greedy eval.

## Repo layout

| path | content |
|---|---|
| [`benchmarks/Benchmark-1..7.md`](benchmarks/) | complete campaign log: baselines → spec-decoding matrix → production cutover → drafter training attempts → external-repo analyses → final verdicts |
| [`tools/`](tools/) | benchmark harnesses (incl. interleaved median-of-N runner), dual-path verification prototype, quality eval |
| [`patches/mmvq-max-batch-rebased.patch`](patches/) | the MMVQ→MMQ verify-dispatch knob (+11%), rebased over upstream #26079 per-arch tables |
| [`config/llama-swap.example.yaml`](config/) | production llama-swap profiles (API key redacted): base / x4 concurrency / maxctx / experimental fill + nvfp4-ingest |
| [`scripts/`](scripts/) | reproducible build scripts for all three binary prefixes |
| [`draft-training/`](draft-training/) | SpecForge drafter-training pipeline scripts (capture → train → probe → export) |

## Key findings (why this config wins)

1. **DFlash2 > MTP > DSpark** for this target (acceptance 4.80 vs 4.28 vs 3.62 tokens/cycle).
2. **Verify width has an optimum at 8 rows**: fewer rows lose τ; more rows cost more latency than
   the extra accepted tokens return (measured ±: truncation −56%…−21%, context-fill tails −21%).
3. **Forcing MMQ for verify batches** (instead of per-row MMVQ re-streaming weights) is worth
   +11% — see patch; runtime-tunable via `GGML_CUDA_MMVQ_MAX_BATCH`.
4. f16 KV beats q8_0/q4_0 at depth despite memory cost; backend sampling worth +3%.
5. The drafter artifact's own oracle (top-16 recall = 6.79 tokens) caps this setup near
   ~380–420 tok/s — beyond that requires a stronger drafter artifact, not configuration.

## What did NOT work (measured, so you don't have to)

- ngram→DFlash2 cascades (both orderings + session cache): −8…−64%
- confidence-gap block truncation: −56%
- Q8_0 drafter (heavier stream, no acceptance gain), PDL launch overlap, MMVQ threshold sweeps
- upstream master as a base (drops DFlash2 selector support as of 2026-08-25; −4.2% even when restored)
- dual-sequence branch verification: blocked by llama.cpp architecture (consecutive-position
  invariant; `seq_cp` unsupported on hybrid recurrent+attention memory) — prototype in [`tools/afap-dualverify.cpp`](tools/)

## Reproduce

```bash
# 1. build the patched fork (see scripts/rebuild-gemm.sh)
# 2. serve behind llama-swap (config/llama-swap.example.yaml — set your API key)
# 3. bench:
python3 tools/harness.py '<json-config>'          # single-stream fixtures
python3 tools/bench_interleaved.py                # interleaved median-of-N A/B
```

Model artifacts: unsloth/Qwen3.8-27B-GGUF (IQ4_XS) + incoai/Qwen3.8-27B-DFlash2-GGUF (Q4_K_M).

## Related work

- [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090) — vLLM-based take on
  the same model for RTX 3090 (context-fill verify blocks; different engine, different cost curve)
- [Neroued/ninfer](https://github.com/Neroued/ninfer) — closed-set CUDA engine (single-stream
  slower than this stack for Qwen3.8; excellent prefill via NVFP4)
- inco.ai DFlash2 blog + z-lab DFlash drafters

## License

MIT for original content. Upstream llama.cpp components retain their respective licenses.
