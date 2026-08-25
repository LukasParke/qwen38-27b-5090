# Benchmark-3 — Final configuration, cutover & roadmap (2026-08-21)

## Delivered state

Production (`llama-swap`, :8080) cut over to the optimized stack. Three selectable profiles
(swap by model name; group `gpu` is exclusive so only one loads at a time):

| model name | purpose | key flags |
|---|---|---|
| `qwen3.8-27b` | default / speed | IQ4_XS main + DFlash2-Q4_K_M draft n7, f16 KV @131072, `-bs`, GEMM-forced verify |
| `qwen3.8-27b-x4` | concurrency | same + `--parallel 4` (32k ctx/slot), aggregate 285 t/s @4 streams |
| `qwen3.8-27b-maxctx` | max native context | `-c 262144` (native, no YaRN), q8_0 target KV + q4_0 draft KV, 27.4 GB VRAM |

Binary: `~/llm/cuda-dflash2-gemm` (build 570 = dflash2 fork head 5ecbe1ac1 + `GGML_CUDA_MMVQ_MAX_BATCH`
runtime knob in `ggml/src/ggml-cuda/mmvq.cu`; unset env ⇒ stock behavior). Config backup:
`~/.config/llama-swap/config.yaml.bak-pre-gemm-20260821`. Old binary prefix `~/llm/cuda-dflash2`
untouched (rollback macro `server_df2` retained in config comments).

## Verified results (live through :8080 proxy)

| workload | t/s | accept |
|---|---|---|
| code generation | **217–258** | 0.54–0.67 |
| math/calculus | **293** | 0.79 |
| technical essay | 114 | 0.23 |
| stock baseline (start of session) | 75 nospec / ~118 prod | — |

Throughput is acceptance-driven and task-dependent: structured output (code/math/reasoning)
runs 200–300 t/s; open-ended prose drafts poorly (0.2) and lands near 110–130.
Single-stream ceiling analysis: verify ≈ one weight pass (~12 ms @ IQ4_XS over ~1.2 TB/s effective);
t/s ≈ 85 steps/s × accepted-tokens-per-cycle. 500 t/s needs τ≈6/cycle (accept ≈0.85 at block 7) —
no public drafter reaches this on this target; even Q1 weights at τ=3.3 cap ~370 t/s.

## What moved the needle

1. **GEMM-forced verify (+11%)**: MMVQ re-streamed weights per drafted token on ≤9-row batches;
   patched dispatch reads weights once per step. Env: `GGML_CUDA_MMVQ_MAX_BATCH=1`.
2. **DFlash2 block-diffusion drafting** (diffusion-style drafter, one denoising pass per block):
   75 → 169.5 t/s. DFlash2 > MTP (4.80 vs 4.28 published accept) > DSpark (rope-mismatched locally).
3. **f16 draft KV** (+2.5% vs q8_0), **backend sampling** (`-bs`, +3%), **IQ4_XS main**
   (14.3 GB vs 17.9 GB, quality parity 24/25 both).
4. Rejected empirically: DSpark draft (115 t/s), MTP under GEMM (139), n_max 5 (< n7),
   p_min/conf-min (inert for DFlash2), q8_0/q4_0 target KV (slower than f16 at depth),
   NVFP4-MTP third-party GGUF (207.6, provenance risk), CUDA graph opt env (no gain),
   ngram+dflash cascade (190 — ngram rarely fired on these workloads).
5. Framework research verdict: SGLang cookbook best published = 206.1 t/s (NVFP4+DFlash2, 5090);
   local result exceeds it. vLLM 0.27 single-card NVFP4 requires `--enforce-eager` (large penalty);
   TRT-LLM lacks MTP for this class. llama.cpp fork path wins.

## Quality/intelligence guarantees

- Speculative decoding (all modes used) is lossless by construction — rejection sampling preserves
  the target distribution exactly.
- Quant change Q4_K_XL → IQ4_XS verified: 24/25 vs 24/25 on greedy verifiable-question eval
  (`eval_quality.py`, logs/eval-*.json). No other quality-affecting changes.
- Context is pure native RoPE (theta 1e7); no YaRN anywhere.

## Concurrency behavior

211 (1 stream) → 195 agg (2) → 285.6 agg (4 streams, `-np 4`). Sublinear because the draft pass
batches per-slot blocks and CPU sampling gates high slot counts (`-bs` mitigates). Higher slots
(`--parallel 8`) possible via config edit; ctx splits across slots.

## Roadmap to 500+ (requires new artifacts, not tuning)

1. **Custom larger-block diffusion drafter** (primary): train DSpark-class drafter (block 16–32,
   Markov head, confidence-scheduled truncation) on Qwen3.8-27B with deepseek-ai/DeepSpec
   (open-sourced with arXiv 2607.05147). τ≥6 ⇒ ≥500 t/s within current physics. Feasibility scan:
   DeepSpecScout (running). Needs: BF16/FP8 target for hidden-state extraction, corpora, GGUF
   conversion via fork's DFLASH arch (`conversion/qwen.py` DSparkModel/DFlashModel).
2. **Draft/verify overlap** in fork (run draft block for cycle N+1 during verify of N): ~10%.
3. **Confidence-scheduled verification** (DSpark §3.2): prunes wasted verify width under
   concurrency; needs a confidence head (currently DFlash2 ships none).
4. Watch for: upstream llama.cpp qwen3_5 converter fix (#27019), SGLang DFlash2 build (≥1cf2b8c),
   larger-block public DFlash releases (z-lab collection).

## Artifacts

- `harness.py`, `bench_deep_conc.py`, `eval_quality.py` — reproducible benches
- `logs/server-*.log`, `logs/eval-*.json` — raw runs
- Patch: `git diff ~/llm/llama.cpp-src` (mmvq.cu only), rebuild script `~/llm/rebuild-gemm.sh`
