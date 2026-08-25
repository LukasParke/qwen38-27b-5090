# Benchmark-1 — Baselines & Environment Recon (2026-08-21)

Mission: Qwen3.8-27B on RTX 5090 (32 GB) → ≥500 t/s single-stream, higher under concurrency,
max native context (no YaRN), no intelligence regression, fully local.

## Environment

| Item | Value |
|---|---|
| GPU | RTX 5090, 32607 MiB, driver 610.43.03, CUDA 13.3, sm_120 |
| CPU/RAM | AMD 7950X3D, 61 GB |
| Model | Qwen3.8-27B (`Qwen3_5ForConditionalGeneration`, text: `qwen3_5_text`) |
| Arch | Hybrid: 64 layers = **48 gated-delta-net linear attention + 16 full attention** (interval 4); GQA 24q/4kv @ head_dim 256, partial RoPE 0.25, hidden 5120, FFN 17408 (+shared expert gate), vocab 248320 |
| Native context | **262144** (rope_theta 1e7, no YaRN needed) |
| MTP | `mtp_num_hidden_layers=1` — model ships its own NextN/MTP head (`blk.64.nextn.*` in GGUF) |
| Serving infra | llama-swap (systemd --user) → llama.cpp fork `~/llm/cuda-dflash2` (head 5ecbe1ac1 "support DFlash2", PR #27342 lineage), port 8080→5800 |

Local model copies:
- `Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf` (17.9 GB) + mmproj-F16 (0.93 GB)
- `Qwen3.8-27B-GGUF-Q2` UD-Q2_K_XL (10.7 GB), `-Q3` UD-Q3_K_XL (13.4 GB)
- `Qwen3.8-27B-DFlash2-GGUF`: draft DFlash2 Q4_K_M (1.14 GB) + Q8_0 (2.06 GB)
- `Qwen3.8-27B-DSpark-GGUF`: DSpark Q8_0 (1.46 GB) (RadixArk; DFlash + Markov head)
- `Qwen3.8-27B-FP8` (HF safetensors, blockwise FP8), `Qwen3.8-27B-NVFP4` (HF safetensors)

Fork speculation types: `draft-simple, draft-eagle3, draft-dflash, draft-dspark, draft-mtp,
ngram-cache, ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod` — comma-separated lists cascade
(first successful draft per sequence wins; priority ngram* → simple → eagle3 → mtp → dflash → dspark).

## Baseline measurements

Harness: `/home/luke/github/afap-qwen3.8/harness.py`; ctx 8192, temp 1.0 / top-k 20 / top-p 0.95
(model gen config), thinking disabled, max_tokens 384, 3 prompts (essay/code/math), fresh context.
Metric = server-reported `predicted_per_second`.

| Config | mean t/s | notes |
|---|---|---|
| Production pre-existing (live server, ctx 131072 f16 KV, thinking on) | 102–118 t/s | measured before takeover |
| nospec (UD-Q4_K_XL, no draft) | **75.4** | matches bandwidth math: 17.9 GB ÷ ~1.2 TB/s eff. ≈ 13.3 ms/token |
| draft-dflash, DFlash2-Q4_K_M, n_max 7, draft KV q4_0 | 166.1 | accept ratio 0.42 |
| draft-dflash, DFlash2-Q8_0, n_max 7, draft KV q8_0 | **169.5** | accept 0.44 — best so far |
| draft-dspark, DSpark-Q8_0, n_max 7 | 115.4 | accept 0.26 — poor fit for this target |
| draft-mtp (built-in head), n_max 3 | 159.9 | accept 0.67 |
| draft-mtp, n_max 5 | 152.0 | accept 0.48 |
| dflash2-q8 + p_min 0.2 | 165.2 | no gain |

Published acceptance lengths for this exact target (DFlash2 blog): **DFlash2 4.80 > MTP 4.28 >
DSpark 3.62** — consistent with local ordering.

## Root-cause finding (the big lever)

llama.cpp CUDA dispatches quantized matmuls with ≤8 token rows to per-token MMVQ kernels
(`MMVQ_MAX_BATCH_SIZE 8`, `ggml-cuda.cu:1793`, `mmvq.cu:339`). Speculative verify batches are
exactly 1+n_draft ≤ 9 rows ⇒ **every drafted token re-reads the full weight matrix** (~8×17.9 GB
per verify step). PR-thread data (H200, Q4_K_M target): forcing GEMM recovered DFlash2 from
0.91× to **1.447×**.

Patch applied to fork: runtime env `GGML_CUDA_MMVQ_MAX_BATCH` (mmvq.cu, default 8 = stock behavior;
`=1` routes all multi-token batches to MMQ/GEMM which reads weights once per step). Rebuild →
`~/llm/cuda-dflash2-gemm`. Expected effect: verify step cost ≈ single-token step ⇒ throughput ≈
75 t/s × accepted-tokens-per-step (≥4.8 with DFlash2 per published numbers) ≈ 300–400+ t/s.

## Framework research verdicts (Aug 2026)

- **SGLang 0.5.18**: official validated 5090 cookbook — NVFP4+DFlash2 = **206.1 tok/s**
  (ISL 8192/OSL 1024, conc 1); NVFP4+EAGLE(MTP) = 152.9. DFlash2 needs build ≥ commit 1cf2b8c
  (newer than PyPI). Hybrid knobs: `--mamba-full-memory-ratio`, `--mamba-ssm-dtype bf16`
  (78.4 MB vs 153.9 MB state slot), `--chunked-prefill-size 2048`.
- **vLLM**: ≥0.27.x supports this arch; NVFP4 real CUTLASS kernels on sm_120; MTP via
  `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`; recipe-measured MTP
  acceptance **0.75–0.90** on 5090; single-card NVFP4 needs `--enforce-eager` (CUDA-graph capture OOM)
  and ≤32K ctx. Local 0.26.0 NVFP4 EngineCore failure consistent with capture-time OOM.
- **TensorRT-LLM 1.3.0rc2x**: arch supported but **MTP=No** for Qwen3Next-class, GeForce sm_120
  partial/community-grade → dropped.
- **llama.cpp upstream**: qwen3_5 GGUF conversion broken upstream (#27019); unsloth prebuilts work;
  draft-mtp merged 2026-05-16; `-np>1`+MTP unsupported upstream (fork's dflash path is per-slot).

Bandwidth ceilings (5090, ~1.2 TB/s effective): dense 27B Q4 ≈ 100–110 t/s hard ceiling without
speculation ⇒ 500 t/s requires spec decode with ~1 weight pass per step × ≥5 accepted tokens,
plus lighter weights (NVFP4 ≈ 14 GB ⇒ ~10.5 ms/pass).

## Next steps
1. Bench GGML_CUDA_MMVQ_MAX_BATCH ∈ {8(stock),1} × {dflash2-q8-n7, mtp-n3}.
2. Tune draft (n_max, conf-min, cascade `draft-mtp,draft-dflash`).
3. NVFP4 GGUF path for lighter weight streaming (fork has NVFP4 MMVQ/MMQ kernels).
4. Context push to 262144 native; KV precision tradeoffs (f16 vs q8_0; user data: q4_0 KV −46% deep-ctx).
5. Concurrency scaling (-np slots, backend sampling); quality parity check; final config via llama-swap.
