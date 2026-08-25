# afap-qwen3.8 — Qwen3.8-27B local inference optimization campaign

**Current production** (`:8080` via llama-swap, profile `qwen3.8-27b`):
IQ4_XS + DFlash2 block-diffusion drafting + `GGML_CUDA_MMVQ_MAX_BATCH` kernel patch +
f16 KV @131072 native context. Verified: **211 t/s sustained / 293 t/s math-shaped
single-stream, 285 t/s aggregate @4 streams**, quality parity with Q4_K_XL (24/25 eval).
Additional profiles: `qwen3.8-27b-x4` (concurrency), `qwen3.8-27b-maxctx` (262144 native).

## Documents

| File | Content |
|---|---|
| Benchmark-1.md | Environment, baselines, framework research verdicts |
| Benchmark-2.md | Speculative-decoding matrix, kernel patch, KV/context findings |
| Benchmark-3.md | Production cutover, concurrency, roadmap |
| Benchmark-4.md | Custom block-16 drafter campaign: infrastructure |
| Benchmark-5.md | Campaign log, diagnosis chain, v2 gate sequence |
| Benchmark-6.md | v2 data campaign: complete results & final verdict |
| Benchmark-7.md | Deep research: speedup levers survey (upstream/runtime/drafter/quant) |

## Delivered 2026-08-24 (Benchmark-7 execution)

- `draft-incoai-email.md` — request for block-14/16 DFlash2 drafter (send manually)
- `~/llm/llamacpp-upstream-20260824` — upstream-master rebase build (`upstream-master-20260824`
  branch @0d0efe92f; MMVQ_MAX_BATCH env knob carried over #26079 tables). Cutover-ready;
  A/B bench pending maintenance window.
- `~/models/Qwen3.8-27B-NVFP4-RadixArk` + GGUF — NVFP4 ingest artifact (conversion pipeline
  validated on master converter; decode-regression caveat applies, prefill-only profile).

## Custom drafter pipeline (built & working; v1 model underpowered)

`draft-training/` contains the complete loop: capture (SGLang hidden-state extraction),
training (SpecForge + 8-bit optimizer patches), probe (`probe_specforge_forward.py`),
export/convert/serve (`export_drafter.sh`). v1 verdict: recipe learns but 12k raw
conversations cap generalization at ~8% top-1 — see Benchmark-5 for the full
experimentally-grounded diagnosis chain.

## Reaching 500+ t/s

Requires τ≥6 accepted tokens/cycle. Two paths, both scoped in Benchmark-5:
1. **v2 data campaign** (~2–4 days GPU): `draft-training/v2_launch.sh` (turnkey;
   regenerates conversations against the target, retrains, gates on live acceptance).
2. Watch z-lab for official larger-block DFlash releases; PARO INT4 as future quality option.
