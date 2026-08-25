# Benchmark-4 — Custom block-16 diffusion drafter: training campaign (2026-08-22)

## Rationale

Post-Benchmark-3 analysis: single-stream t/s ≈ (one weight pass ≈ 12 ms @ IQ4_XS)⁻¹ × accepted
tokens/cycle ≈ 85 × τ. DFlash2 block-8 delivers τ≈3.3 → ~210–290 t/s task-dependent. 500 t/s
requires τ≥6 ⇒ a drafter accepting ≥37% per position across a **block-16** window — only a
semi-autoregressive (Markov-head) diffusion drafter plausibly sustains that (DSpark paper's core
claim: mitigates suffix decay). No public block-16 drafter exists for Qwen3.8-27B ⇒ train one.

## Pipeline (SpecForge v0.3.0, the trainer RadixArk used for the shipped Qwen3.8 DSpark)

Working dir: `draft-training/`. Venv: `~/github/specforge-src/.venv` (py3.13; sglang 0.5.14 pinned
by specforge; torch 2.11).

| Stage | Result |
|---|---|
| Data | perfectblend subset, 12,000 conversations (`cache/dataset/perfectblend_train.jsonl`) |
| Hidden-state capture | **11,480 records / 375 GB** (`cache/hidden_states/qwen38-27b-dspark-b16/`), FP8 target via embedded SGLang, ~75 min |
| Config | `qwen3.8-27b-dspark-b16.json`: block_size **16**, 5 dense layers, GQA 32q/8kv hd128, Markov rank 256 vanilla, confidence head, target layers [3,15,27,39,51] (= RadixArk's FA-layer choice) |
| Training | `qwen3.8-27b-dspark-b16-offline.yaml`: 3 epochs / 6000 steps max, lr 6e-4 cosine, warmup 4%, batch 1, objective_chunk_blocks 32, **optimizer_cpu_offload** (fp32 Adam states → CPU; GPU couldn't hold them) |

### Host-specific patches required (all in `~/github/specforge-src`, git-diff visible)

1. `scripts/prepare_hidden_states.py::_sglang_kwargs` — added `mamba_ssm_dtype: bfloat16`
   (fp32 GDN state = 146.8 MB/req made the pool unallocatable next to FP8 weights).
2. `specforge/distributed.py::_distributed_backend` — `SPECFORCE_DIST_BACKEND` env override;
   NCCL cannot bootstrap in the ~1 GB left beside FP8 weights; collectives run over gloo.
3. SGLang site-packages: applied SpecForge's own `apply_sglang_spec_capture_patch.sh`, then hand-
   patched `models/qwen3_5.py`: inner model gained `set_dspark_layers_to_capture` alias; VL wrapper
   (`Qwen3_5ForConditionalGeneration`) gained dflash/dspark setter forwarders — upstream exposes
   neither (offline DSpark capture for qwen3_5 VL targets was an untested path).
4. `specforge/offline_capture/sglang_backend/capture.py::capture_eagle3` — SGLang returns
   `(final_hidden, [per-layer tensors])` tuples through the VL wrapper; decode that structure
   (final → last_hidden_states; layers → concat on hidden dim = aux).
5. Ops: `CUDA_HOME=/opt/cuda` (flashinfer JIT), `MAX_JOBS=2` (cicc OOM-killed at default
   parallelism), `TMPDIR=/home/luke/tmpdir` (/tmp tmpfs was full from unrelated projects),
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Memory map that finally fit (single 5090): FP8 weights ~29.5 GB + bf16 GDN slots 5×74.8 MB +
KV pool @0.97 fraction, batch 1, ctx 2048 (4096-length forwards OOMed in fla chunk kernels;
records captured at ≤2048 tokens, training max_length matched).

### Training status

Stepping at ~7 steps/min (CPU-offloaded Adam): loss decreasing (2.78 → 2.71 by step 20–40);
transient CE spike at step 60 during warmup recovered by step 80 (acc 8.7% top-1 @ 248k vocab,
early). ETA to 6000 steps ≈ 15 h. Checkpoints every 1000 steps →
`draft-training/outputs/qwen38-27b-dspark-b16/`.

## Next

1. Export HF (`specforge export --to hf`) → convert GGUF via fork converter
   (`convert_hf_to_gguf.py <hf> --target-model-dir <target> --outtype f16`) → serve
   `--spec-type draft-dspark --spec-draft-n-max 16`.
2. Bench vs DFlash2-Q4 block-8 on harness + eval_quality gate.
3. Acceptance math: block-16 needs mean ≥0.37/position to beat block-8's 0.48-equivalent τ≈3.3;
   τ_target = Σ survival probabilities. If suffix decay dominates, iterate (gated head,
   loss_decay_gamma 8, more data/regenerated outputs).
4. Graph-reuse multi-slot patch (separate track, ~6% ceiling found in A/B: reuse-on 197.4 vs
   reuse-off 186.2 t/s).


### Run log
- Run 1 (53 min): crashed at ~step 200 — `ValueError: offline DFlash-family samples require two
  consecutive supervised tokens`. Root cause: 91/11,479 captured records have <2 consecutive
  supervised tokens within the 1024-token training window (assistant span truncated away).
  Fix: scanned all records with the exact train-time predicate (`scan_bad_records.py`, mmap +
  24 workers), removed the 91 offenders (`bad_records.json`). save_interval 1000 → 500.
- Run 2: healthy restart; warmup to step 240, CE declining (10.6 → 9.3 @ step 60), per-position
  top-1 climbing. ~7 steps/min ≈ 15 h to 6000 steps.

### Reliability engineering (run 3 death → runs 5-6)
- Run 3 killed by the benchmark tool's own 1 h timeout ⇒ moved to a hub-managed persistent
  process (`dspark-train`).
- Run 4 died silently at step 123 (exit 137, no traceback in PTY). Root cause via kernel log:
  **global OOM killer** — trainer anon-RSS ~19-21 GB (CPU-offloaded fp32 Adam) + user's desktop
  workloads (Chrome 55 GB VM) exhausted the 61 GB host; OOM also took Chrome processes.
- Mitigations: supervisor loop (`train_supervised.sh`) with `training.resume_from` full-state
  resume; save_interval 250→**50** steps (~7 min blast radius); full output tee'd to
  `train_live.log` (tracebacks survive PTY loss); `oom_score_adj +300` on the trainer so under
  pressure it dies before user applications.
- 91/11,479 records removed for the <2-consecutive-supervised-tokens predicate (see Run 1).

### Pipeline validation (CPU-only, while training runs)
step-50 checkpoint → `specforge export --to hf` ✓ → `convert_hf_to_gguf.py --target-model-dir
<FP8 target>` ✓ (needs `gguf` pip-installed into the specforge venv; run converter with the venv
python) → 3.7 GB GGUF, metadata verified: `dflash.block_size=16`, `sample_from_anchor=true`
(16 drafts/cycle), mask_token 248070, Markov + confidence heads present.
Remaining untested: GPU serve as `--spec-type draft-dspark --spec-draft-n-max 15`.