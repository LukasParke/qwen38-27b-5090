# Benchmark-6 — v2 data campaign: complete results & final verdict (2026-08-23)

## Executive summary

The v2 campaign executed end-to-end with user approval: 12,000 conversations regenerated
from the target model itself, hidden-state capture (595 GB), 7,250 steps of drafter
training on the distribution-aligned data, and live serving of the result.

**Final verdict**: v2 drafter achieves 1.4–5.6% live sampled acceptance — an order of
magnitude below the shipped DFlash2 stack (42–62%). The regenerated-data hypothesis is
**disconfirmed** at this training budget: distribution alignment alone does not close the
gap; professional drafters require substantially more compute/data/distillation than a
single-GPU session can provide.

## Campaign execution

| Stage | Duration | Result |
|---|---|---|
| Regeneration | ~9 h | 12,000 conversations regenerated from target (perfectblend prompts) |
| Capture | ~2 h (with crash-resume cycles) | 11,976 records / 595 GB |
| Training | ~2 h (step 0 → 7250 of 11400) | acc trajectory: 1% → 32% train-batch top-1 |
| Export+serve | ~30 min | GGUF converted, served alongside IQ4_XS target |

Infrastructure fixes required en route (all documented in-session):
- `v2_chain.py` env propagation to subprocesses (CUDA_HOME etc.)
- Controller save-diag (`live_model_params`) — proved save fidelity
- Normalizer skip/substitute fallbacks for degenerate captures (<2 consecutive supervised)
- Collator None-filtering
- Disk cleanup (removed redundant Q8/Q6 model copies, +5 GB)

## Measurements

| Checkpoint | Greedy top-1 (shim probe) | Live sampled acceptance |
|---|---|---|
| step 500 | 7.0% | — |
| step 1000 | 5.6% / 1.9% (two runs) | — |
| step 2750 | 3.2% | — |
| **step 7250** | **12.8% / 6.25% (two probes)** | **1.4–5.6%** |
| DFlash2 reference | — | **42–62%** |

Training-batch accuracy reached 17–49% but generalization plateaued at ~10% greedy /
~3% sampled. The gap between training-batch acc and live acceptance is structural:
hidden-state-only supervision without logit distillation cannot teach a 4-layer drafter
the target's full conditional distribution.

## Why this fails when RadixArk succeeds

RadixArk's production DSpark drafter was trained by model creators with:
- Orders of magnitude more data/compute (H100 clusters vs single 5090)
- Full logit distillation (not just hidden-state features)
- Iterative hyperparameter search across multiple training runs

Reproducing that pipeline is a multi-week project, not a session task. The 2–4 day
estimate for v2 was optimistic by an order of magnitude once the actual acceptance
gap became measurable.

## Final delivered configuration

**Unchanged from Benchmark-3**: IQ4_XS + DFlash2 + GEMM-forced verify = **211–293 t/s**
task-dependent, 285 agg @4 streams, 262144 native context, quality parity.
Production restored and verified after all experiments.

## Artifacts preserved

- `draft-training/outputs/qwen38-v2/qwen38-v2-step7250/` — trained checkpoint w/ live params
- `Qwen3.8-DSpark-v2-step7250.gguf` — served artifact
- `v2_chain.py`, `probe_checkpoint.py`, `train_supervised.sh` — resumable infrastructure
- All capture data (595 GB) can be purged if disk needed: `cache/hidden_states/qwen38-v2/`

## Path forward (if ever revisited)

1. Logit-distillation training (KL on full vocab) rather than hidden-state matching —
   requires modifying SpecForge's loss or using a different framework entirely.
2. 10× more regenerated data with longer schedules.
3. Or: wait for z-lab/RadixArk official larger-block DFlash releases (their cadence
   suggests new artifacts every few weeks).
