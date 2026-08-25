# Benchmark-7 — Deep research: speedup levers survey (2026-08-24)

## Question
How much faster can production get? Sources swept: upstream llama.cpp (git ground truth),
drafter-artifact ecosystem (HF/GitHub), alternative runtimes, weight/quant landscape,
NInfer (evaluated separately, see session log).

## Headline answer

**Nothing available today beats our stack at its own game (single-stream decode).**
Every published reproducible number for Qwen3.8-27B on a 5090 sits below our 211–293 t/s:
SGLang v0.5.18 first-class DFlash2 ≈203 t/s; NInfer nvfp4 151–220 t/s; everything else
slower or broken on sm_120. Our DFlash2-Q4_K_M block-7 drafter remains SOTA for this target.
The realistic levers are: an official larger-block drafter (path to the 500+ goal),
an NVFP4 prefill/ingest profile (~2–3× prefill), and fork hygiene via rebase.

## Lever ranking

| # | Lever | Expected gain | Cost | Risk |
|---|---|---|---|---|
| 1 | **Request block-14/16 DFlash2 from incoai** (contact@inco.ai offers custom drafters; their published top-16 oracle recall = 6.79 tokens proves τ≥6 is reachable) | ~470–550 t/s [INFERENCE: 85 steps/s × τ] if shipped | one email + bench when it lands | none |
| 2 | **NVFP4 ingest profile** (RadixArk/unsloth checkpoints now clean-provenance; fork ALREADY ships NVFP4 vec_dot + sm_120 MMQ kernels; conversion needs open PR #27161) | prefill ~2–3× vs our measured 1,706 tok/s [INFERENCE]; decode −20–25% (21.5 GB streams more bytes than IQ4_XS's 13.3 GB) | ~1 day | VRAM: needs its own low-ctx profile |
| 3 | **Rebase onto upstream master** (118 commits behind; only real conflict = #26079 which adds official `GGML_CUDA_MMVQ_MAX` runtime knob + Blackwell K-quant switch points) | small (+cuBLAS static workspace, concat memcpy, scheduler race fix); replaces our mmvq patch cleanly | ~half day incl. re-bench | patch region overlaps #26311 when it merges |
| 4 | ~~Native ~230k ctx @ f16 KV~~ **REFUTED post-measurement**: production PID uses 30.5 GiB ⇒ non-KV overhead ≈7.4 GiB (compute buffers @ub1024, CUDA ctx, draft), not ~2 GiB as budgeted. f16 ceiling ≈141k < default 131k. Keep q8_0 maxctx profile. | — | — | — |
| 5 | Watchlist: SGLang tree drafting (#36196) + dynamic verification (#36136/#36127, both opened 08-24); Apathy block-16 drafter v3 (needs GGUF converter); NInfer groupwise-int benchmark; PR #26311 NVFP4 MMVQ repack | unknown | zero (subscribe) | — |

## Negative results (do not spend cycles)

- **Drafters**: official larger-block DFlash did NOT ship (incoai org frozen at 4 repos, 08-18).
  DSpark ceiling confirmed lower by vendor data (τ 3.35–3.39 vs DFlash2 4.80). Drafter requants
  pointless (τ-bound). PARO INT4 target: zero llama.cpp support. DeepSpec pretrained drafters: don't exist.
- **Runtimes**: vLLM ≥0.28rc2 has real DFlash2 (224.6 t/s on H200) but sm_120 init broken
  (#50705) + quantized-drafter bug (#51581). TRT-LLM rc24 known-broken for hybrid MTP on SM120.
  ExLlamaV3/TabbyAPI/ik_llama.cpp/KTransformers: no block spec-decode or wrong design point.
- **Weights/KV**: no unsloth tier above IQ4_XS changed (v3 gains are ≤3-bit tiers); official Qwen
  FP8 doesn't fit VRAM and streams 2× bytes; q8/int8/fp8/iqk KV reconfirmed slower-at-depth +
  long-context quality risk (unsloth discussion #89 KLD study); UD-Q3_K_XL would give ~+7%
  bandwidth but unproven acceptance impact — eval-gated only.
- **Fork code**: fused-GDN ops (`LLM_FUSED_OP_GDN_AR/CH`) probed by default but implemented by NO
  backend — dead path. mamba2 GEMM-flatten (#27513) N/A: qwen35 GDN projections (wqkv, wqkv_gate,
  ssm_out) are already flat 2D mul_mats (verified in source). Upstream #26079's Blackwell switch
  points cover K-quants only — IQ4_XS still needs the runtime override (set `GGML_CUDA_MMVQ_MAX=1`
  post-rebase; also covers MUL_MAT_ID dispatch, unlike our hand patch).

## Key measurements this session

- Prefill (production server, warm, native /completion timings, 9,891-token prompt):
  **1,706 tok/s** (cold first-touch: 564 — CUDA graph capture artifact; chat-endpoint wall-clock:
  243 — includes template+queue overhead). NInfer reference: 8,340 tok/s @7.7k (their kernel, W4A4).

## Decision

Keep production unchanged today. Execute levers 2–4 on next maintenance window (requires GPU
downtime for rebase-rebuild + A/B bench); send lever-1 email immediately (zero downtime).

## Post-review verification (2026-08-24, inco.ai/dflash2 blog)

Reviewed the release post against our stack. Production already runs FULL DFlash2: drafter GGUF
metadata carries `dflash.conv_kernel_size=2`, `conv_group_size=16`, `selector_rank=256`,
`selector_top_k=16`, and `src/models/dflash.cpp` implements both modules (selector walk + conv).
Our live τ (3.4–4.4) is consistent with their published Qwen3.8-27B mean accept 4.80 @ block 8
(Table 4); our 211–293 t/s exceeds their claimed 2.7–3.4× AR range on stock llama.cpp because of
the MMVQ patch + f16 draft KV + backend sampling. Blog adds proof points for our lever-1 email:
Muse Glimmer ships at block size 16 (mean accept 5.70 vs 4.44 @ block 8-class DFlash), and the
custom-drafter offer explicitly covers fine-tuned targets. Remaining headroom per their own data:
oracle top-16 selection = 6.79 acceptance vs shipped ~4.6-4.8 — captured partly by SGLang tree
drafting (#36196), partly by future selector versions.

## Execution log (2026-08-24, same session)

| Item | State |
|---|---|
| Upstream rebase build | **DONE** — branch `upstream-master-20260824` @`0d0efe92f` (= master `f280b2698` + MMVQ env knob rebased over #26079 tables + qwen converter mod). Installed to `~/llm/llamacpp-upstream-20260824`, version prints `0.2.0-dev build 571`; DFlash2 spec flags present. NOTE: upstream's runtime `GGML_CUDA_MMVQ_MAX` did NOT survive review — hand patch still required and carried. Build gotcha: `/tmp` is a full 31G tmpfs shared with other tools; builds must set `TMPDIR=/home/luke/llm/tmp`. |
| llama-swap cutover profile | **DONE** — macro `server_up` + profile `qwen3.8-27b-upstream` (identical flags to prod; only binary differs) + group member added; YAML validated. Backup: `config.yaml.bak-pre-upstream-20260824`. Next window: request the profile once, run `harness.py` A/B vs prod. |
| incoai email | **DRAFTED** — `draft-incoai-email.md` (needs manual send). |
| NVFP4 ingest GGUF | **DONE + profile staged** — RadixArk checkpoint → `~/models/Qwen3.8-27B-NVFP4-RadixArk/Qwen3.8-27B-NVFP4.gguf` (28.2 GiB, arch qwen35/65 blocks validated, 313 NVFP4-class weight tensors). llama-swap profile `qwen3.8-27b-nvfp4-ingest` added (text-only, 32k ctx, ttl 300, dormant until requested). GPU smoke test pending next window. |
| Draft-sampling knob | **SCOPED, DEFERRED** — chain is hard-greedy (dflash.cpp:333 TODO); stochastic-chain flag = cheapest real lever for prose acceptance, next window alongside A/B bench. |

## A/B bench (2026-08-25, live GPU window)

Protocol: harness.py, identical flags to prod profile (IQ4_XS + DFlash2 n7 + f16 KV @131072 +
MMVQ_MAX_BATCH=1), essay/code/math ×2 rounds, T=1.0.

| Candidate | mean t/s | accept | essay | code | math | verdict |
|---|---|---|---|---|---|---|
| **A: prod binary** (5ecbe1ac1 + patch, build 570) | **216.5** | 0.443 | 120.9 | 212.3 | 316.3 | baseline reproduced |
| B: stock master build 571 | — | — | — | — | — | **cannot load production artifacts** |
| B': master + restored DFlash2 (build 571 @`eabe2a2cf`) | 207.3 | 0.384 | 113.0 | 193.2 | 315.7 | loads+runs; **−4.2% vs A**, lower acceptance → **no cutover** |
**B failure root cause**: between 5ecbe1ac1 and master, upstream REMOVED DFlash2 selector+conv
support from src/models/dflash.cpp + llama-arch.cpp (−267 lines; loader claims only 58 of the
drafter's 81 tensors → fatal count mismatch). Master's dflash targets a different drafter family
(DSpark-markov/hc). Their own blog still routes Qwen3.8 users to PR branch pr-27342, not master.

**B' staged**: revert restoring selector/conv loading applied on top of master (commit on
`upstream-master-20260824`); compile blocked by co-tenant NVMe saturation (ex-caliber build storm,
in_flight>130). Build auto-resumes when IO frees; bench then takes ~3 min via harness.py cfg in
Benchmark-7 history. Production restored and healthy after tests (:8080 serving).

## External intel: @dataTranslator (X, 2026-08-25)

Claims 500+ t/s @128k ctx, qwen3.8-NVFP4, single 5090 via **suffix-ngram cascade with DFlash2
failover** (SuffixDecoding-inspired, arXiv 2411.04975): decode → ngram hit reuses session tokens /
miss falls back to DFlash2; target verifies everything ⇒ lossless. Author: ±4% broad tasks,
2–3× on edit-heavy; no repo yet. **Consistent with our rejected Benchmark-3 cascade (190 t/s)** —
ngram rarely fired on our FRESH-generation fixtures; it pays on EDIT/copy-shaped traffic we never
benched. Actions: re-test cascade on edit-shaped fixtures before dismissing; watch for promised
writeup. NVFP4 target choice is orthogonal to the cascade mechanism.

## Cascade test + block-8 (2026-08-25, "2x" investigation)

Tested @dataTranslator's mechanism directly: the fork ALREADY chains spec implementations
(`--spec-type` accepts comma lists; first successful draft wins per sequence). Six fixtures:
3 fresh-gen + 3 NEW copy/edit-shaped (edit-code module rewrite, copy-table reformat,
doc-extend changelog) where suffix-reuse should pay.

| Config (n_max 7) | mean t/s | accept | edit-code | copy-table | doc-extend |
|---|---:|---:|---:|---:|---:|
| C0: draft-dflash only | **267.0** | 0.585 | 271.4 | **357.6** | **352.5** |
| C1: ngram-simple,dflash defaults | 221.9 | 0.466 | 199.6 | 347.0 | 207.8 |
| C2: cascade tuned (n16/m24/hits1) | 246.2 | 0.563 | 247.9 | 354.6 | 283.3 |

**Cascade REFUTED on this stack**: plain DFlash2 already owns copy/edit fixtures at 271–358 t/s
(block-diffusion drafting exploits context repeats itself); the ngram layer only adds wasted
verify width. The X post's gain does not transfer to this engine.

**Adopted instead**: `--spec-draft-n-max 7 → 8` on all profiles. Artifact metadata
(`dflash.block_size = 8`) shows vendor trained/published at block 8; we shipped 7. Measured
same-session 267.0 → 274.6 t/s (+2.9%), accept 0.585 → 0.605; within noise individually but
matches vendor config so promoted as default. Live server restarted with n8, verified serving.

**"2×" physics check**: decode ≈ ~88 cycles/s × τ; τ now ≈ 4.9 at block 8. Reaching 2× needs
τ≈10 — above the artifact's own oracle ceiling (top-16 recall = 6.79), so not reachable by
configuration. It requires a larger-block drafter (the incoai request) or a different workload
class (NVFP4 prefill for TTFT-bound ingest). `bench_cascade.py` committed for future fixture
reuse.

## Tier-1 config matrix (2026-08-25, n8 fixtures ×6, single-pass)

| Config | mean t/s | accept | verdict |
|---|---:|---:|---|
| MMVQ=1 (prod) | 269.9 | 0.621 | local optimum |
| MMVQ=2 | 272.0 | 0.613 | noise (+0.8%) |
| MMVQ=3 | 262.9 | 0.594 | worse |
| Q8_0 drafter | 261.5 | 0.599 | worse — heavier stream, no acceptance gain |
| PDL=1 + MMVQ=2 | 265.9 | 0.598 | no gain |

Config space exhausted; production config confirmed at its local optimum.

## Remaining-ideas build & benchmark (2026-08-25, goal session)

Built both code-level ideas on branch `afap-cascade` (prod lineage + features, binary in
`~/llm/llamacpp-afap-cascade`, env-gated off by default):

| Idea | Implementation | Result (interleaved median-of-3, same binary) |
|---|---|---|
| Selector-gap truncation (`SPEC_TRUNC_GAP`) | top1−top2 gap recorded per walk position; block cut before hopeless tail | **REFUTED for speed**: GAP=1.0 halves throughput (essay 100.8→85.3, code 189.9→104.5) even though acceptance RISES (essay 0.238→0.619). Verify rows are nearly free; tokens/cycle is everything. |
| Linear ngram tail (`SPEC_TAIL_N`) | dflash block 0–6 + context-copy continuation 7–13 verified in one pass | Parity (240.5 vs 243.3): copy fixtures already accept 0.977–0.997 within 7 drafts — nothing left for the tail to add; fresh-gen never matches. |
| Session ngram-cache cascade (`ngram-cache,draft-dflash`) | config-only, built-in chain | **REFUTED**: math 317→145–195, copy-table 358→120–274 across passes. First-impl-wins semantics let weak ngram drafts displace strong DFlash drafts. |
| In-cycle drafter re-iteration | feasibility read | Infeasible without retraining: DFlash2 is single-pass denoising by design ([id_last + masks] → lattice); DSpark anchor-first is the alternative architecture and its drafter measured strictly worse for this target. |

**Law of this stack (now measured three independent ways)**: throughput = cycles/s × τ,
cycles/s ≈ const (~88), verify width ≈ free, τ is bounded by drafter quality, and drafter
quality ceiling = artifact oracle (top-16 recall 6.79 ⇒ max ~380–420 t/s on favorable fixtures).
Every width/selection manipulation moves τ down or sideways, never up.

**Bonus findings**: (a) the server already feeds request temperature into the DFlash2 selector
walk (`server-context.cpp:2944` → dp.temperature) — stochastic path drafting is live at T>0;
(b) `--spec-draft-n-max 8` is clamped to 7 internally (`block_size − 1`), so the earlier "block-8
gain" was session noise — reverted understanding, config unchanged in effect; (c) B′ −4% vs prod
was semantic hybridization: master's dflash impl samples via backend logits, having dropped the
lattice walk our artifact requires.

**Open (goal stays active)**: dual-pass wide verification — two stochastic blocks per cycle via
per-cycle KV fork (`llama_memory_seq_cp`) + best-path accept. Premise validated (width ≈ free);
estimated 2–3 days of systems work across speculative.cpp/server-context; deferred to avoid
production regression risk in a single session.

## Dual-path wide verification: build attempt (2026-08-25, continued)

Standalone measurement tool `afap-dualverify.cpp` (preserved in repo + ~/llm) written against
llama.h + common_speculative: two stochastic DFlash2 blocks per cycle, verified in one batched
target pass across two sequences (`n_parallel 2`), greedy-lossless acceptance, best branch wins
(`llama_memory_seq_cp` returns winner to canonical seq). Single-mode smoke test reached model load;
**blocked at first drafter noise decode**: `llama_decode(ctx_dft)` returns "failed to initialize
batch" with n_tokens=1 — impl's internal `params.n_max` reads 0 through the stripped-argv path
(production server path sets it via -nd). Next session: pass `--spec-draft-n-max 8` explicitly in
the tool args and debug with `-lv 3`; if the noise batch still under-fills, instrument
`common_speculative_impl_draft_dflash::draft()`.

Design is otherwise complete and validated where testable:
- width-free verification premise confirmed by GAP-truncation experiment (fewer rows ≠ faster);
- lossless best-path accept semantics mirror `common_sampler_sample_and_accept_n(dists)`;
- expected gain bounded +25–30% short-context per oracle ceiling (τ ≤ 6.79).

**Debug addendum (same session)**: with `--spec-draft-n-max 8` explicit, the standalone tool
fails identically — `process()` sees a 1-row seq-main chunk whose `llama_decode(ctx_dft)` returns
rc=-1 at offset=0, on every attempt (single+dual). Hypotheses ranked: (1) drafter context
`n_ctx_seq`/`n_ubatch` mis-sized through the standalone `common_speculative_init_from_params`
path (server sets these differently); (2) `params.draft.n_max`=0 reaching the impl despite CLI
(example printed n_max=3 with `-n 48`, suggesting arg coupling elsewhere); (3) latent assert in
per-seq range scan. Next session: run tool with `-lv 3` and read the drafter-context print_info
block; compare `llama_n_ubatch(ctx_dft)` vs server's; consider initializing the speculator via
the server binary itself (`--spec-type ngram-cache,draft-dflash` style composition) instead of
standalone. Production restored and healthy after all windows.

**Debug addendum 2**: instrumented walk confirms drafting itself is healthy in the standalone
tool (`AFAPWALK: n_block=8 result=7`, gaps sane at both T=0 and T=1.0). Failure is downstream:
the hand-built verify batch triggers the consecutive-position invariant with reported values
(X=83 vs Y=86) that do not match the constructed positions (anchor=n_past=84) — indicating
row reordering/rejection inside batch handling under `--parallel 2` / multi-seq that needs
`-lv 4` tracing to resolve. Tool source preserved at `afap-dualverify.cpp`; build one-liner in
session log. Production restored healthy.

**Debug addendum 3**: batch dump (`AFAP_DUMP=1`) proves constructed rows are CORRECT
(anchor@84 + drafts through 91, all seq 0, logits on; `NPAST=84`), yet the validator rejects
claiming start Y=86 vs stored X=83 — i.e. the discrepancy lives INSIDE the memory/balloc layer
under multi-sequence contexts, not in batch construction. Working hypothesis: the unified/recurrent
cache's per-seq position tracking interacts with `--parallel 2` context creation such that seq 1's
absence from prefill shifts seq 0's expected start. Next session: (a) retry dual-mode with BOTH
sequences prefilled (duplicate prompt rows across seq 0+1 during prefill so neither is empty);
(b) if still failing, instrument `llama-batch.cpp init()` directly.

## Goal-session close-out: all remaining ideas resolved (2026-08-25)

| Idea | Build | Benchmark | Verdict |
|---|---|---|---|
| Beam-2 selector walk (`SPEC_BEAM=2`, T=0) | ✅ deployed in cascade binary | math **+4.1%** (150.4→156.5 paired), edit-code −2.7%; acceptance identical (τ unchanged) → **parity, kept as flag** |
| Dual-seq wide verification (idea 1 core) | ✅ standalone tool `afap-dualverify.cpp` | **ARCHITECTURE-BLOCKED, demonstrated**: (i) consecutive-position invariant rejects branch batches (`Y=X+1`); (ii) `seq_cp()` asserts unsupported on `llama_memory_hybrid` (GDN recurrent state not forkable) — `llama-kv-cache.cpp:502`. Requires upstream tree-attention + forkable hybrid state. |
| Selector-gap truncation | ✅ | Refuted (−56% at gap=1.0 despite acceptance gains) |
| Session ngram-cache cascade | ✅ config-level | Refuted (−33…−64% mixed traffic) |
| Drafter re-iteration | Assessed | Infeasible without retraining (single-pass denoising by design) |

Also corrected en route: `--spec-draft-n-max 8` clamps to 7 internally (block_size−1) — the
"block-8 gain" earlier was session noise; and request temperature already drives stochastic
selector walks in production (`server-context.cpp:2944`).

**Campaign conclusion**: production stack remains the fastest known Qwen3.8-27B configuration on
one RTX 5090 (211–293 t/s verified envelope; ~243–275 t/s mean across the extended fixture set).
Every remaining idea is now either adopted (block-width alignment), refuted by measurement, or
blocked by demonstrated architecture limits. The only route past the ~380–420 t/s oracle ceiling
is a better drafter artifact: incoai block-14/16 request (email drafted, awaiting send).

## syv-ai/qwen38-27b-rtx3090 transfer analysis (2026-08-25)

Ingested the vLLM-based RTX 3090 repo (same model family, LABD lookup-augmented drafting).
Root-caused OUR earlier tail parity: helper refuses drafts shorter than key size
(`copy_max < n_draft_min -> empty`) — key8/mgram7 could never fire. Fixed (mgram >= key);
tail now fires (verbatim fixture, T0 greedy): tau 5.82 -> 6.60 (+13%), BUT t/s 145.7 -> 115.2
(**-21%**): cycle cost grows ~1.5 ms/row beyond width 8 (attention/logits/draft-inject scale),
while tau gains dilute (accept ratio 0.70 -> 0.46). Combined with the truncation experiment
(fewer rows also loses), **our stack sits exactly at its verify-width optimum (8 rows)**.
syv's +47% reproduction gain exists because vLLM's V2 runner has flat-cost fixed-16-row
verify blocks — a different cost curve, not a portable mechanism.

Other concepts audited: prefix-cache recurrent-state resume — already equivalent
(--cache-ram); draft-vocab restriction / int4 calibrated lm_head — vLLM drafter-format only,
not portable to our GGUF drafter; split-KV verify attention / marlin-int8 / KVarN lossy KV —
vLLM-specific or previously rejected quality trades. Net performance transfer: none.

Their numbers for context (RTX 3090 24 GB, vLLM): single-stream 120–133 t/s, reproduction
382 t/s @25k ctx, batch 1035 t/s @64 concurrent — all below or orthogonal to our envelope.

## Dual-path final verdict (2026-08-25, corrected-env matrix)

Single-path baseline (env-parity): essay 103.7 / math 122.9 / edit-code 109.2 / verbatim 128.8 t/s
(absolute values lower than server envelope because standalone tool lacks backend sampling —
relative comparisons valid). Dual-path runs: ALL crash at `llama-kv-cache.cpp:502
seq_cp() only supported for full KV buffers` via `llama_memory_hybrid::seq_cp` — the GDN
recurrent-state container has no fork primitive. Combined with the consecutive-position
invariant, branch verification is architecturally impossible on current llama.cpp APIs for this
model. Would require upstream: fork/copy support for recurrent state in llama_memory_hybrid +
tree-mask attention for branch batches. CLOSED as architecture-blocked (not estimate — assert).

Beam-2 walk (T=0 paired): math 150.4→156.5 (+4.1%), edit-code 188.5→183.5 (−2.7%), accept
identical — parity; kept behind SPEC_BEAM=2.

## NVFP4 ingest smoke test (2026-08-25) — PASSED

Cascade binary loads `Qwen3.8-27B-NVFP4.gguf` (RadixArk conversion, 28.2 GiB) and serves.
Measured at ~9.3k-token prompt, T=0, no spec: **prefill 3,805 tok/s (+123% vs IQ4_XS baseline
1,706)**; no-spec decode 53.9 t/s (−28% vs IQ4_XS ~75, as predicted from byte math). Profile
`qwen3.8-27b-nvfp4-ingest` validated end-to-end for prefill-heavy/ingest workloads.
