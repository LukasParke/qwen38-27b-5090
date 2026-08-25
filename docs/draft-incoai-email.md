# Draft email — request for block-14/16 DFlash2 drafter (Qwen3.8-27B)

**To:** contact@inco.ai
**Subject:** Custom drafter request: block-14/16 DFlash2 for Qwen3.8-27B (+ GGUF)

---

Hi Inco team,

Congratulations on the DFlash 2 release — the path-selector results are impressive, and
we're running your Qwen3.8-27B-DFlash2 in production on an RTX 5090 via llama.cpp
(PR #27342 lineage) with the MMVQ→MMQ verify patch.

Your blog's own numbers motivate this request:

1. **Block-size headroom.** Muse Glimmer ships at block size 16 with mean acceptance 5.70,
   and your Table 1 shows Qwen3.8-class targets have top-16 oracle recall of 6.79 tokens/cycle
   vs ~4.6–4.8 shipped today. A block-14/16 Qwen3.8-27B drafter is the single biggest
   throughput lever left on our stack (~85 verify steps/s × τ; τ≥5.5 puts us near 500 t/s
   single-stream).
2. **The suffix-decay conv** makes longer blocks viable at +3% params / +0.7% cycle latency
   per your Figure 2 — exactly what a bigger block needs.
3. **GGUF distribution.** We consume drafters through the `dflash` GGUF arch
   (`incoai/Qwen3.8-27B-DFlash2-GGUF` works today, selector + conv metadata intact).
   Could a larger-block variant ship with a GGUF mirror?

Our workload: single-user agent coding/analysis on Qwen3.8-27B IQ4_XS, temperature
0.6–1.0, long contexts (32k–131k). Live sampled acceptance with current DFlash2 is
42–62% task-dependent; prose remains our weakest bucket.

You mentioned custom drafters on request, including fine-tunes. If a Qwen3.8-27B
block-14/16 variant is on your roadmap or feasible as a commissioned artifact, we'd
like to evaluate it day one and share benchmarks back.

Thanks,
Luke
afap-qwen3.8 inference campaign

---

*Send from Luke's personal/professional address. Optionally attach: live benchmark table
(Benchmark-3.md verified results) as evidence of serious evaluation.*
