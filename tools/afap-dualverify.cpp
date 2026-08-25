// afap-dualverify: standalone measurement of dual-path wide verification.
// Two independent stochastic DFlash2 blocks per cycle are verified in ONE
// batched target pass across two sequences; the better branch becomes the
// continuation (lossless greedy check). Measures steady-state decode t/s.
#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static const char * PROMPTS[] = {
    // 0 essay (fresh prose, low accept)
    "Write a detailed technical essay about the history of GPU computing, covering CUDA, tensor cores, and modern inference optimizations. Be thorough and specific.",
    // 1 math
    "Solve step by step: A factory produces widgets where the cost function is C(x) = 0.01x^3 - 3x^2 + 500x + 2000 and revenue R(x) = 250x - 0.5x^2. Find the production level x that maximizes profit, show all calculus work, and compute maximum profit.",
    // 2 edit-code (copy-heavy)
    "Here is a Python module:\n\nclass Ledger:\n    def __init__(self):\n        self.entries = []\n    def add(self, account, amount):\n        self.entries.append((account, amount))\ndef transfer(ledger, src, dst, amount):\n    ledger.add(src, -amount)\n    ledger.add(dst, amount)\n\nclass Ledger:\n    def __init__(self):\n        self.entries = []\n    def add(self, account, amount):\n        self.entries.append((account, amount))\ndef transfer(ledger, src, dst, amount):\n    ledger.add(src, -amount)\n    ledger.add(dst, amount)\n\nRename every 'entries' to 'records' and output the complete updated module verbatim.",
};

static llama_token argmax_row(llama_context * ctx, int row, const llama_vocab * vocab) {
    const float * lg = llama_get_logits_ith(ctx, row);
    llama_token best = 0;
    for (llama_token t = 1; t < (llama_token) llama_vocab_n_tokens(vocab); ++t) {
        if (lg[t] > lg[best]) best = t;
    }
    return best;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    if (argc < 3) {
        fprintf(stderr, "usage: %s single|dual fixture_idx [n_predict]\n", argv[0]);
        return 1;
    }

    // strip our leading args so common_params_parse only sees model/sampling flags
    const bool dual = strcmp(argv[1], "dual") == 0;
    const int fixture = atoi(argv[2]);
    const int n_predict_cap = argc > 3 ? atoi(argv[3]) : 384;

    std::vector<char *> pargs;
    pargs.push_back(argv[0]);
    for (int i = 4; i < argc; ++i) { pargs.push_back(argv[i]); }

    common_params params;
    common_init();
    if (!common_params_parse((int) pargs.size(), pargs.data(), params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }
    params.prompt = PROMPTS[fixture];
    params.n_parallel = dual ? 2 : 1;

    const auto output_limits = common_speculative_get_output_limits(
            params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));
    params.n_outputs_max = output_limits.total;
    params.n_outputs_max_per_seq = output_limits.per_seq;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init_tgt = common_init_from_params(params);
    llama_model * model_tgt = llama_init_tgt->model();
    llama_context * ctx_tgt = llama_init_tgt->context();
    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    // NOTE: spec_init must outlive everything below - owns the drafter model + context
    common_params params_dft = common_base_params_to_speculative(params);
    auto spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);
    params.speculative.draft.ctx_tgt = ctx_tgt;
    params.speculative.draft.ctx_dft = spec_init->context();
    llama_context * ctx_dft = params.speculative.draft.ctx_dft;
    GGML_ASSERT(ctx_dft);

    const llama_seq_id s_main = 0; // canonical continuation sequence (speculator uses this too)
    const llama_seq_id s_aux  = dual ? 1 : 0; // scratch sequence for path B

    struct common_speculative * spec = common_speculative_init(params.speculative, 2);
    GGML_ASSERT(spec);
    common_sampler_ptr smpl(common_sampler_init(model_tgt, params.sampling));

    std::vector<llama_token> inp = common_tokenize(ctx_tgt, params.prompt, true, true);

    {
        llama_batch bp = llama_batch_init(inp.size(), 0, 1);
        for (size_t i = 0; i + 1 < inp.size(); ++i) {
            common_batch_add(bp, inp[i], (llama_pos) i, { s_main }, false);
        }
        llama_decode(ctx_tgt, bp);
        if (!common_speculative_process(spec, bp)) { fprintf(stderr, "prefill feed failed\n"); return 1; }
        llama_batch_free(bp);
    }
    llama_token id_last = inp.back();
    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    int n_past = (int) inp.size() - 1;

    common_speculative_begin(spec, s_main, prompt_tgt);

    int n_predict = 0, n_drafted = 0, n_accept = 0, n_cycles = 0, n_dual_wins = 0;
    bool has_eos = false;
    llama_batch batch_tgt = llama_batch_init(std::max<llama_pos>(64, 2 * (llama_pos)(params.speculative.draft.n_max + 2)), 0, 2);

    llama_tokens draft_a, draft_b;
    std::vector<common_speculative_token_dist> dists;

    const auto t0 = ggml_time_us();

    while (!has_eos && n_predict < n_predict_cap) {
        ++n_cycles;

        auto reset_dft = [&]() {
            llama_memory_seq_rm(llama_get_memory(ctx_dft), s_main, n_past, -1);
        };
        auto draft_once = [&](llama_tokens & out, uint32_t seed) {
            out.clear();
            dists.clear();
            int room = (int) llama_n_ctx(ctx_tgt) / (dual ? 2 : 1) - n_past - 2;
            room = std::min(room, n_predict_cap - n_predict - 1);
            room = std::max(room, 0);
            common_speculative_get_draft_params(spec, s_main) = {
                /* .drafting    = */ true,
                /* .n_max       = */ room,
                /* .n_past      = */ n_past,
                /* .id_last     = */ id_last,
                /* .prompt      = */ &prompt_tgt,
                /* .result      = */ &out,
                /* .dists       = */ &dists,
                /* .temperature = */ params.sampling.temp,
                /* .seed        = */ seed,
            };
            common_speculative_draft(spec);
        };

        draft_once(draft_a, 1000u + (uint32_t) n_cycles);
        reset_dft();
        if (dual) {
            draft_once(draft_b, 2000u + (uint32_t) n_cycles);
            reset_dft();
        }

        // no usable draft: plain decode of id_last
        if (draft_a.empty() && (!dual || draft_b.empty())) {
            common_batch_clear(batch_tgt);
            common_batch_add(batch_tgt, id_last, n_past++, { s_main }, true);
            if (llama_decode(ctx_tgt, batch_tgt) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
            if (!common_speculative_process(spec, batch_tgt)) { fprintf(stderr, "process failed\n"); return 1; }
            common_sampler_sample(smpl.get(), ctx_tgt, -1, true);
            id_last = common_sampler_last(smpl.get());
            prompt_tgt.push_back(id_last);
            ++n_predict;
            if (llama_vocab_is_eog(vocab, id_last)) has_eos = true;
            continue;
        }

        // ---- one batched verification pass over both paths ----
        common_batch_clear(batch_tgt);
        // anchor + drafts for path A on s_main
        common_batch_add(batch_tgt, id_last, n_past, { s_main }, true);                       // row 0
        for (size_t i = 0; i < draft_a.size(); ++i) {
            common_batch_add(batch_tgt, draft_a[i], n_past + 1 + (llama_pos) i, { s_main }, true);
        }
        const int row_anchor_b = draft_a.size() + 1;                                          // B anchor row
        if (dual && !draft_b.empty()) {
            common_batch_add(batch_tgt, id_last, n_past, { s_aux }, true);                    // row row_anchor_b
            for (size_t i = 0; i < draft_b.size(); ++i) {
                common_batch_add(batch_tgt, draft_b[i], n_past + 1 + (llama_pos) i, { s_aux }, true);
            }
        }

        if (llama_decode(ctx_tgt, batch_tgt) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
        if (!common_speculative_process(spec, batch_tgt)) { fprintf(stderr, "process failed\n"); return 1; }

        // ---- greedy lossless acceptance per path ----
        auto accept_path = [&](int row_anchor, const llama_tokens & draft) {
            int acc = 0;
            while (acc < (int) draft.size() &&
                   argmax_row(ctx_tgt, row_anchor + acc, vocab) == draft[acc]) {
                ++acc;
            }
            llama_token bonus = argmax_row(ctx_tgt, row_anchor + acc, vocab);
            return std::make_pair(acc, bonus);
        };
        auto ra = accept_path(0, draft_a);
        int acc_a = ra.first; llama_token tok_a = ra.second;

        int acc_b = -1; llama_token tok_b = 0;
        if (dual && !draft_b.empty()) {
            auto rb = accept_path(row_anchor_b, draft_b);
            acc_b = rb.first; tok_b = rb.second;
        }

        // ---- pick winner; canonical continuation returns to s_main ----
        bool b_won = dual && !draft_b.empty() && acc_b > acc_a;
        const llama_tokens & win_draft = b_won ? draft_b : draft_a;
        const int   win_acc = b_won ? acc_b : acc_a;
        llama_token win_tok = b_won ? tok_b : tok_a;
        if (b_won) { ++n_dual_wins; }

        if (b_won) {
            llama_memory_t mem = llama_get_memory(ctx_tgt);
            llama_memory_seq_rm(mem, s_main, n_past, -1);
            llama_memory_seq_cp(mem, s_aux, s_main, n_past, n_past + win_acc);
            llama_memory_seq_rm(mem, s_aux, n_past, -1);
        } else {
            llama_memory_seq_rm(llama_get_memory(ctx_tgt), s_main, n_past + win_acc + 1, -1);
            if (dual) {
                llama_memory_seq_rm(llama_get_memory(ctx_tgt), s_aux, n_past, -1);
            }
        }
        llama_memory_seq_rm(llama_get_memory(ctx_dft), s_main, n_past, -1);

        n_drafted += (int)(draft_a.size() + (dual && !draft_b.empty() ? draft_b.size() : 0));
        n_accept  += win_acc;
        n_predict += win_acc + 1;
        n_past    += win_acc + 1;

        for (int i = 0; i < win_acc; ++i) { prompt_tgt.push_back(id_last); }
        prompt_tgt.push_back(win_tok);
        id_last = win_tok;

        draft_a.clear(); draft_b.clear(); dists.clear();
        if (llama_vocab_is_eog(vocab, id_last)) has_eos = true;
    }

    const auto t1 = ggml_time_us();
    const double sec = (t1 - t0) / 1e6;
    printf("RESULT {\"mode\":\"%s\",\"fixture\":%d,\"n_predict\":%d,\"sec\":%.3f,"
           "\"tps\":%.2f,\"accept\":%.3f,\"cycles\":%d,\"tau\":%.2f,\"dual_wins\":%d}\n",
           dual ? "dual" : "single", fixture, n_predict, sec,
           sec > 0 ? n_predict / sec : 0.0,
           n_drafted > 0 ? (double) n_accept / n_drafted : 0.0,
           n_cycles, n_cycles > 0 ? n_predict / (double) n_cycles : 0.0, n_dual_wins);

    common_speculative_free(spec);
    llama_batch_free(batch_tgt);
    llama_backend_free();
    return 0;
}
