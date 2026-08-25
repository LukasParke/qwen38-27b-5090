#!/usr/bin/env python3
"""Run the trained checkpoint through specforge's own _forward_draft_blocks on
training data and measure teacher-forced top-1. Splits 'checkpoint save lost
the training' vs 'llama.cpp runtime diverges'."""
import torch, glob, json, sys
sys.path.insert(0, "/home/luke/github/specforge-src")
sys.path.insert(0, "/home/luke/llm/llama.cpp-src")

from safetensors import safe_open
from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel as DFlashFamilyModel

TARGET = "/home/luke/models/Qwen3.8-27B-FP8"
CKPT_DIR = "/home/luke/github/afap-qwen3.8/draft-training/outputs/qwen38-27b-dspark-b16"
CFG = json.load(open("/home/luke/github/afap-qwen3.8/draft-training/qwen3.8-27b-dspark-b16.json"))
MASK = CFG["dflash_config"]["mask_token_id"]; BS = CFG["block_size"]

# --- load trained draft from checkpoint ---
st = torch.load(f"{CKPT_DIR}/qwen38-27b-dspark-b16-step500/training_state.pt",
                map_location="cpu", weights_only=False)
dsd = st["draft_state_dict"]
m = DSparkDraftModel.from_pretrained(
        "/home/luke/github/afap-qwen3.8/draft-training/dspark-hf-test",
        config="/home/luke/github/afap-qwen3.8/draft-training/dspark-hf-test/config.json",
        dtype=torch.float32).eval()
missing, unexpected = m.load_state_dict(dsd, strict=False)
print("load: missing", len(missing), "unexpected", len(unexpected))

# frozen embed + lm_head from target
embW = headW = None
import os
for f in sorted(os.listdir(TARGET)):
    if not f.endswith(".safetensors"): continue
    with safe_open(os.path.join(TARGET, f), framework="pt", device="cpu") as sf:
        for k in sf.keys():
            if embW is None and k.endswith("embed_tokens.weight"): embW = sf.get_tensor(k)
            if headW is None and k == "lm_head.weight": headW = sf.get_tensor(k)
    if embW is not None and headW is not None: break
import torch.nn as nn
emb = nn.Embedding(embW.shape[0], embW.shape[1]); emb.weight.data = embW.float()
head = nn.Linear(headW.shape[1], headW.shape[0], bias=False); head.weight.data = headW.float()

# --- shim exposing exactly what _forward_draft_blocks uses ---
class _Shim(DFlashFamilyModel):
    def __init__(self):
        torch.nn.Module.__init__(self)
shim = _Shim()
shim.draft_model = m
shim.loss_type = "dflash"
shim.dspark_l1_loss_alpha = 0.0
shim.dspark_confidence_head_alpha = 0.0
shim.embed_tokens = emb
shim.block_size = BS
shim.num_anchors = 1
shim.mask_token_id = MASK
shim.sliding_window = None
shim.lm_head = head
shim.loss_type = "dflash"
shim.dspark_l1_loss_alpha = 0.0
shim.dspark_confidence_head_alpha = 0.0           # a few blocks per record is plenty
shim.attention_backend = "sdpa"

# bind the unbound method
fwd = DFlashFamilyModel._forward_draft_blocks.__get__(shim)

files = sorted(glob.glob("/home/luke/github/afap-qwen3.8/draft-training/cache/hidden_states/qwen38-27b-dspark-b16/rows_0-2000/*.ckpt"))
tot_hit, tot_sup = 0, 0
per = [0]*BS; cnt = [0]*BS
tested = 0
for fp in files[:10]:
    d = torch.load(fp, map_location="cpu", weights_only=False)
    ids = d["input_ids"].unsqueeze(0); lm = d["loss_mask"].float().unsqueeze(0)
    hs = d["hidden_states"].float()
    if hs.shape[1] < ids.shape[1]:  # truncate ids/lm to feature length
        ids = ids[:, :hs.shape[1]]; lm = lm[:, :hs.shape[1]]
    with torch.no_grad():
        anchor_positions, block_keep_mask, out_hidden = fwd(ids, hs, lm)
    POS = out_hidden.shape[1]
    logits = head(out_hidden.reshape(1, POS, -1))[0]
    pred = logits.argmax(-1)
    starts = anchor_positions[0].tolist()
    keep = block_keep_mask[0].tolist() if block_keep_mask is not None else [True]*len(starts)
    for b, st_ in enumerate(starts):
        if not keep[b]: continue
        if st_ + BS > ids.shape[1]: continue
        want = ids[0, st_:st_+BS]
        got = pred[b*BS:(b+1)*BS]
        for k in range(BS):
            hit = int(got[k] == want[k])
            tot_hit += hit; tot_sup += 1; per[k] += hit; cnt[k] += 1
    tested += 1
    print(f"record {tested}: anchors={len(starts)} running_acc={100*tot_hit/max(1,tot_sup):.1f}%", flush=True)
print(f"FINAL through specforge machinery, checkpoint weights: {tot_hit}/{tot_sup} = {100*tot_hit/max(1,tot_sup):.1f}%")
print("acc by position:", [f"{100*per[k]/max(1,cnt[k]):.0f}%" for k in range(BS)])
