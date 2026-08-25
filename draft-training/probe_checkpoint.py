#!/usr/bin/env python3
"""Probe newest checkpoint via shim machinery; print generalization curve."""
import torch, glob, json, os, sys
sys.path.insert(0, "/home/luke/github/specforge-src")
CFG = json.load(open("/home/luke/github/afap-qwen3.8/draft-training/qwen3.8-27b-dspark-b16.json"))
MASK = CFG["dflash_config"]["mask_token_id"]; BS = CFG["block_size"]
from safetensors import safe_open
import torch.nn as nn
from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel as Fam

TARGET="/home/luke/models/Qwen3.8-27B-FP8"
embW=headW=None
for f in sorted(os.listdir(TARGET)):
    if not f.endswith(".safetensors"): continue
    with safe_open(os.path.join(TARGET,f), framework="pt", device="cpu") as sf:
        for k in sf.keys():
            if embW is None and k.endswith("embed_tokens.weight"): embW=sf.get_tensor(k)
            if headW is None and k=="lm_head.weight": headW=sf.get_tensor(k)

files=sorted(glob.glob("/home/luke/github/afap-qwen3.8/draft-training/cache/hidden_states/qwen38-27b-dspark-b16/rows_0-2000/*.ckpt"))[:10]
CACHE=[(torch.load(f,map_location="cpu",weights_only=False)) for f in files]

def probe(weights):
    m=DSparkDraftModel.from_pretrained(
        "/home/luke/github/afap-qwen3.8/draft-training/dspark-hf-test",
        config="/home/luke/github/afap-qwen3.8/draft-training/dspark-hf-test/config.json",
        dtype=torch.float32).eval()
    res=m.load_state_dict(weights, strict=False)
    emb=nn.Embedding(embW.shape[0],embW.shape[1]); emb.weight.data=embW.float()
    head=nn.Linear(headW.shape[1],headW.shape[0],bias=False); head.weight.data=headW.float()
    class Shim(Fam):
        def __init__(self): torch.nn.Module.__init__(self)
    shim=Shim()
    shim.draft_model=m; shim.embed_tokens=emb; shim.block_size=BS; shim.num_anchors=1
    shim.attention_backend="sdpa"; shim.mask_token_id=MASK; shim.sliding_window=None
    tot,sup=0,0
    for d in CACHE:
        ids=d["input_ids"].unsqueeze(0); lm=d["loss_mask"].float().unsqueeze(0)
        hs=d["hidden_states"].float(); T=hs.shape[1]
        ids=ids[:,:T]; lm=lm[:,:T]
        with torch.no_grad():
            ap,keep,out=Fam._forward_draft_blocks(shim, ids, hs, lm)
        logits=head(out[0]); pred=logits.argmax(-1); start=int(ap[0,0])
        L=min(BS, T-start); want=ids[0,start:start+L]
        for k in range(L): tot+=int(pred[k]==want[k]); sup+=1
    return 100*tot/max(1,sup)

if __name__ == "__main__":
    ck = sys.argv[1]
    st = torch.load(os.path.join(ck,"training_state.pt"), map_location="cpu", weights_only=False)
    if "live_model_params" in st and st["live_model_params"]:
        w = {("model."+k if not k.startswith("model.") else k): v for k,v in st["live_model_params"].items()}
        # names in ckpt are raw param names already matching model keys order;
        # load by position instead: build ordered list of trainable params
        tmp=DSparkDraftModel.from_pretrained(
            "/home/luke/github/afap-qwen3.8/draft-training/dspark-hf-test",
            config="/home/luke/github/afap-qwen3.8/draft-training/dspark-hf-test/config.json",
            dtype=torch.float32)
        names=[n for n,p in tmp.named_parameters() if p.requires_grad]
        sd={}
        for i,n in enumerate(names):
            key=f"live_{i}"
            if key in st["live_model_params"]: sd[n]=st["live_model_params"][key]
        del tmp
        print(probe(sd), "LIVE", ck.split("/")[-1], flush=True)
    else:
        w={k:v for k,v in st.get("draft_state_dict",{}).items()}
        print(probe(w), "FILTERED", ck.split("/")[-1], flush=True)
