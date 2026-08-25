#!/usr/bin/env bash
# Export trained DSpark drafter -> HF -> GGUF -> smoke test.
# Usage: bash export_drafter.sh <checkpoint_dir> [outtype]
set -euo pipefail
CKPT="${1:?checkpoint dir e.g. .../outputs/qwen38-27b-dspark-b16/step_6000}"
OUTTYPE="${2:-f16}"
SF=/home/luke/github/specforge-src
WORK=/home/luke/github/afap-qwen3.8/draft-training
DRAFTHF=$WORK/dspark-qwen38-27b-b16-hf
GGUF=$WORK/Qwen3.8-27B-DSpark-B16-$OUTTYPE.gguf
TARGET=/home/luke/models/Qwen3.8-27B-FP8

export TMPDIR=/home/luke/tmpdir
export CUDA_HOME=/opt/cuda
export PATH="/opt/cuda/bin:$PATH"

echo "== 1) export HF"
cd "$SF"
.venv/bin/specforge export --to hf \
  --checkpoint "$CKPT" \
  --draft-config "$WORK/qwen3.8-27b-dspark-b16.json" \
  --embedding-source "$TARGET" \
  --output-dir "$DRAFTHF"

echo "== 2) convert GGUF"
cd /home/luke/llm/llama.cpp-src
python3 convert_hf_to_gguf.py "$DRAFTHF" \
  --target-model-dir "$TARGET" \
  --outtype "$OUTTYPE" \
  --outfile "$GGUF"

echo "== 3) smoke: serve + gen"
LD_LIBRARY_PATH=/home/luke/llm/cuda-dflash2-gemm/lib GGML_CUDA_MMVQ_MAX_BATCH=1 \
/home/luke/llm/cuda-dflash2-gemm/bin/llama-server \
  --host 127.0.0.1 --port 5941 -fa on -ngl 999 --no-mmap --jinja \
  -ub 1024 -b 1024 -c 4096 \
  -m /home/luke/models/Qwen3.8-27B-GGUF-IQ4XS/Qwen3.8-27B-UD-IQ4_XS.gguf \
  --spec-type draft-dspark --model-draft "$GGUF" --spec-draft-n-max 15 -bs &
SRV=$!
sleep 25
for i in $(seq 1 60); do curl -s -m 2 http://127.0.0.1:5941/health | grep -q ok && break; sleep 3; done
python3 - <<'EOF'
import json, urllib.request
body={"model":"x","messages":[{"role":"user","content":"Write a detailed essay about GPU computing history."}],
      "max_tokens":384,"temperature":1.0,"top_k":20,"top_p":0.95,
      "chat_template_kwargs":{"enable_thinking":False}}
req=urllib.request.Request("http://127.0.0.1:5941/v1/chat/completions",
    data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
d=json.loads(urllib.request.urlopen(req,timeout=300).read())
t=d["timings"]; r=(t.get("draft_n_accepted") or 0)/max(1,t.get("draft_n") or 1)
print(f"B16-DSPARK: {t['predicted_per_second']:.1f} t/s accept={r:.3f}")
EOF
kill $SRV 2>/dev/null || true
echo "DONE: $GGUF"
