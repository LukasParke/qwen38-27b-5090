#!/bin/bash
# Build upstream-master-20260824 branch (0d0efe92f) into isolated prefix.
set -u
export TMPDIR=/home/luke/llm/tmp   # /tmp is a full 31G tmpfs shared with other tools
mkdir -p "$TMPDIR"
cd ~/llm/llama.cpp-src
export PATH=/opt/cuda/bin:$PATH
echo "=== CONFIGURE $(date -Is) ==="
cmake -B build-cascade -S . \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DGGML_CUDA_F16=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/home/luke/llm/llamacpp-afap-cascade \
  -DLLAMA_BUILD_NUMBER=572 || exit 1
echo "=== BUILD $(date -Is) ==="
cmake --build build-cascade -j32 || exit 1
echo "=== INSTALL $(date -Is) ==="
cmake --install build-cascade || exit 1
LD_LIBRARY_PATH=/home/luke/llm/llamacpp-afap-cascade/lib /home/luke/llm/llamacpp-upstream-20260824/bin/llama-server --version
echo "=== ALL DONE $(date -Is) ==="
