#!/bin/bash
# Rebuild llama.cpp fork (head 5ecbe1ac1 + GGML_CUDA_MMVQ_MAX_BATCH env knob)
# into a separate prefix so production cuda-dflash2 stays untouched.
set -u
cd ~/llm/llama.cpp-src
export PATH=/opt/cuda/bin:$PATH
echo "=== CONFIGURE $(date -Is) ==="
rm -rf build-gemm
cmake -B build-gemm -S . \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DGGML_CUDA_F16=ON \
  -DCMAKE_CUDA_FLAGS="-O3 -Xptxas=-v" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/home/luke/llm/cuda-dflash2-gemm \
  -DLLAMA_BUILD_NUMBER=570 || exit 1
echo "=== BUILD $(date -Is) ==="
cmake --build build-gemm -j32 || exit 1
echo "=== INSTALL $(date -Is) ==="
cmake --install build-gemm || exit 1
LD_LIBRARY_PATH=/home/luke/llm/cuda-dflash2-gemm/lib /home/luke/llm/cuda-dflash2-gemm/bin/llama-server --version
echo "=== ALL DONE $(date -Is) ==="
