#!/bin/bash
# Anchor reproduction: literal copy of _raw/glm-nvfp4-prefill/2node-isl16384-osl1024-2026-05-09/launch_node1.sh
# Target: 4,574.19 total tok/s (output 262.54)
set -euo pipefail
cd /sgl-workspace/sglang 2>/dev/null || cd /
exec python3 -m sglang.launch_server \
  --model-path lukealonso/GLM-5.1-NVFP4 \
  --served-model-name GLM-5.1 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --dp-size 8 \
  --enable-dp-attention \
  --nnodes 2 \
  --node-rank 0 \
  --dist-init-addr 10.0.1.2:5000 \
  --trust-remote-code \
  --quantization modelopt_fp4 \
  --kv-cache-dtype bfloat16 \
  --fp4-gemm-backend flashinfer_cutlass \
  --attention-backend flashinfer \
  --moe-runner-backend flashinfer_cutlass \
  --disable-shared-experts-fusion \
  --mem-fraction-static 0.90 \
  --max-running-requests 768 \
  --chunked-prefill-size 8192 \
  --enable-dynamic-chunking \
  --cuda-graph-max-bs 128 \
  --page-size 64 \
  --enable-fused-moe-sum-all-reduce \
  --enable-dp-lm-head \
  --tokenizer-worker-num 16 \
  --enable-flashinfer-allreduce-fusion \
  --disable-custom-all-reduce \
  --disable-radix-cache \
  --host 0.0.0.0 \
  --port 30000
