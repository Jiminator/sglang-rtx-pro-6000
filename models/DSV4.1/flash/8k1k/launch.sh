#!/bin/bash
# DeepSeek-V4.1-Flash, official checkpoint, 8x RTX PRO 6000 (SM120), one node.
# Upstream recipe (shivajid/sglang-rtx-pro-6000, models/DeepSeekv4.1-Flash/
# sglang-dsv41-flash-1node.yaml) plus four changes:
#   NCCL_MIN_NCHANNELS=16
#   --cuda-graph-max-bs-decode 128                  (upstream: 64)
#   --fp8-gemm-backend flashinfer_cutlass
#   --chunked-prefill-size 16384 --max-prefill-tokens 32768
set -euo pipefail
IMAGE=${IMAGE:-lmsysorg/sglang:dev-dsv41}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR to the DeepSeek-V4.1-Flash checkpoint}
docker rm -f sglang-dsv41 >/dev/null 2>&1 || true
docker run -d --name sglang-dsv41 \
  --gpus all --network host --ipc host --privileged \
  --cap-add IPC_LOCK --cap-add SYS_PTRACE \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$MODEL_DIR":/models/DeepSeek-V4.1-Flash \
  -e SGLANG_SM120_FLASHMLA_BACKEND=triton \
  -e SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=0 \
  -e SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 \
  -e NCCL_MIN_NCHANNELS=16 \
  -e NCCL_P2P_DISABLE=0 \
  -e NCCL_IB_DISABLE=1 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e NCCL_DEBUG=WARN \
  -e SAFETENSORS_FAST_GPU=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e SGLANG_ENABLE_DEEP_GEMM=0 \
  -e SGLANG_ENABLE_JIT_DEEPGEMM=0 \
  -e SGLANG_DISABLE_DEEP_GEMM=1 \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e OMP_NUM_THREADS=24 \
  -e SGLANG_SET_CPU_AFFINITY=1 \
  "$IMAGE" \
  python3 -m sglang.launch_server \
    --model /models/DeepSeek-V4.1-Flash \
    --trust-remote-code \
    --tp 8 \
    --ep-size 8 \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.8 \
    --enable-deepseek-v4-fp4-indexer \
    --enable-decoder-swa-bounded-replay \
    --cuda-graph-max-bs-decode 128 \
    --fp8-gemm-backend flashinfer_cutlass \
    --chunked-prefill-size 16384 \
    --max-prefill-tokens 32768 \
    --reasoning-parser auto \
    --tool-call-parser auto \
    --host 0.0.0.0 \
    --port 30000
