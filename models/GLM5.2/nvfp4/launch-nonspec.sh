#!/usr/bin/env bash
# GLM-5.2-NVFP4 — non-spec fp8 config (max-batch one-shot SOTA: 194.6 tok/s/GPU @ b191, gsm8k 0.97).
# Simpler than the EAGLE config (no speculative decode); the three stacking memory levers only.
# Single node, 8x RTX PRO 6000 Blackwell (SM120). Requires the glm-opt branch build.
set -euo pipefail

IMAGE="${IMAGE:-sglang-glmopt:tf512}"
MODEL="${MODEL:-/root/.cache/huggingface/hub/models--nvidia--GLM-5.2-NVFP4/snapshots/b0b2b68d4be5ee00e95ae013ea0949fe5c0b5a56}"

sudo docker run --rm --gpus all --network host --shm-size 32g \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  --entrypoint python3 \
  -e HF_HUB_OFFLINE=1 \
  -e SGLANG_DISABLE_DSA_INDEXER_FUSION=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e NCCL_P2P_LEVEL=SYS -e NCCL_MIN_NCHANNELS=8 -e NCCL_ALLOC_P2P_NET_LL_BUFFERS=1 \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_IB_DISABLE=1 \
  -e NCCL_SOCKET_IFNAME=ens3 -e GLOO_SOCKET_IFNAME=ens3 \
  --name glm52-nonspec \
  "$IMAGE" -m sglang.launch_server \
    --model-path "$MODEL" \
    --tp-size 8 --enable-dp-attention --dp-size 8 \
    --attention-backend flashinfer \
    --kv-cache-dtype fp8_e4m3 \
    --moe-a2a-backend none --ep-size 1 --moe-runner-backend flashinfer_cutlass \
    --disable-shared-experts-fusion \
    --mem-fraction-static 0.97 \
    --chunked-prefill-size 2048 \
    --cuda-graph-bs "8 16 24 32" \
    --max-running-requests 1024 \
    --trust-remote-code \
    --host 0.0.0.0 --port 8000

# Stock-branch fallback (no glm-opt): fp8 KV is unavailable; use --kv-cache-dtype bfloat16
# on the stock lmsysorg/sglang:nightly-dev-cu13 image (transformers>=5.10) for the 147 tok/s/GPU
# bf16-dense baseline (mfs 0.94, --cuda-graph-max-bs = batch).
