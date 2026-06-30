#!/usr/bin/env bash
# GLM-5.2-NVFP4 — BEST CONFIG: EAGLE 3-step speculative decoding (throughput SOTA + latency winner)
# Single node, 8x RTX PRO 6000 Blackwell (SM120). ~300 tok/s/GPU sustained, gsm8k 0.94, accept-len ~3.9.
# Requires the glm-opt branch build (image sglang-glmopt:tf512). See ../README.md.
set -euo pipefail

IMAGE="${IMAGE:-sglang-glmopt:tf512}"
# Pin the complete snapshot — refs/main drifted to a weightless partial.
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
  --name glm52-eagle \
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
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --speculative-moe-runner-backend flashinfer_cutlass \
    --speculative-moe-a2a-backend none \
    --host 0.0.0.0 --port 8000

# Notes:
# - mfs 0.97 is the one-shot/max-throughput setting. For CONTINUOUS serving above the pool
#   ceiling, use --mem-fraction-static 0.92 --chunked-prefill-size 1024 to avoid the
#   prefill-activation OOM at over-subscription (over-subscription then queues gracefully).
# - --speculative-eagle-topk MUST be 1 (flashinfer-MLA supports topk=1 only for spec on SM120).
# - --speculative-moe-runner-backend/a2a pinned to flashinfer_cutlass/none: avoids the dead
#   deep_gemm/deepep auto-route for modelopt_fp4 + EAGLE on SM120.
