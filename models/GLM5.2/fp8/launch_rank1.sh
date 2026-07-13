#!/bin/bash
# GLM-5.2-FP8 (zai-org, full-FP8 704 GB) — 2-node PP=2 + DP-attention, RANK 1.
# Identical to launch_rank0.sh except --node-rank 1. Run on the OTHER node within ~60s of rank0.
# DIST_ADDR must be RANK 0's private IP (not this node's). NICs here are ens3 + enp128s4. See README.md.
set -euo pipefail
DIST_ADDR="${DIST_ADDR:?set DIST_ADDR to rank0's private IP, e.g. 10.0.17.5}"
NAME=${NAME:-glm52_fp8_pp2_rank1}
sudo docker run -d --name "$NAME" --gpus all --shm-size 32g --network host --ipc host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e SGLANG_PP_LAYER_PARTITION=38,40 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e NCCL_SOCKET_IFNAME=enp128s4,ens3 -e GLOO_SOCKET_IFNAME=ens3 \
  -e NCCL_P2P_LEVEL=SYS -e NCCL_MIN_NCHANNELS=8 -e NCCL_ALLOC_P2P_NET_LL_BUFFERS=1 \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_IB_DISABLE=1 \
  lmsysorg/sglang:dev-cu13 \
  python3 -m sglang.launch_server \
    --model zai-org/GLM-5.2-FP8 --quantization fp8 --trust-remote-code \
    --tensor-parallel-size 8 --pipeline-parallel-size 2 --nnodes 2 --node-rank 1 \
    --dist-init-addr "${DIST_ADDR}:5000" \
    --dp-size 8 --enable-dp-attention \
    --attention-backend flashinfer \
    --moe-runner-backend triton --moe-a2a-backend none --ep-size 1 \
    --kv-cache-dtype fp8_e4m3 \
    --disable-shared-experts-fusion --disable-radix-cache \
    --mem-fraction-static 0.80 --page-size 64 \
    --host 0.0.0.0 --port 8000
