#!/bin/bash
# GLM-5.2-FP8 (zai-org, full-FP8 704 GB) — 2-node PP=2 + DP-attention, RANK 0 (dist-init master).
# Measured on stock lmsysorg/sglang:dev-cu13 (v0.5.15): 1K/8K ~137 tok/s/GPU, 8K/64K ~24.85 tok/s/GPU,
# gsm8k 0.92. See README.md. Run launch_rank1.sh on the OTHER node within ~60s.
#
# THE UNLOCK (no source patch): SGLANG_PP_LAYER_PARTITION=38,40 moves the PP=2 stage boundary off the
# DSA skip-topk layer 39 onto full-topk layer 38 (sglang issue #28659 / PR #28785). PP=2 also sidesteps
# the TP=16 cross-node symm-mem warmup deadlock (PP uses point-to-point sends, not the multimem collective).
#
# SM120 backends (load-bearing): flashinfer attn (DSA sparse decode is SM100-only on this checkpoint -> dense
# MLA); triton MoE (flashinfer_cutlass is FP4-only, crashes Fp8MoEMethod); moe-a2a none (DeepEP needs RDMA/NVL);
# kv fp8_e4m3; mfs 0.80 (0.85 OOMs in cuda-graph capture on the heavier stage).
#
# Set DIST_ADDR to THIS node's private IP (the address rank1 dials). NICs here are ens3 + enp128s4.
set -euo pipefail
DIST_ADDR="${DIST_ADDR:?set DIST_ADDR to rank0's private IP, e.g. 10.0.17.5}"
NAME=${NAME:-glm52_fp8_pp2_rank0}
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
    --tensor-parallel-size 8 --pipeline-parallel-size 2 --nnodes 2 --node-rank 0 \
    --dist-init-addr "${DIST_ADDR}:5000" \
    --dp-size 8 --enable-dp-attention \
    --attention-backend flashinfer \
    --moe-runner-backend triton --moe-a2a-backend none --ep-size 1 \
    --kv-cache-dtype fp8_e4m3 \
    --disable-shared-experts-fusion --disable-radix-cache \
    --mem-fraction-static 0.80 --page-size 64 \
    --host 0.0.0.0 --port 8000
# Bench (from rank0): python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 8000 \
#   --model zai-org/GLM-5.2-FP8 --dataset-name random --random-input-len 1024 --random-output-len 8192 \
#   --random-range-ratio 1.0 --request-rate inf --max-concurrency 512 --num-prompts 1024
