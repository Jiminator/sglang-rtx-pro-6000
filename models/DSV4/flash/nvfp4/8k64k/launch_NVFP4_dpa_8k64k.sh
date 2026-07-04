#!/bin/bash
# DeepSeek-V4-Flash NVFP4 + DP-attention — 8K/64K on 1x g4 (8x RTX PRO 6000, SM120).
# nvidia/DeepSeek-V4-Flash-NVFP4, latest-main dev-cu13, bench_serving rrr=1.0, ISL 8192 / OSL 65536.
# STEADY-STATE DECODE PLATEAU @ CONC=512 (64 running-req/rank): see TUNING_REPORT.md (~1900 tok/s class,
# ~3.4x the prior offline anchor 551.9). MXFP4-Marlin ties NVFP4; FP8 ~4% behind.
#
# ⚠️ Same TWO SM120 env vars as 1k/8k (decode+prefill; latest-main dev-cu13 is doubly-broken without them).
# 8K/64K MEASUREMENT NOTE: at inf request-rate, high concurrency floods the 8192-token prefill backlog and
# the server never reaches sustained decode (CONC=1024 = prefill-swamped, no plateau). The clean steady-state
# ceiling is ~64 running-req/rank (CONC=512); above ~70/rank prefill saturates. Measure at CONC=512.
set -euo pipefail
NAME=${NAME:-dsv4_nvfp4_dpa_8k64k}
IMAGE=${IMAGE:-lmsysorg/sglang:dev-cu13}
sudo docker run -d --name "$NAME" --gpus all --shm-size 32g --network host --ipc host \
  -v ~/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  -e NCCL_P2P_LEVEL=SYS -e NCCL_MIN_NCHANNELS=8 -e NCCL_ALLOC_P2P_NET_LL_BUFFERS=1 \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_IB_DISABLE=1 \
  -e NCCL_SOCKET_IFNAME=enp128s4,ens3 -e GLOO_SOCKET_IFNAME=ens3 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e SGLANG_SM120_FLASHMLA_BACKEND=triton \
  -e SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=0 \
  "$IMAGE" \
  python3 -m sglang.launch_server \
    --model-path nvidia/DeepSeek-V4-Flash-NVFP4 --served-model-name nvidia/DeepSeek-V4-Flash-NVFP4 \
    --trust-remote-code --tp 8 --dp-size 8 --enable-dp-attention \
    --moe-a2a-backend none --moe-runner-backend flashinfer_cutlass \
    --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85 \
    --cuda-graph-max-bs 64 --context-length 73728 --max-running-requests 2048 \
    --host 0.0.0.0 --port 8000
# Bench (read steady-state decode plateau from server logs; requests don't complete in-window at OSL 65536):
#   python3 -m sglang.bench_serving --backend sglang --model nvidia/DeepSeek-V4-Flash-NVFP4 \
#     --dataset-name random --random-input-len 8192 --random-output-len 65536 --random-range-ratio 1.0 \
#     --max-concurrency 512 --num-prompts 1024
# FP8 alt: --model-path sgl-project/DeepSeek-V4-Flash-FP8 --moe-runner-backend triton
# MXFP4 alt: --model-path deepseek-ai/DeepSeek-V4-Flash --moe-runner-backend marlin
