#!/bin/bash
# DeepSeek-V4-Flash NVFP4 + DP-attention — 1K/8K WINNER on 1x g4 (8x RTX PRO 6000, SM120).
# nvidia/DeepSeek-V4-Flash-NVFP4 (hybrid FP8+NVFP4 MoE), latest-main dev-cu13, bench_serving rrr=1.0.
# STEADY-STATE DECODE PLATEAU: ~3753 output tok/s (~469/GPU) @ 256 running-req/rank, gsm8k 0.960.
# See TUNING_REPORT.md. (MXFP4-Marlin 3717 and FP8 3694 tie within ~1.6%; NVFP4 is the top + smallest ckpt.)
#
# ⚠️ LATEST-MAIN SM120 REQUIRES TWO ENV VARS (no source patch). Stock dev-cu13 is doubly-broken for DSV4:
#   SGLANG_SM120_FLASHMLA_BACKEND=triton   -> decode: flashinfer 0.6.12 dropped `_sparse_mla_sm120`;
#                                             triton routes to the in-tree SM120 sparse-MLA decode kernel.
#   SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=0   -> prefill: the AOT `sparse_prefill_fwd` is SM90a/SM100f-only;
#                                             =0 uses dense SM120 prefill (chunk 1024/forward < 11673 gate).
# SM120 MoE-runner is per-checkpoint: NVFP4 -> flashinfer_cutlass (ModelOptNvFp4FusedMoEMethod).
# DSV4 hard-locks KV to fp8_e4m3 (bf16 rejected by deepseek_v4_hook). #28231 (Marlin) is in this image.
set -euo pipefail
NAME=${NAME:-dsv4_nvfp4_dpa_1k8k}
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
    --cuda-graph-max-bs 384 --context-length 9472 --max-running-requests 2048 \
    --host 0.0.0.0 --port 8000
# Bench (loadgen, read steady-state decode plateau from server logs):
#   python3 -m sglang.bench_serving --backend sglang --model nvidia/DeepSeek-V4-Flash-NVFP4 \
#     --dataset-name random --random-input-len 1024 --random-output-len 8192 --random-range-ratio 1.0 \
#     --max-concurrency 2048 --num-prompts 4096
# Ceiling is 256/rank (= max-running-requests 2048 / dp-size 8); decode is BW-bound, past the knee.
# FP8 alt: --model-path sgl-project/DeepSeek-V4-Flash-FP8 --moe-runner-backend triton
# MXFP4 alt: --model-path deepseek-ai/DeepSeek-V4-Flash --moe-runner-backend marlin  (#28231)
