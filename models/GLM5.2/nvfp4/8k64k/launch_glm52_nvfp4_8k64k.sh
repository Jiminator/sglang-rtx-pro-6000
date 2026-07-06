#!/bin/bash
# GLM-5.2-NVFP4 + DP-attention — 8K/64K on 1x g4 (8x RTX PRO 6000, SM120).
# nvidia/GLM-5.2-NVFP4 (GlmMoeDsa DSA arch), STOCK latest-main dev-cu13, bench_serving rrr=1.0, ISL 8192 / OSL 65536.
# STEADY-STATE DECODE PLATEAU @ CONC=128 (16 running-req/rank): ~1081 output tok/s (~135/GPU). See TUNING_REPORT.md.
# +10% vs bf16 (~123/GPU) via the concurrency-ceiling lift (16/rank vs bf16's 9) from fp8's ~2.4x KV pool.
#
# HEADLINE (same as 1k/8k): fp8_e4m3 KV DSA decode works on STOCK latest main. ONLY env var for fp8 is
# SGLANG_DISABLE_DSA_INDEXER_FUSION=1 (DSV4's flash-MLA env vars are INERT for GLM). NVFP4 MoE = flashinfer_cutlass.
#
# 8K/64K MEASUREMENT NOTE: GLM full-DSA long context is KV-pool-bound AND prefill-sensitive. At CONC>~160 the
# 8192-token prefill backlog swamps the scheduler and the server never reaches a sustained decode plateau.
# Measure at CONC~128. Sequences never complete in-window (OSL 65536), so the plateau reflects the growing
# phase (a lower-bound-ish steady state). gsm8k not separately gated (same kernels as 1k/8k; gsm8k is short-context).
set -euo pipefail
NAME=${NAME:-glm52_nvfp4_dpa_8k64k}
IMAGE=${IMAGE:-lmsysorg/sglang:dev-cu13}
# nvidia/GLM-5.2-NVFP4 — pin the complete snapshot b0b2b68 under HF_HUB_OFFLINE.
CKPT=${CKPT:-nvidia/GLM-5.2-NVFP4}
sudo docker run -d --name "$NAME" --gpus all --shm-size 32g --network host --ipc host \
  -v ~/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  -e NCCL_P2P_LEVEL=SYS -e NCCL_MIN_NCHANNELS=8 -e NCCL_ALLOC_P2P_NET_LL_BUFFERS=1 \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_IB_DISABLE=1 \
  -e NCCL_SOCKET_IFNAME=enp128s4,ens3 -e GLOO_SOCKET_IFNAME=ens3 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e SGLANG_DISABLE_DSA_INDEXER_FUSION=1 \
  "$IMAGE" \
  python3 -m sglang.launch_server \
    --model-path "$CKPT" --trust-remote-code \
    --tp 8 --dp-size 8 --enable-dp-attention \
    --moe-a2a-backend none --ep-size 1 --moe-runner-backend flashinfer_cutlass \
    --attention-backend flashinfer --kv-cache-dtype fp8_e4m3 \
    --disable-shared-experts-fusion \
    --mem-fraction-static 0.97 --chunked-prefill-size 2048 \
    --cuda-graph-bs "4 8 12 16 24" --cuda-graph-max-bs 32 \
    --context-length 73728 --max-running-requests 1024 \
    --host 0.0.0.0 --port 8000
# Bench (read steady-state decode plateau from server logs; requests don't complete in-window at OSL 65536):
#   python3 -m sglang.bench_serving --backend sglang --model nvidia/GLM-5.2-NVFP4 \
#     --dataset-name random --random-input-len 8192 --random-output-len 65536 --random-range-ratio 1.0 \
#     --max-concurrency 128 --num-prompts 256
# bf16 alt (~123/GPU, gsm8k 0.920): drop SGLANG_DISABLE_DSA_INDEXER_FUSION, --kv-cache-dtype bfloat16 --mem-fraction-static 0.94.
# ⚠️ CONC>~128 prefill-swamps (8192-token backlog dominates); do not raise --max-concurrency past ~128.
