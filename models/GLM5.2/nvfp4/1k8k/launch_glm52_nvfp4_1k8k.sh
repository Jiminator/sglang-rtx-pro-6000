#!/bin/bash
# GLM-5.2-NVFP4 + DP-attention — 1K/8K WINNER on 1x g4 (8x RTX PRO 6000, SM120).
# nvidia/GLM-5.2-NVFP4 (GlmMoeDsa DSA arch, expert-only NVFP4), STOCK latest-main dev-cu13, bench_serving rrr=1.0.
# STEADY-STATE DECODE PLATEAU: ~2680 output tok/s (~335/GPU) @ 52 running-req/rank, gsm8k 0.900. See TUNING_REPORT.md.
# Ties the EAGLE spec variant (~330/GPU); non-spec is the top single number, spec has the correctness/latency edge.
#
# HEADLINE: fp8_e4m3 KV DSA decode now works on STOCK latest main (the old glm-opt-branch-only "no SM120 DSA
# decode kernel" crash is gone). The ONLY env var needed for fp8 is SGLANG_DISABLE_DSA_INDEXER_FUSION=1
# (ablation A1a: gsm8k 0.940). The DSV4 vars (SGLANG_SM120_FLASHMLA_BACKEND / SGLANG_OPT_FLASHMLA_SPARSE_PREFILL)
# are INERT for GLM (different DSA dispatch). PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is for boot.
# NVFP4 MoE runner = flashinfer_cutlass (only viable on SM120; marlin gsm8k 0.02). Pool is the lever; mfs0.975 ceiling.
set -euo pipefail
NAME=${NAME:-glm52_nvfp4_dpa_1k8k}
IMAGE=${IMAGE:-lmsysorg/sglang:dev-cu13}
# nvidia/GLM-5.2-NVFP4 — pin the complete snapshot b0b2b68 under HF_HUB_OFFLINE (refs/main drifted to a partial).
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
    --mem-fraction-static 0.975 --chunked-prefill-size 2048 \
    --cuda-graph-bs "16 32 48 64" --context-length 9472 --max-running-requests 1024 \
    --host 0.0.0.0 --port 8000
# Bench (loadgen, read steady-state decode plateau from server logs):
#   python3 -m sglang.bench_serving --backend sglang --model nvidia/GLM-5.2-NVFP4 \
#     --dataset-name random --random-input-len 1024 --random-output-len 8192 --random-range-ratio 1.0 \
#     --max-concurrency 512 --num-prompts 2048
#
# ---- ALTERNATES ----
# (a) EAGLE-3 spec variant (~330/GPU, gsm8k 0.940, accept-len 4.0, 14-16 running/rank — correctness+latency pick).
#     mfs 0.97 (draft+verify pool). Add:
#       --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-num-draft-tokens 4 \
#       --speculative-eagle-topk 1 --speculative-moe-runner-backend flashinfer_cutlass \
#       --speculative-moe-a2a-backend none
#     (--speculative-eagle-topk MUST be 1: flashinfer-MLA topk=1 only for spec on SM120.)
# (b) bf16 conservative fallback (~158/GPU, gsm8k 0.920, pool-bound & decays): drop SGLANG_DISABLE_DSA_INDEXER_FUSION,
#     set --kv-cache-dtype bfloat16 --mem-fraction-static 0.94.
