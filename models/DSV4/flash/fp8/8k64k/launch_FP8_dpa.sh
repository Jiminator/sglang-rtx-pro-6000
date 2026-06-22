#!/bin/bash
# DeepSeek-V4-Flash FP8 + DP-attention — the WINNING single-node config (1× g4, 8× RTX PRO 6000, SM120).
# True 8K/64K: 551.9 output tok/s @ batch 33 (~69/GPU), 2.45x over pure TP=8. See TUNING_REPORT.md.
#
# DP-attention parallelizes the 83%-DSA-attention decode bottleneck across 8 workers (one per GPU).
# REQUIRED on SM120:
#   --moe-a2a-backend none      deepep needs deep_gemm, hard-disabled at sm_version==120
#   --moe-runner-backend triton auto force-selects marlin, whose Fp8MoEMethod lacks self.runner
#                               -> AttributeError at cuda-graph capture (triton is the only runner)
#   --dp-size == --tp           dp<tp short-circuits the allreduce and hangs
# DP-attention auto-caps chunked-prefill; do not set it. KV auto-locks fp8_e4m3.
set -euo pipefail
NAME=${NAME:-dsv4_flash_fp8_dpa}
sudo docker run -d --name "$NAME" --gpus all --shm-size 32g --network host --ipc host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e NCCL_P2P_LEVEL=SYS -e NCCL_MIN_NCHANNELS=8 -e NCCL_ALLOC_P2P_NET_LL_BUFFERS=1 \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_IB_DISABLE=1 \
  -e NCCL_SOCKET_IFNAME=enp128s4,ens3 -e GLOO_SOCKET_IFNAME=ens3 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  lmsysorg/sglang:v0.5.13.post1-cu130 \
  python3 -m sglang.launch_server \
    --model-path sgl-project/DeepSeek-V4-Flash-FP8 --served-model-name sgl-project/DeepSeek-V4-Flash-FP8 \
    --trust-remote-code --tp 8 --dp-size 8 --enable-dp-attention \
    --moe-a2a-backend none --moe-runner-backend triton \
    --mem-fraction-static 0.85 --cuda-graph-max-bs 64 --context-length 73728 \
    --host 0.0.0.0 --port 8000
# EAGLE: net loss here (DSA-bound) — omit. FP4 ckpt: slower (36% MoE-GEMV tax) — use FP8.
