#!/bin/bash
# DeepSeek-V4-Flash FP8 + DP-attention — WINNING single-node config for the 1K/8K workload
# (ISL 1024 / OSL 8192) on 1x g4 (8x RTX PRO 6000, SM120).
# Offline bench_one_batch_server, batch maxed to the SWA pool: 3412.9 tok/s @ B=1016 (~427/GPU),
# 7.1x over pure TP=8. See TUNING_REPORT.md.
#
# WHY DPA: decode is memory-bandwidth-bound, so throughput scales with batch. The hybrid model's
# small SWA/DSA-sparse KV pool is the batch ceiling; pure TP=8 has ONE pool (batch ~146), DP-attention
# gives each of 8 GPUs its own pool (batch ~1016). Same per-request rate -> ~7x aggregate.
#
# SM120 required flags:
#   --moe-a2a-backend none      deepep needs deep_gemm, hard-disabled at sm_version==120
#   --moe-runner-backend triton FP8 MoE runner (auto->marlin lacks self.runner -> cuda-graph crash)
#   --dp-size == --tp           dp<tp short-circuits the allreduce and hangs
# Default --swa-full-tokens-ratio (0.1) is best: raising it adds batch but decode is already
# saturated at ~1016 (neutral-to-negative). KV auto-locks fp8_e4m3.
set -euo pipefail
NAME=${NAME:-dsv4_flash_fp8_dpa_1k8k}
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
    --mem-fraction-static 0.85 --cuda-graph-max-bs 384 --context-length 9216 \
    --max-running-requests 2048 \
    --host 0.0.0.0 --port 8000
# Offline bench: bench_one_batch_server --input-len 1024 --output-len 8192 --batch-size 1016
# (1016 = SWA-pool-bound: 0.1 x per-worker-pool / ~2048 sparse-tok/seq x dp_size=8). EAGLE: see report.
