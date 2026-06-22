#!/bin/bash
# DeepSeek-V4-Flash (MXFP4 experts) + DP-attention — MAX batch-scaled config for the 1K/8K workload
# (ISL 1024 / OSL 8192) on 1x g4 (8x RTX PRO 6000, SM120).
# Offline bench_one_batch_server, steady-state decode plateau: 497.6 tok/s aggregate (62.2/GPU)
# @ 214 running-req/rank. See REPORT.md.
#
# NOTE: this is the launch-space CEILING for the MXFP4 checkpoint and is 6.9x BELOW the FP8 checkpoint
# (sgl-project/DeepSeek-V4-Flash-FP8 = 3412.9 tok/s, see ../../fp8/1k8k/). For production at 1K/8K, ship
# FP8. This bundle documents the best achievable with the MXFP4 weights and NO source-code changes.
#
# WHY ONLY ~497: the only working SM120 MXFP4 MoE kernel is the per-(token,expert)-slot triton GEMV
# (_mxfp4_slot_gemv_kernel in fused_moe_triton/mxfp4_moe_sm120_triton.py). It reloads each expert's
# weights for every token routed to it (no reuse across the token dim) -> weight-bandwidth-bound ->
# per-request decode collapses to ~0.29 tok/s and the MoE is compute-saturated. Adding batch does NOT
# help (see the flat curve in results/): 180/rank=491, 214/rank=498, 224/rank=492 tok/s (all ~noise).
#
# WHY batch-scaling is the only knob and it's exhausted:
#   - max_running_requests is force-capped at 256/rank by DeepseekV4ForCausalLM.
#   - --swa-full-tokens-ratio enlarges the SWA pool (the batch binder) -> more concurrent req/rank,
#     but throughput is flat because the MXFP4 MoE GEMV is compute-bound, not batch-bound.
#   - ratio 0.15 + mfs 0.9 admits ~214/rank (the nominal max); 0.1/0.2 tie within noise.
#
# SM120 required flags:
#   --moe-a2a-backend none       deepep needs deep_gemm, hard-disabled at sm_version==120
#   --moe-runner-backend marlin  MXFP4 experts -> marlin reroutes to the sm120_triton GEMV
#                                (triton routes mxfp4 through the FP8 path -> hidden-size crash;
#                                 marlin CUDA mxfp4 kernel NaNs on SM120 -> auto sm120_triton fallback)
#   --dp-size == --tp            dp<tp short-circuits the allreduce and hangs
# KV auto-locks fp8_e4m3. attention_backend auto-resolves to 'dsv4' (DSA).
set -euo pipefail
NAME=${NAME:-dsv4_flash_mxfp4_dpa_1k8k}
sudo docker run -d --name "$NAME" --gpus all --shm-size 32g --network host --ipc host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e NCCL_P2P_LEVEL=SYS -e NCCL_MIN_NCHANNELS=8 -e NCCL_ALLOC_P2P_NET_LL_BUFFERS=1 \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_IB_DISABLE=1 \
  -e NCCL_SOCKET_IFNAME=enp128s4,ens3 -e GLOO_SOCKET_IFNAME=ens3 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  lmsysorg/sglang:v0.5.13.post1-cu130 \
  python3 -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V4-Flash --served-model-name deepseek-ai/DeepSeek-V4-Flash \
    --trust-remote-code --tp 8 --dp-size 8 --enable-dp-attention \
    --moe-a2a-backend none --moe-runner-backend marlin \
    --mem-fraction-static 0.9 --swa-full-tokens-ratio 0.15 \
    --cuda-graph-max-bs 256 --context-length 9216 \
    --max-running-requests 2048 \
    --host 0.0.0.0 --port 8000
# Offline bench: bench_one_batch_server --input-len 1024 --output-len 8192 --batch-size 1712
#   (1712 = 214/rank x dp_size 8; SWA-pool-bound at ratio 0.15 / mfs 0.9).
# Measurement: steady-state decode `gen throughput` x 8 ranks once the prefill ramp drains
#   (~50 min at OSL 8192 — FP4 prefill is also slow, ~64 tok/s/rank, same GEMV bottleneck).
