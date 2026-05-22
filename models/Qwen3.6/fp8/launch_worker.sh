#!/bin/bash
# Qwen3.6-35B-A3B-FP8 — ship config.
# 8 single-GPU TP=1 sglang.launch_server processes (one per GPU 0..7).
# Run identical script on both G4 nodes; the router (router_launch.sh) on node-1 fans out to all 16 endpoints.
#
# Sweep history (Qwen3.6 SOTA iteration 2026-05-15 / 19):
#   - 16x TP=1 + SMG round_robin (this config) — 273 ms median TTFT (best observed)
#   - 8x TP=2 + SMG                 — 308 ms (more queueing under conc=10 per replica)
#   - 4x TP=4 + lever stack + SMG   — 294 ms (queueing penalty)
#   - 1x TP=4 + lever stack         — 1810 ms (single replica drowns conc=10)
#
# Closed dimensions (do NOT retry without source-code reason):
#   - fa3/fa4 attention rejected on Blackwell SM12
#   - flashinfer_trtllm/flashinfer_cutlass/triton_kernel/deep_gemm MoE backends all crash on this stack
#   - TP=8 incompatible with FP8 block_n=128 (output_size 64 not divisible)
#   - --enable-torch-compile autotune does not converge in reasonable time on TP=4
#
# NCCL/GLOO env vars (load-bearing on PCIe Blackwell) are set by the top-level
# scripts/docker_run_sglang_worker.sh wrapper. NCCL_NCHANNELS=16 below shaves an extra
# few % off P99 TTFT (carried over from the Qwen3.5 v38 ship).
#
# Invoke INSIDE the sglang-worker container (mounted by docker_run_sglang_worker.sh).

set -uo pipefail
export FLASHINFER_DISABLE_VERSION_CHECK=1
export NCCL_NCHANNELS=16

python3 -c "import sglang; print('sglang:', sglang.__version__)"

PIDS=()
for GPU in 0 1 2 3 4 5 6 7; do
  PORT=$((30000 + GPU))
  CUDA_VISIBLE_DEVICES=$GPU python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3.6-35B-A3B-FP8 \
    --tp 1 \
    --chunked-prefill-size 16384 \
    --max-prefill-tokens 16384 \
    --mem-fraction-static 0.7 \
    --disable-radix-cache \
    --enable-fused-qk-norm-rope \
    --enable-fused-moe-sum-all-reduce \
    --enforce-piecewise-cuda-graph \
    --enable-mixed-chunk \
    --host 0.0.0.0 --port "$PORT" \
    >/tmp/worker_${GPU}.log 2>&1 &
  PIDS+=($!)
  echo "launched gpu=$GPU port=$PORT pid=${PIDS[-1]}"
done

for pid in "${PIDS[@]}"; do
  wait "$pid"
done
