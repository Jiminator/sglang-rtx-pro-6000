#!/bin/bash
# Qwen3.5-397B-A17B-FP8 — ship config (v38).
# Single-node TP=8 worker; run identical script on both G4 nodes, then launch the router (router_launch.sh) on one.
#
# Sweep history: v6 (single-node baseline) -> v20b (PP=2 cross-node) -> v24 (SMG mixed-chunk) -> v30 (+ fused kernels) -> v38 (+ NCCL_NCHANNELS=16).
# See TUNING_REPORT.md for the full sweep.
#
# NCCL/GLOO env vars (load-bearing on PCIe Blackwell) are set by the top-level
# `scripts/docker_run_sglang_worker.sh` wrapper. NCCL_NCHANNELS=16 below shaves
# an extra ~4.6% off P99 TTFT.

set -euo pipefail
export FLASHINFER_DISABLE_VERSION_CHECK=1
export NCCL_NCHANNELS=16

python3 -c "import sglang; print('sglang:', sglang.__version__)"

cd /
exec python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-397B-A17B-FP8 \
  --tp 8 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 32768 \
  --enable-mixed-chunk \
  --enable-fused-qk-norm-rope \
  --enable-fused-moe-sum-all-reduce \
  --enable-flashinfer-allreduce-fusion \
  --enforce-piecewise-cuda-graph \
  --mem-fraction-static 0.8 \
  --disable-radix-cache \
  --host 0.0.0.0 --port 30000
