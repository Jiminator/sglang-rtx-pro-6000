#!/bin/bash
# Pinned sglang==0.5.10.post1 + K2.5-INT4 prior dp=8 base + 5 optimizations layered back on.
# Skipped --enable-flashinfer-allreduce-fusion because:
#   (a) silent no-op on PCIe Blackwell (no NVL/multicast substrate)
#   (b) flashinfer 0.6.7.post3 wrapper vs 0.6.8.post1 cubins drift could bite the AR kernel call
set -euo pipefail
NODE_RANK="${NODE_RANK:?NODE_RANK env var required}"

echo "=== Installing sglang==0.5.10.post1 ==="
pip install --quiet sglang==0.5.10.post1 2>&1 | tail -5 || { echo "PIP_INSTALL_FAILED"; exit 1; }
python3 -c "import sglang; print('sglang:', sglang.__version__)"

cd /
exec python3 -m sglang.launch_server \
  --model-path moonshotai/Kimi-K2.5 \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --dp-size 8 \
  --enable-dp-attention \
  --nnodes 2 \
  --node-rank "$NODE_RANK" \
  --dist-init-addr 10.0.1.2:5000 \
  --trust-remote-code \
  --reasoning-parser kimi_k2 \
  --tool-call-parser kimi_k2 \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 16384 \
  --kv-cache-dtype fp8_e5m2 \
  --schedule-policy fcfs \
  --disable-shared-experts-fusion \
  --disable-custom-all-reduce \
  --disable-radix-cache \
  --enable-dp-lm-head \
  --enable-fused-moe-sum-all-reduce
