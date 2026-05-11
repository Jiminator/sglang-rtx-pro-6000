#!/bin/bash
# Run J — Prefill server (node-1, single-node TP=8 DPA + PD-disagg prefill mode, NIXL backend).
# Same per-server flags as the 4,395 tok/s baseline (`2node-smg-2x-mfs-0.85-2026-05-08`),
# extended with PD-disaggregation: --disaggregation-mode prefill,
# --disaggregation-transfer-backend nixl. No --disaggregation-ib-device → NIXL UCX-TCP.
exec python3 -m sglang.launch_server \
  --model-path nvidia/DeepSeek-V3.2-NVFP4 \
  --tokenizer-path deepseek-ai/DeepSeek-V3.2 \
  --trust-remote-code \
  --quantization modelopt_fp4 \
  --disable-shared-experts-fusion \
  --mem-fraction-static 0.85 \
  --schedule-policy fcfs \
  --chunked-prefill-size 16384 \
  --enable-dp-attention \
  --dp-size 8 \
  --enable-dp-lm-head \
  --enable-fused-moe-sum-all-reduce \
  --reasoning-parser deepseek-v3 \
  --tool-call-parser deepseekv32 \
  --nsa-prefill-backend tilelang \
  --nsa-decode-backend tilelang \
  --page-size 64 \
  --kv-cache-dtype bfloat16 \
  --attention-backend flashinfer \
  --moe-runner-backend flashinfer_cutlass \
  --enable-flashinfer-allreduce-fusion \
  --disable-custom-all-reduce \
  --disable-radix-cache \
  --tp-size 8 \
  --pp-size 1 \
  --nnodes 1 \
  --node-rank 0 \
  --host 0.0.0.0 \
  --port 30010 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend nixl \
  --disaggregation-bootstrap-port 8998
