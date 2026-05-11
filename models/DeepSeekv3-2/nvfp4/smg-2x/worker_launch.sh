#!/bin/bash
# Run G — One-replica SGLang worker. Identical to Run B except:
#   --mem-fraction-static 0.82  →  0.85  (+0.03)
#
# Purpose: free ~3 GB / GPU for the L1 KV pool to test the KV-pressure
# hypothesis directly, without offloader interaction. OffloaderV1 (V1) and
# OffloaderV2 (V2) both share the `offloader.py:259 functional_call` codepath
# that crashes on DSv3.2 MLA-absorb's `self.w_kc` instance attribute (Run D
# failure + Run E §1 evidence). `--mem-fraction-static` does not touch the
# offloader.
#
# Risk: activation OOM under burst (the standard mem-fraction-static knob
# tradeoff). If it OOMs, retry at 0.84.
#
# Container preconditions:
#   * --network host
#   * --ulimit nofile=131072:131072
#   * --shm-size 64g
#   * --gpus all
#
# NO HiCache, NO Mooncake/NIXL L3 backend, NO pip install, NO offloader.
# This is the cleanest single-variable probe of the L1-KV-pool theory.
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
  --host 0.0.0.0 \
  --port 8000
