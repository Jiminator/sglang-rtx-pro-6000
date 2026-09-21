#!/bin/bash
# 8K in / 1K out at concurrency 128. Every request is exactly 8192/1024 tokens
# (--random-range-ratio 1.0), and the prefix cache is flushed after warmup.
set -euo pipefail
docker exec sglang-dsv41 python3 -m sglang.bench_serving \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --model /models/DeepSeek-V4.1-Flash \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 1024 \
  --random-range-ratio 1.0 \
  --num-prompts 256 \
  --max-concurrency 128 \
  --seed 42 \
  --flush-cache \
  --cache-report \
  --output-file /tmp/dsv41_8k1k.json
