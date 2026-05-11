#!/bin/bash
# Run G — Router launch (sglang-router on dedicated container, node-1).
ulimit -n 131072
exec python3 -m sglang_router.launch_router \
  --worker-urls http://10.0.1.2:8000 http://10.0.1.4:8000 \
  --policy round_robin \
  --host 0.0.0.0 \
  --port 30000 \
  --prometheus-port 19090
