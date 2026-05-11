#!/bin/bash
# Run J — Router with --pd-disaggregation, on node-1, host port 30000 (matches bench URL).
ulimit -n 131072
exec python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://10.0.1.2:30010 8998 \
  --decode http://10.0.1.4:30011 \
  --policy cache_aware \
  --host 0.0.0.0 \
  --port 30000 \
  --prometheus-port 19090
