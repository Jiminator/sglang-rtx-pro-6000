#!/bin/bash
# Qwen3.6-FP8 SMG router — round-robins requests across the 16 single-GPU TP=1 workers
# (8 per node × 2 nodes). Run on node-1 (or any host that can reach both worker IPs).
#
# Replace the worker URLs below with the actual node-1 / node-2 internal IPs
# (here using the gcp-kimi 10.2.1.x VPC).
set -euo pipefail

exec ~/.local/bin/sglang-router launch \
  --worker-urls \
    http://10.2.1.2:30000 http://10.2.1.2:30001 http://10.2.1.2:30002 http://10.2.1.2:30003 \
    http://10.2.1.2:30004 http://10.2.1.2:30005 http://10.2.1.2:30006 http://10.2.1.2:30007 \
    http://10.2.1.4:30000 http://10.2.1.4:30001 http://10.2.1.4:30002 http://10.2.1.4:30003 \
    http://10.2.1.4:30004 http://10.2.1.4:30005 http://10.2.1.4:30006 http://10.2.1.4:30007 \
  --policy round_robin \
  --host 0.0.0.0 --port 8000 \
  --request-timeout-secs 1800 \
  --log-level info
