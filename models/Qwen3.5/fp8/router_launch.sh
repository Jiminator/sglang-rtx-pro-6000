#!/bin/bash
# Qwen3.5-FP8 SMG router — round-robins requests between the two single-node TP=8 workers.
# Run on node-1 (or any host that can reach both worker IPs).
#
# Replace the worker URLs below with the actual node-1 / node-2 internal IPs (here using the gcp-kimi 10.2.1.x VPC).
set -euo pipefail

exec ~/.local/bin/sglang-router launch \
  --worker-urls http://10.2.1.2:30000 http://10.2.1.4:30000 \
  --policy round_robin \
  --host 0.0.0.0 --port 8000 \
  --request-timeout-secs 1800 \
  --log-level info
