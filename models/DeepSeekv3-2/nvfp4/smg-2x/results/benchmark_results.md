# DeepSeek-V3.2-NVFP4 — 2× Replica behind SMG (router)

Benchmark running `nvidia/DeepSeek-V3.2-NVFP4` on **2 nodes G4 / 16 GPUs total**, deployed as **two independent single-node replicas** sitting behind one sglang-router (round-robin policy). The router is the unified `:30000` endpoint.

- 1,536 prompts, ISL=1024 / OSL=8192, max-conc 512, random dataset, seed=1, `--apply-chat-template`
- Per-replica topology: single-node TP=8 PP=1 + DPA dp_size=8, mfs=**0.85**, page=64
- Image: `lmsysorg/sglang:dev-cu13`
- NCCL/GLOO env vars passed via wrapper (load-bearing — see notes below)

**This config is the ship recommendation** for DSv3.2-NVFP4 random workloads: **output 4,395.44 tok/s, total 4,931.62 tok/s** — beats the single-node EAGLE config (2,675 / 3,012) by ~64% while serving 16 GPUs instead of 8.

```
============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    inf
Max request concurrency:                 512
Successful requests:                     1536
Benchmark duration (s):                  1463.99
Total input tokens:                      784969
Total input text tokens:                 784969
Total generated tokens:                  6434886
Total generated tokens (retokenized):    6367072
Request throughput (req/s):              1.05
Input token throughput (tok/s):          536.18
Output token throughput (tok/s):         4395.44
Peak output token throughput (tok/s):    6911.00
Peak concurrent requests:                518
Total token throughput (tok/s):          4931.62
Concurrency:                             412.36
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   393029.93
Median E2E Latency (ms):                 395780.91
P90 E2E Latency (ms):                    695690.51
P99 E2E Latency (ms):                    790482.13
---------------Time to First Token----------------
Mean TTFT (ms):                          4062.28
Median TTFT (ms):                        395.95
P99 TTFT (ms):                           15990.04
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          96.63
Median TPOT (ms):                        94.98
P99 TPOT (ms):                           105.22
---------------Inter-Token Latency----------------
Mean ITL (ms):                           93.03
Median ITL (ms):                         90.72
P95 ITL (ms):                            103.44
P99 ITL (ms):                            223.17
Max ITL (ms):                            14580.67
==================================================
```

## Deploying

1. Launch `worker_launch.sh` inside an `lmsysorg/sglang:dev-cu13` container on **each of the two G4 nodes**. Each replica is independent — they don't talk to each other. Both listen on `:8000` locally.
2. Launch `router_launch.sh` on node-1 (or anywhere with network reachability to both `:8000` workers). Router listens on `:30000` and routes by round_robin across `http://10.0.1.2:8000` and `http://10.0.1.4:8000`.
3. Bench client → `http://10.0.1.2:30000` (router).

NCCL/GLOO env vars are load-bearing and must be present at container start. Use `scripts/docker_run_sglang_worker.sh` from the parent repo, or set them manually:
`NCCL_P2P_LEVEL=SYS`, `NCCL_MIN_NCHANNELS=8`, `NCCL_ALLOC_P2P_NET_LL_BUFFERS=1`, `NCCL_NVLS_ENABLE=0`, `NCCL_CUMEM_ENABLE=0`, `NCCL_IB_DISABLE=1`, `NCCL_SOCKET_IFNAME=enp128s4,ens4`, `GLOO_SOCKET_IFNAME=ens4`. Without these, throughput drops ~10–15% and TPOT regresses ~16%.

## Why this beats single-node EAGLE

Two single-node replicas behind a router gives **2× model copies, 2× KV pool, 2× decode bandwidth** vs one single-node deployment. The EAGLE config trades raw throughput for per-token latency via speculative decoding — useful when you want lower TPOT, but not the winner on tokens/s. For high-throughput batch serving, 2× SMG replica wins decisively.

The mfs=0.85 setting is calibrated: anything higher (0.86+) OOMs on the FusedMoE workspace; anything lower wastes KV pool headroom that's measurably useful for tail TTFT.
