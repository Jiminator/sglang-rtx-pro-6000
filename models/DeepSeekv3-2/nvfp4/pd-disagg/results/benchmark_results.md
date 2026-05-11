# DeepSeek-V3.2-NVFP4 — PD-Disaggregation (NIXL UCX-TCP)

Benchmark running `nvidia/DeepSeek-V3.2-NVFP4` on **2 nodes G4 / 16 GPUs total**, deployed in **prefill-decode separated topology**: one prefill server on node-1, one decode server on node-2, sglang-router with `--pd-disaggregation` policy in front. KV state copies from prefill → decode over NIXL UCX-TCP (no IB device, no RDMA on this cluster).

- 1,536 prompts, ISL=1024 / OSL=8192, max-conc 512, random dataset, seed=1, `--apply-chat-template`
- Per-server topology: single-node TP=8 PP=1 + DPA dp_size=8, mfs=0.85, page=64
- Transfer backend: **NIXL UCX-TCP** (`--disaggregation-transfer-backend nixl`, no `--disaggregation-ib-device` → falls back to TCP)
- Image: `lmsysorg/sglang:dev-cu13`
- NCCL/GLOO env vars passed via wrapper (load-bearing)

**Result: output 3,730.39 tok/s, total 4,185.45 tok/s — −15.1% output vs 2× SMG.** On this PCIe-Blackwell + gVNIC-TCP cluster, PD-Disaggregation does **not** beat 2× SMG unified routing. The inter-node KV-state copy over TCP costs more than the decode-isolation benefit gains.

```
============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    9999.0
Max request concurrency:                 512
Successful requests:                     1536
Benchmark duration (s):                  1724.99
Total input tokens:                      784969
Total input text tokens:                 784969
Total generated tokens:                  6434886
Total generated tokens (retokenized):    6354847
Request throughput (req/s):              0.89
Input token throughput (tok/s):          455.06
Output token throughput (tok/s):         3730.39
Peak output token throughput (tok/s):    5886.00
Peak concurrent requests:                517
Total token throughput (tok/s):          4185.45
Concurrency:                             405.70
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   455615.12
Median E2E Latency (ms):                 457594.53
P90 E2E Latency (ms):                    788053.98
P99 E2E Latency (ms):                    933401.34
---------------Time to First Token----------------
Mean TTFT (ms):                          18250.79
Median TTFT (ms):                        498.54
P99 TTFT (ms):                           259647.94
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          111.64
Median TPOT (ms):                        107.10
P99 TPOT (ms):                           118.34
---------------Inter-Token Latency----------------
Mean ITL (ms):                           104.53
Median ITL (ms):                         100.99
P95 ITL (ms):                            111.22
P99 ITL (ms):                            269.24
==================================================
```

## When PD-Disaggregation does and doesn't help on this cluster

**This stack:** −15% vs unified 2× SMG. The mean TTFT jumped from 4.1 s (unified) to 18.3 s, with a long tail (p99 = 259 s) reflecting requests queued for KV transfer. Decode TPOT is similar (107 vs 95 ms unified) — the architectural decode-never-interrupted-by-prefill premise does **not** convert to a per-token win because the KV-transfer copies block decode batches at sub-step granularity.

**Where PD-Disagg would win:** RDMA-equipped clusters (Mellanox HCAs visible to NIXL), NVL-72 / NVSwitch fabrics, or workloads where inter-node copy bandwidth isn't the binding constraint (very long OSL with small ISL, where transfer-per-request is much smaller).

For prefill-heavy long-context workloads on no-RDMA hardware, PD-Disagg is decisively negative (see GLM-5.1-NVFP4 16k1k bundle's testing for a worse case: −80.7%).

## Deploying

1. **Prefill** on node-1: `prefill_launch.sh` inside the container, listens on `:30010` with bootstrap port `:8998`.
2. **Decode** on node-2: `decode_launch.sh` inside the container, listens on `:30011`.
3. **Router** on node-1 (any reachable host works): `router_launch.sh`, listens on `:30000`. Wires to prefill+decode endpoints.
4. Bench client → `http://10.0.1.2:30000`.

NCCL/GLOO env vars same as the SMG case — set at container start via `scripts/docker_run_sglang_worker.sh` wrapper.
