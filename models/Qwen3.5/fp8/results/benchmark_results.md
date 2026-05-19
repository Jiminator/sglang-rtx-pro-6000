# Qwen3.5-397B-A17B-FP8 — 2× SMG TP=8 + v38 ship config

Benchmark running `Qwen/Qwen3.5-397B-A17B-FP8` (FP8 block-scaled W8A8, ~400 GB on disk) on **2 nodes G4 / 16 GPUs total**, replicated as **2× single-node TP=8 workers behind an `sglang-router`** with `--policy round_robin`. Image: `lmsysorg/sglang:dev-cu13`, sglang `0.0.0.dev1+gedb1b3f8f`.

- 200 prompts, ISL target 20K / OSL target 1K, `--random-range-ratio 0.0` (variable lengths → effective ISL ~10K avg, OSL ~500 avg)
- `--max-concurrency 40`, `--request-rate inf`
- Topology: 2× single-node TP=8 SMG fan-out (no PP, no cross-node hot path)
- `--chunked-prefill-size 4096`, `--max-prefill-tokens 32768`, `--enable-mixed-chunk`
- `--enable-fused-qk-norm-rope`, `--enable-fused-moe-sum-all-reduce`, `--enable-flashinfer-allreduce-fusion`, `--enforce-piecewise-cuda-graph`
- `--mem-fraction-static 0.8`, `--disable-radix-cache`
- env: NCCL/GLOO baseline (via `scripts/docker_run_sglang_worker.sh`) + `NCCL_NCHANNELS=16`

**Result: median TTFT 1,180 ms, mean TTFT 2,256 ms, P99 TTFT 14,320 ms, total throughput 14,182 tok/s, 200/200 completed.**

```
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 40
Successful requests:                     200
Total input tokens:                      2,062,211
Output token throughput (tok/s):         654.45
Peak output token throughput (tok/s):    1,500.00
Peak concurrent requests:                44
Total token throughput (tok/s):          14,182.22
Concurrency:                             37.19
---------------Time to First Token----------------
Mean TTFT (ms):                          2,256.43
Median TTFT (ms):                        1,180.25
P99 TTFT (ms):                           14,320.58
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          56.46
Median TPOT (ms):                        53.84
P99 TPOT (ms):                           161.83
==================================================
```

## Deploying

Both nodes run the same `launch_worker.sh` (no rendezvous required since each replica is fully self-contained):

```bash
# On both nodes
./launch_worker.sh

# On node-1 only
./router_launch.sh
```

The router needs the worker URLs updated for your VPC; the included script targets `10.2.1.2:30000` (node-1) and `10.2.1.4:30000` (node-2) for the `sglang-jax-1126` cluster.

NCCL/GLOO env vars passed via `scripts/docker_run_sglang_worker.sh` wrapper. `NCCL_NCHANNELS=16` is set inside `launch_worker.sh` itself.

## Closed dimensions

Do not retry without new evidence (see `../TUNING_REPORT.md` for the full list and root causes):

- Cross-node PP=2: lower throughput than 2× SMG (9132 vs 14182 tok/s) at near-identical TTFT
- PD-Disaggregation (mooncake or NIXL/UCX): bandwidth-bound on cross-node gVNIC TCP for 20K-ISL KV transfer
- Speculative decoding (NEXTN, NGRAM, EAGLE3-on-Qwen3.5): all hurt mean TTFT despite improving TPOT
- Context parallelism: cross-rank attention sync exceeds saved kernel compute
- Triton attention backend: 105 % slower than flashinfer for this model
- `--page-size > 1`: flashinfer paged attention is optimized for page=1
- `--prefill-max-requests` caps and `--max-running-requests` caps below conc=40: both regress TTFT
- Custom `--cuda-graph-bs` lists: extra captures eat KV pool

## Workload note

These numbers are at a **TTFT-focused workload** (conc=40, ISL 20K, OSL 1K) — different from the throughput-focused workloads used by the other models in this repo. Conc, prompt count, and ISL/OSL all favor first-token latency over steady-state throughput. Direct comparison to other entries' total throughput numbers should be adjusted for the workload difference (~7× fewer concurrent reqs, ~10× shorter total bench).
