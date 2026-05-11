# GLM-5.1-NVFP4 — 16K ISL / 1K OSL workload

These are the benchmark results running `lukealonso/GLM-5.1-NVFP4` on 2 nodes G4 / 16 GPUs (g4-standard-384) with a **long-context prefill-dominant workload**:

- 1,536 prompts
- ISL = **16,384** input tokens / OSL = **1,024** output tokens (per request)
- max concurrency 512
- random dataset, seed=1, `--apply-chat-template`
- Topology: 2-node TP=8 PP=2 + DPA dp_size=8, mfs=0.90, page=64
- Key optimization: `--enable-dynamic-chunking` + `SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR=0.65`
- Image: `lmsysorg/sglang:dev-cu13`
- NCCL/GLOO env vars passed via `scripts/docker_run_sglang_worker.sh` wrapper (load-bearing)

**+5.66% total throughput** over the prior static-chunked-prefill anchor (4,574.19 → 4,833.78 tok/s) by enabling dynamic chunking (`SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR=0.65`) on top of the existing PP=2 + DPA + flashinfer + FI-AR-fusion stack.

```
============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    9999.0
Max request concurrency:                 512
Successful requests:                     1536
Benchmark duration (s):                  2786.92
Total input tokens:                      12698185
Total input text tokens:                 12698185
Total generated tokens:                  773190
Total generated tokens (retokenized):    768403
Request throughput (req/s):              0.55
Input token throughput (tok/s):          4556.34
Output token throughput (tok/s):         277.43
Peak output token throughput (tok/s):    3698.00
Peak concurrent requests:                516
Total token throughput (tok/s):          4833.78
Concurrency:                             502.71
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   912123.62
Median E2E Latency (ms):                 831562.03
P90 E2E Latency (ms):                    1803280.13
P99 E2E Latency (ms):                    2503868.67
---------------Time to First Token----------------
Mean TTFT (ms):                          117962.69
Median TTFT (ms):                        12973.86
P99 TTFT (ms):                           674109.49
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          2440.57
Median TPOT (ms):                        1717.21
P99 TPOT (ms):                           17797.15
---------------Inter-Token Latency----------------
Mean ITL (ms):                           1589.75
Median ITL (ms):                         845.14
P95 ITL (ms):                            5993.64
P99 ITL (ms):                            16084.23
Max ITL (ms):                            749602.85
==================================================
```

## Deploying

Both nodes use the same docker image and NCCL env vars. Node 1 is the rendezvous (`--node-rank 0 --dist-init-addr 10.0.1.2:5000`); node 2 is `--node-rank 1`. Bench client points at `http://<node-1>:30000`.

Use the `scripts/docker_run_sglang_worker.sh` wrapper to bake in load-bearing NCCL/GLOO env vars: `NCCL_P2P_LEVEL=SYS`, `NCCL_MIN_NCHANNELS=8`, `NCCL_ALLOC_P2P_NET_LL_BUFFERS=1`, `NCCL_NVLS_ENABLE=0`, `NCCL_CUMEM_ENABLE=0`, `NCCL_IB_DISABLE=1`, `NCCL_SOCKET_IFNAME=enp128s4,ens4`, `GLOO_SOCKET_IFNAME=ens4`. Without these, throughput drops ~10–15%.

In addition, export `SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR=0.65` inside the container before launch (the `launch_node*.sh` scripts already do this).
