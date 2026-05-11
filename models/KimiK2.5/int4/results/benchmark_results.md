# Kimi-K2.5-INT4 — 2-node TP=8 PP=2 + DPA (ship anchor)

Benchmark running `moonshotai/Kimi-K2.5` (compressed-tensors INT4, group=32, ~555 GB on disk) on **2 nodes G4 / 16 GPUs total**, using pinned `sglang==0.5.10.post1` over the `dev-cu13` image.

- 1,536 prompts, ISL=1024 / OSL=8192, max-conc 512, random dataset, seed=1, `--apply-chat-template`
- Topology: 2-node TP=8 PP=2 + DPA dp_size=8, mfs=0.85, page=1 (default), chunked-prefill 16384 (auto-capped by DPA to 2048)
- KV cache dtype: **fp8_e5m2** (halves KV memory vs bf16)
- Image: `lmsysorg/sglang:dev-cu13` with `pip install sglang==0.5.10.post1` overriding the image's `0.0.0.dev1+gcbc2bee54` (bypasses a cudagraph buffer regression on triton attention)
- NCCL/GLOO env vars passed via wrapper (load-bearing)

**Result: output 3,206.10 tok/s, total 3,597.20 tok/s, 1,536/1,536 completed.**

This is the ship anchor for K2.5-INT4. Native 2× SMG replicas (one server per node, no PP cross-node) were tested and lost: with K2.5-INT4's 555 GB weight footprint, single-node TP=8 leaves only ~26 GB per GPU for KV+activation, which KV-starves the bench. PP=2 splits weights across both nodes so each GPU holds only 35 GB, leaving 61 GB headroom — the only viable shape on this hardware.

```
============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    9999.0
Max request concurrency:                 512
Successful requests:                     1536
Benchmark duration (s):                  2007.08
Total input tokens:                      784969
Total input text tokens:                 784969
Total generated tokens:                  6434886
Total generated tokens (retokenized):    6429523
Request throughput (req/s):              0.77
Input token throughput (tok/s):          391.10
Output token throughput (tok/s):         3206.10
Peak output token throughput (tok/s):    4785.00
Peak concurrent requests:                517
Total token throughput (tok/s):          3597.20
Concurrency:                             409.53
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   535126.26
Median E2E Latency (ms):                 532632.37
P90 E2E Latency (ms):                    940758.02
P99 E2E Latency (ms):                    1061801.89
---------------Time to First Token----------------
Mean TTFT (ms):                          4684.12
Median TTFT (ms):                        311.84
P99 TTFT (ms):                           22596.81
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          133.50
Median TPOT (ms):                        126.41
P99 TPOT (ms):                           143.98
---------------Inter-Token Latency----------------
Mean ITL (ms):                           126.78
Median ITL (ms):                         118.96
P95 ITL (ms):                            182.56
P99 ITL (ms):                            281.94
Max ITL (ms):                            23020.88
==================================================
```

## Deploying

Same `launch_node.sh` runs on both nodes — only `NODE_RANK` env var differs:

```bash
# node 1 (rendezvous)
NODE_RANK=0 ./launch_node.sh
# node 2
NODE_RANK=1 ./launch_node.sh
```

The script pip-installs `sglang==0.5.10.post1` inside the container at startup before launching the server. This pin is mandatory because the `dev-cu13` image's bundled sglang dev build has a cudagraph buffer regression on triton attention that drops K2.5-INT4 throughput by ~5%.

NCCL/GLOO env vars passed via `scripts/docker_run_sglang_worker.sh` wrapper, same as other configs in this repo. Specifically `FLASHINFER_DISABLE_VERSION_CHECK=1` is needed to bypass the pinned sglang's flashinfer wrapper-vs-cubin drift after pip install.

## Closed dimensions

These were tested and shown not to help (do not retry without new evidence):

- **`--enable-flashinfer-allreduce-fusion`**: silent no-op on PCIe Blackwell (no NVL multicast substrate); flashinfer 0.6.7 wrapper vs 0.6.8 cubins after the pin caused stability concerns. Drop the flag.
- **2× SMG replica with `dp_size=8`**: per-DP-rank KV pool collapses to ~8.5K tokens; 75% of bench requests time out. Closed.
- **2× SMG replica with `dp_size=1` (pure TP=8 attention)**: 1,805.61 tok/s, completes 1,536/1,536 but mean TTFT = 11.4 min. Pool is 5× undersized for the bench. Closed.
- **ktransformers CPU-expert offload**: blocked by Blackwell SM_120 not being in kt-kernel's pre-built support matrix (Ampere/Ada/Hopper only). Would require custom CMake build with `TORCH_CUDA_ARCH_LIST="12.0"`; out of scope.
