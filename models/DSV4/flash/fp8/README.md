# DeepSeek-V4-Flash FP8 — single node (RTX PRO 6000, SM120)

`sgl-project/DeepSeek-V4-Flash-FP8`. Best single-node config by workload. **DP-attention wins both**,
for the same structural reason: DeepSeek-V4-Flash is a **hybrid-attention** model whose small
**SWA/DSA-sparse KV pool** (~10% of memory) caps the batch — and DP-attention gives each of the 8 GPUs
its own pool (8× the batch ceiling).

| Workload (ISL/OSL) | Ship config | Output tok/s | /GPU | vs pure TP=8 | dir |
|---|---|---:|---:|---:|---|
| **1024 / 8192** | TP=8 + DP-attention | **3412.9** @ B=1016 | 426.6 | **7.1×** | [`1k8k/`](1k8k/) |
| **8192 / 65536** | TP=8 + DP-attention | **551.9** @ B=33 | 69.0 | **2.45×** | [`8k64k/`](8k64k/) |

(Offline `bench_one_batch_server`, batch maxed to the KV pool.)

## Why DP-attention wins (both workloads)

- **8K/64K:** decode is **DSA-sparse-attention-bound** (the sparse decode kernel is 83% of decode GPU
  time). DPA runs it as 8 independent per-worker streams instead of TP-sharding the single MLA KV head.
- **1K/8K:** decode is **memory-bandwidth-bound** → throughput scales with batch. The single small SWA
  pool caps pure-TP at batch 146; DPA's 8 per-worker pools reach batch 1016. Per-request rate is
  identical (3.30 vs 3.36 tok/s/req) — the 7.1× is pure batch-scaling, not a compute speedup.

## SM120 required flags (both)

`--tp 8 --dp-size 8 --enable-dp-attention --moe-a2a-backend none --moe-runner-backend triton`
(deepep needs deep_gemm — disabled at sm120; auto→marlin lacks `self.runner` → cuda-graph crash;
`dp_size` must equal `tp_size`). KV auto-locks fp8_e4m3.

**FP4 loses on both** — the mxfp4 MoE GEMV (no fast SM120 kernel) becomes compute-bound at batch
(1K/8K FP4-DPA ≈ 497 tok/s, 6.9× below FP8). See [`../nvfp4/`](../nvfp4/).
