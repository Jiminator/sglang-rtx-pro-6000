# Qwen3.5-397B-A17B-FP8 Latency Benchmark Report

This document outlines the performance characteristics of the **Qwen3.5-397B-A17B-FP8** model, focusing specifically on latency-oriented metrics (TTFT, TPOT, ITL). The benchmarks compare performance with Hierarchical Cache (HiCache) enabled versus a configuration with Radix Cache disabled.

## Benchmarking Criteria

### Model Details
- **Model Name:** Qwen/Qwen3.5-397B-A17B-FP8
- **Precision:** FP8
- **Architecture:** Mixture-of-Experts (MoE)

### Hardware & Software Environment
- **Platform:** GKE (Google Kubernetes Engine)
- **Node Type:** single-node with 8x GPUs
- **Framework:** SGLang (vLLM-compatible runtime)
- **Image:** `lmsysorg/sglang:dev-cu13`
- **Tensor Parallelism (TP):** 8

### Server Configuration
Common SGLang parameters used across runs:
- `--tp 8`
- `--chunked-prefill-size 4096`
- `--max-prefill-tokens 32768`
- `--enable-mixed-chunk`
- `--enable-fused-qk-norm-rope`
- `--enable-fused-moe-sum-all-reduce`
- `--enable-flashinfer-allreduce-fusion`
- `--enforce-piecewise-cuda-graph`
- `--mem-fraction-static 0.8`

### Traffic Profile
- **Total Requests:** 2000 (successful)
- **Max Concurrency:** 40
- **Total Input Tokens:** ~19.7M
- **Total Generated Tokens:** ~986K

---

## Results Summary (Qwen3.5-397B-A17B-FP8)

The following table summarizes the latency metrics for the two tested configurations for the **Qwen3.5-397B-A17B-FP8** model.

| Metric | HiCache (Enabled) | No Radix Cache |
| :--- | :---: | :---: |
| **Mean TTFT (ms)** | **1,121.17** | 1,371.28 |
| **Median TTFT (ms)** | **1,054.01** | 1,128.88 |
| **P99 TTFT (ms)** | **3,359.76** | 14,052.80 |
| **Mean TPOT (ms)** | 100.59 | **90.45** |
| **Median TPOT (ms)** | 101.18 | **90.41** |
| **P99 TPOT (ms)** | 165.27 | **140.58** |
| **Mean ITL (ms)** | 100.03 | **89.97** |
| **Median ITL (ms)** | 37.36 | **36.25** |

### Key Observations

1.  **Time to First Token (TTFT):**
    -   HiCache significantly improves TTFT, especially at the tail (P99). The P99 TTFT dropped from **14.05 seconds** to **3.36 seconds** with HiCache enabled.
    -   Mean TTFT saw a ~18% improvement with HiCache.

2.  **Time per Output Token (TPOT) & ITL:**
    -   The "No Radix Cache" configuration showed slightly better TPOT and Inter-Token Latency (ITL). This is expected as HiCache and Radix Cache management introduce some overhead during the generation phase, though the trade-off favors HiCache for large-scale prefill scenarios.

3.  **End-to-End Latency:**
    -   Mean E2E latency was slightly higher with HiCache (~50s vs ~45s), likely due to the slightly higher TPOT accumulated over longer sequences, despite the much faster TTFT.

## Detailed Metrics

### 1. Hierarchical Cache (HiCache) Enabled
*Configuration: `--enable-hierarchical-cache --hicache-ratio=2.0 --hicache-io-backend=kernel`*

- **Mean TTFT:** 1121.17 ms
- **P99 TTFT:** 3359.76 ms
- **Mean TPOT:** 100.59 ms
- **Mean ITL:** 100.03 ms
- **P99 ITL:** 393.16 ms

### 2. No Radix Cache
*Configuration: `--disable-radix-cache`*

- **Mean TTFT:** 1371.28 ms
- **P99 TTFT:** 14052.80 ms
- **Mean TPOT:** 90.45 ms
- **Mean ITL:** 89.97 ms
- **P99 ITL:** 337.00 ms
