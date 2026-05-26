# Qwen3.5-397B-A17B-FP8 Benchmarks

Latency oriented benchmarks for the **Qwen3.5-397B-A17B-FP8** model. These benchmarks focus on Time to First Token (TTFT) and Inter-Token Latency (ITL) rather than raw throughput.

 - ISL -  20K
 - OSL - 1K
 - Concurrency - 40

## Benchmark Results (Qwen3.5-397B-A17B-FP8)

| Configuration | Metric | Mean | Median | P99 |
| :--- | :--- | :---: | :---: | :---: |
| **HiCache (Enabled)** | **TTFT (ms)** | **1,121.17** | **1,054.01** | **3,359.76** |
| | **TPOT (ms)** | 100.59 | 101.18 | 165.27 |
| | **ITL (ms)** | 100.03 | 37.36 | 393.16 |
| **No Radix Cache** | **TTFT (ms)** | 1,371.28 | 1,128.88 | 14,052.80 |
| | **TPOT (ms)** | **90.45** | **90.41** | **140.58** |
| | **ITL (ms)** | **89.97** | **36.25** | **337.00** |

See the [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) for detailed results and criteria.
