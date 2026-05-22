# Qwen3.5-397B-A17B-FP8 Benchmarks

Latency oriented benchmarks for the **Qwen3.5-397B-A17B-FP8** model. These benchmarks focus on Time to First Token (TTFT) and Inter-Token Latency (ITL) rather than raw throughput.

## Benchmark Results (Qwen3.5-397B-A17B-FP8)

| Configuration | Mean TTFT | P99 TTFT | Mean TPOT | Median ITL |
| :--- | :---: | :---: | :---: | :---: |
| **HiCache (Enabled)** | **1,121.17 ms** | **3,359.76 ms** | 100.59 ms | 37.36 ms |
| **No Radix Cache** | 1,371.28 ms | 14,052.80 ms | **90.45 ms** | **36.25 ms** |

See the [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) for detailed results and criteria.
