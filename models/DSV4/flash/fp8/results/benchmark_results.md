# DeepSeek-V4-Flash FP8 — benchmark results (true 8K/64K)

Offline `sglang.bench_one_batch_server`, ISL 8192 / OSL 65536, single node (8× RTX PRO 6000 SM120),
`lmsysorg/sglang:v0.5.13.post1-cu130`, sglang v0.5.13.post1.

## Summary

| Config | batch | Output tok/s | /GPU | ITL | E2E latency (s) |
|---|---:|---:|---:|---:|---:|
| **FP8 + DP-attention** (ship) | 33 | **551.92** | **69.0** | ~60 ms | 3,998.6 |
| FP8 pure-TP (baseline) | 38 | 225.45 | 28.2 | ~169 ms | 11,581.6 |

**DP-attention = 2.45× over pure TP=8.** Raw per-run JSONLs below.

## Raw — FP8 + DP-attention @ batch 33 (`results_dpa_true_b33.jsonl`)

```json
{"run_name": "default", "batch_size": 33, "input_len": 8192, "output_len": 65536, "latency": 3998.5722, "input_throughput": 3374.0, "output_throughput": 551.92, "overall_throughput": 608.47, "last_ttft": 80.1232, "last_gen_throughput": 66.18, "acc_length": -1.0, "cache_hit_rate": null}
```

## Raw — FP8 pure-TP @ batch 38 (`results_tp_true_b38.jsonl`)

```json
{"run_name": "default", "batch_size": 38, "input_len": 8192, "output_len": 65536, "latency": 11581.5906, "input_throughput": 581.67, "output_throughput": 225.45, "overall_throughput": 241.91, "last_ttft": 535.1799, "last_gen_throughput": 220.75, "acc_length": -1.0, "cache_hit_rate": null}
```

**Attribution**: tuning and benchmark execution by **Jimmy Shong** (RadixArk).
