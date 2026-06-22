# DeepSeek-V4-Flash FP8 — benchmark results (1K/8K)

Offline `sglang.bench_one_batch_server`, **ISL 1024 / OSL 8192**, batch maxed to the SWA pool per
config. Single node (8× RTX PRO 6000 SM120), `lmsysorg/sglang:v0.5.13.post1-cu130`, sglang v0.5.13.post1.

## Summary

| Config | batch | output tok/s | /GPU | tok/s/req | E2E lat (s) |
|---|---:|---:|---:|---:|---:|
| **FP8 + DP-attention** (ship) | 1016 | **3412.91** | 426.6 | 3.36 | 2688.9 |
| FP8 + DPA, swa-ratio 0.16 | 1120 | 3357.92 | 419.7 | 3.00 | 3008.0 |
| FP8 + DPA, swa-ratio 0.20 | 1160 | 3336.72 | 417.1 | 2.88 | 3135.9 |
| FP8 2× TP=4 (sum of 2 replicas) | 2×36 | 678.05 | 84.8 | 9.41 | ~930 |
| FP8 TP=8 pure | 146 | 481.36 | 60.2 | 3.30 | 2727.8 |
| FP4 TP=8 (marlin) | 185 | 266.65* | 33.3 | 1.44 | (steady-state) |
| FP4 DP-attention (marlin) | ~1336 | ~497* | ~62 | 0.37 | (steady-state) |

*FP4 numbers are steady-state aggregate decode rate from the server log (FP4 benches exceeded the
100-min cap due to the slow mxfp4 MoE kernel); both ran 0 retractions.

**DP-attention = 7.1× over pure TP=8** — batch-scaling on a BW-bound decode (per-request rate is
identical, 3.30 vs 3.36; DPA's 8 per-worker SWA pools lift the batch ceiling 146→1016).

## Raw — FP8 + DP-attention @ batch 1016 (winner)

```json
{"run_name": "default", "batch_size": 1016, "input_len": 1024, "output_len": 8192, "latency": 2688.9099, "input_throughput": 4158.13, "output_throughput": 3412.91, "overall_throughput": 3482.25, "last_ttft": 250.2049, "last_gen_throughput": 425.93, "acc_length": -1.0, "cache_hit_rate": null}
```

## Raw — FP8 TP=8 pure @ batch 146

```json
{"run_name": "default", "batch_size": 146, "input_len": 1024, "output_len": 8192, "latency": 2727.7809, "input_throughput": 615.04, "output_throughput": 481.36, "overall_throughput": 493.27, "last_ttft": 243.0794, "last_gen_throughput": 478.15, "acc_length": -1.0, "cache_hit_rate": null}
```

Per-config JSONLs are in this directory (`results_*.jsonl`).

**Attribution**: tuning and benchmark execution by **Jimmy Shong** (RadixArk).
