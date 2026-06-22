# MXFP4 1K/8K batch-scaling sweep — raw results

Workload: ISL 1024 / OSL 8192, offline, `bench_one_batch_server`, `--dataset-name random`.
Metric: steady-state decode `gen throughput (token/s)` per DP rank × 8 ranks (decode-dominant at
OSL 8192; read once the prefill ramp drains). All configs: FP4+DPA, TP=8, dp_size=8,
`--moe-runner-backend marlin` (→ sm120_triton GEMV), KV fp8_e4m3, attn=dsv4.

| EXP | ratio | mfs | max_total_num_tokens | decode req/rank | gen_tps/rank | **AGG tok/s** | /GPU | ramp-to-plateau |
|---|---|---|---|---|---|---|---|---|
| bsv2_r10 | 0.10 | 0.85 | 3,432,960 | 180 | 61.37 | 490.96 | 61.4 | ~52 min |
| **bsv2_r15** | **0.15** | **0.90** | **2,719,744** | **214** | **62.20** | **497.60** | **62.2** | ~64 min |
| bsv2_r20 | 0.20 | 0.90 | 2,133,504 | 224 | 61.49 | 491.92 | 61.5 | ~68 min |

**Conclusion:** flat (~491–498, ±1.4% = noise). Batch-scaling saturated; per-rank rate ~61–62 tok/s
independent of batch → MoE compute-bound. Max = **497.60 @ 214 req/rank (bsv2_r15)**.

`max_running_requests` force-capped at 256/rank by `DeepseekV4ForCausalLM` (so the 224/rank config is
near the hard cap and still doesn't beat 214/rank). Raising `--swa-full-tokens-ratio` shrinks
`max_total_num_tokens` (more of the hybrid pool goes to the SWA partition) but the binding constraint is
MoE compute, not pool size.

Per-rank decode log excerpt (winner, r0.15): see `srv_decode_excerpt_r15.txt` — e.g.
`Decode batch, #running-req: 214, ... gen throughput (token/s): 62.20`.

Earlier under-ramped reads (discarded): a 300-s window caught only 16 req/rank @ 35 tok/s (282 agg) and
a 37-min window caught pure prefill (0 decode lines) — both artifacts of the slow FP4 prefill ramp, not
real plateaus.
