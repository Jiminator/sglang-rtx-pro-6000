# Qwen3.6-35B-A3B-FP8 — Benchmark Results

## Headline (ship: 16× TP=1 + SMG round_robin)

| Metric | Value |
|---|---:|
| Median TTFT | **273 ms** |
| Mean TTFT | 294 ms |
| P99 TTFT | 368 ms |
| Median TPOT | 7.35 ms |
| Total throughput | 26,323 tok/s |
| Successful requests | 50 / 50 |

Reproduced 273 / 275 ms across two independent runs on 2026-05-19 (within 1 %).

## Workload

| Field | Value |
|---|---|
| `--dataset-name` | random |
| `--num-prompts` | 50 |
| `--max-concurrency` | 10 |
| `--random-input-len` | 10000 |
| `--random-output-len` | 500 |
| `--random-range-ratio` | 1.0 (deterministic ISL/OSL) |
| `--request-rate` | inf |
| Backend | sglang-oai |

## Full bench output

```
Backend:                                 sglang-oai
Traffic request rate:                    inf
Max request concurrency:                 10
Successful requests:                     50
Benchmark duration (s):                  19.94
Total input tokens:                      500000
Total generated tokens:                  25000
Request throughput (req/s):              2.51
Input token throughput (tok/s):          25069.64
Output token throughput (tok/s):         1253.48
Peak output token throughput (tok/s):    1368.00
Peak concurrent requests:                20
Total token throughput (tok/s):          26323.12
Concurrency:                             9.92
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   3955.86
Median E2E Latency (ms):                 3947.73
P90 E2E Latency (ms):                    4013.92
P99 E2E Latency (ms):                    4079.67
---------------Time to First Token----------------
Mean TTFT (ms):                          293.72
Median TTFT (ms):                        272.72
P99 TTFT (ms):                           367.85
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          7.34
Median TPOT (ms):                        7.35
P99 TPOT (ms):                           7.47
---------------Inter-Token Latency----------------
Mean ITL (ms):                           7.40
Median ITL (ms):                         7.37
P95 ITL (ms):                            7.49
P99 ITL (ms):                            7.65
Max ITL (ms):                            74.12
==================================================
```

## TTFT comparison across parallelism shapes (same workload)

| Shape | Median TTFT | Notes |
|---|---:|---|
| **16× TP=1 + SMG (ship)** | **273-275 ms** | Best observed — 0.625 conc/replica, minimum queueing |
| 16× TP=1 + `--enable-mixed-chunk` | 273 ms | Within noise |
| 8× TP=2 + SMG | 308 ms | +14 % from queueing |
| 4× TP=4 + lever stack + SMG | 294 ms | +8 % from queueing |
| 1× TP=4 + lever stack | 1810 ms | Single replica drowns conc=10 |
| 1× TP=2 + lever stack | 1781 ms | Same shape |
| Single-prefill TP=1 (conc=1) | 249 ms | Compute floor on single GPU |

## Bench command

```bash
python3 -m sglang.bench_serving \
  --backend sglang-oai \
  --base-url http://<router>:8000 \
  --model Qwen/Qwen3.6-35B-A3B-FP8 \
  --tokenizer Qwen/Qwen3.6-35B-A3B-FP8 \
  --dataset-name random \
  --num-prompts 50 \
  --max-concurrency 10 \
  --random-input-len 10000 \
  --random-output-len 500 \
  --random-range-ratio 1.0
```
