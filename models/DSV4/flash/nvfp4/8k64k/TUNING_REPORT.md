# DeepSeek-V4-Flash — Tuning Report (8K/64K, single node, SM120, latest main)

**As of 2026-07-04.** Workload **ISL 8192 / OSL 65536**, no-zipfian (`--random-range-ratio 1.0`), online
**`sglang.bench_serving`** loadgen → steady-state decode plateau from server logs. **Latest-main
`lmsysorg/sglang:dev-cu13` + the two SM120 env vars** (see the 1k/8k report / launch script).

## Headline

**FP4 checkpoints ~1935 output tok/s (~242/GPU) @ 64 running-req/rank — 3.5× the prior offline anchor
(551.9).** MXFP4-Marlin and NVFP4 tie; FP8 ~3% behind. See [`launch_NVFP4_dpa_8k64k.sh`](launch_NVFP4_dpa_8k64k.sh).

## Results (bench_serving steady-state, ISL 8192 / OSL 65536, CONC=512 = ~ceiling)

| Checkpoint | MoE runner | plateau /rank | **agg tok/s** | /GPU |
|---|---|---:|---:|---:|
| deepseek-ai/DeepSeek-V4-Flash (MXFP4, #28231) | marlin | 64 @ 242.6 | **~1941** | 243 |
| **nvidia/DeepSeek-V4-Flash-NVFP4** | flashinfer_cutlass | 64 @ 241.7 | **~1934** | 242 |
| sgl-project/DeepSeek-V4-Flash-FP8 | triton | 64 @ 235.1 | ~1881 | 235 |

MXFP4≈NVFP4 (tie within 0.4%); both ~3% over FP8. The FP4 lead over FP8 is bigger here (~3%) than at 1k/8k
(~1.6%) — smaller experts help more over 64K decode steps. Prior offline anchor: FP8+DPA 551.9 @ B=33.

## Concurrency sweep (the 8K/64K measurement subtlety)

At OSL 65536 the 8192-token **prefill** backlog dominates at high concurrency (inf request-rate). CONC=1024
(128/rank offered) goes **prefill-swamped** — the server admits only ~70/rank and never reaches a sustained
decode plateau. The clean steady-state ceiling is **~64 running-req/rank (CONC=512)**; above ~70/rank prefill
saturates. Measure at CONC=512.

| CONC | /rank | FP8 | NVFP4 | MXFP4 |
|---:|---:|---:|---:|---:|
| 256 | 32 | 1658 | 1730 | 1742 |
| 384 | 48 | 1772 | 1831 | 1843 |
| **512** | **64** | **1881** | **1934** | **1941** |
| 1024 | 128 | prefill-swamped (no decode plateau) | | |

Throughput keeps rising to 64/rank (diminishing: per-req rate falls, BW-bound decode). ~64/rank is the
practical ceiling on this hardware.

## Config

Identical recipe to 1k/8k (TP=8+DPA, 2 SM120 env vars, fp8_e4m3 KV, mfs 0.85, `--moe-a2a-backend none`),
except **`--cuda-graph-max-bs 64`** and **`--context-length 73728`** for the long context. Bench at
`--max-concurrency 512`. Correctness: same decode/prefill/MoE kernels as 1k/8k (gsm8k 0.960); the 64K path
uses no new kernels, so it is not separately gated (gsm8k is short-context). See launch script.
