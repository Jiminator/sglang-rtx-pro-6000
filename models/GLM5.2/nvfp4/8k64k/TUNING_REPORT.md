# GLM-5.2-NVFP4 — Tuning Report (8K/64K, single node, SM120, latest main)

**As of 2026-07-05.** Workload **ISL 8192 / OSL 65536**, no-zipfian (`--random-range-ratio 1.0`), online
**`sglang.bench_serving`** loadgen → steady-state decode plateau from server logs (aggregate = per-rank
gen-throughput × dp_size 8). Checkpoint `nvidia/GLM-5.2-NVFP4` (`GlmMoeDsa`), snapshot `b0b2b68`. Image:
**stock latest-main `lmsysorg/sglang:dev-cu13`** (sglang `0.0.0.dev1+gb28bc1060`, transformers 5.12.1) + the
one fp8 env var (`SGLANG_DISABLE_DSA_INDEXER_FUSION=1`; see the 1k/8k report / launch script).

## Headline

**fp8 KV = ~1081 output tok/s (~135/GPU) @ 16 running-req/rank (CONC=128), +10% over bf16 (~984, 123/GPU).**
See [`launch_glm52_nvfp4_8k64k.sh`](launch_glm52_nvfp4_8k64k.sh).

## Results (bench_serving steady-state, ISL 8192 / OSL 65536, CONC=128)

| Config | KV | mfs | pool/rank | CONC | running/rank | agg tok/s | /GPU |
|---|---|---:|---:|---:|---:|---:|---:|
| bf16 baseline | bfloat16 | 0.94 | 91.5K | 128 | 9 | ~984 | 123 |
| **fp8** 🥇 | fp8_e4m3 | 0.97 | 220.5K | 128 | 16 | **~1081** | **135** |

fp8's ~2.4× larger pool (220.5K vs 91.5K tokens/rank) lifts the concurrency ceiling from **9 running-req/rank
(bf16, pool-bound at 8192-token prefixes) to 16 (fp8)** at the same offered CONC=128 — that admission-ceiling
lift is the entire +10%. The win is smaller than 1k/8k's +108% because at 64K each surviving sequence eats
far more pool, so the ~2.4× density buys fewer additional concurrent sequences.

## Why 8K/64K is subtle: pool-bound AND prefill-sensitive

GLM full-DSA long context is bound on two axes at once — the KV pool caps concurrency, and the 8192-token
**prefill** backlog swamps the scheduler at high offered load. The clean steady-state window is CONC~128:

| CONC | offered/rank | outcome | /GPU (fp8) |
|---:|---:|---|---:|
| 96 | 12 | underfilled (pool has headroom) | ~108 |
| 128 | 16 | clean decode plateau | **~135** |
| 160 | 20 | **prefill-swamps** (8192-token backlog dominates; no sustained decode) | — |

Above ~160 the prefill queue dominates and the server never reaches a sustained decode plateau. Measure at
CONC~128.

## Measurement caveat

At OSL 65536, sequences **never complete in-window** — so the reported plateau reflects the *growing* phase
(sequences accumulating KV toward the pool ceiling), i.e. a lower-bound-ish steady state rather than a true
completed-request throughput. This is the same growing-sequence caveat as the DSV4 8K/64K bundle.

## Correctness

Not separately gated at 8K/64K: it uses the **same decode/prefill/MoE kernels** as 1k/8k (gsm8k 0.900 fp8 /
0.940 spec), and gsm8k is a short-context benchmark. The 64K path introduces no new kernels.

## Config

Identical recipe to 1k/8k (TP=8+DPA, `SGLANG_DISABLE_DSA_INDEXER_FUSION=1`, fp8_e4m3 KV, mfs 0.97,
flashinfer_cutlass MoE, `--disable-shared-experts-fusion`, `--chunked-prefill-size 2048`), except
**`--context-length 73728`**, **`--cuda-graph-bs "4 8 12 16 24"`**, **`--cuda-graph-max-bs 32`** for the long
context. Bench at `--max-concurrency 128 --num-prompts 256`. Full run data:
`runs/20260705_glm5.2_sota_humanize/` in the gcp-kimi repo.
