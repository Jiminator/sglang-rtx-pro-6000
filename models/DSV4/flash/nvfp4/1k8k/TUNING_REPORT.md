# DeepSeek-V4-Flash — Tuning Report (1K/8K, single node, SM120, latest main)

**As of 2026-07-04.** Workload **ISL 1024 / OSL 8192**, no-zipfian (`--random-range-ratio 1.0`), online
**`sglang.bench_serving`** as load generator → steady-state decode plateau read from server logs (at OSL
8192 requests don't complete in-window, so #running-req pins at the admission ceiling; aggregate =
per-rank gen-throughput × dp_size 8). **Latest-main `lmsysorg/sglang:dev-cu13`** (has #28231). Correctness
gate: gsm8k 50q on the idle server before load.

## Headline

**NVFP4 + DP-attention = ~3753 output tok/s (~469/GPU) @ 256 running-req/rank, gsm8k 0.960.** All three
DSV4-Flash checkpoints tie within ~1.6% (near-noise); NVFP4 is the top and the smallest checkpoint (157 GB).
The big result vs the prior campaign is **MXFP4 via upstream #28231 (SM120→Marlin) = 7.4× the old per-slot
GEMV (497 tok/s).** See [`launch_NVFP4_dpa_1k8k.sh`](launch_NVFP4_dpa_1k8k.sh).

## Results (bench_serving steady-state, ISL 1024 / OSL 8192, latest-main dev-cu13 + 2 SM120 env vars)

| Checkpoint | MoE runner | gsm8k | plateau /rank | **agg tok/s** | /GPU |
|---|---|---:|---:|---:|---:|
| **nvidia/DeepSeek-V4-Flash-NVFP4** | flashinfer_cutlass | 0.960 | 256 @ 469.1 | **~3753** | **469** |
| deepseek-ai/DeepSeek-V4-Flash (MXFP4, #28231) | marlin | 0.960 | 256 @ 464.6 | ~3717 | 465 |
| sgl-project/DeepSeek-V4-Flash-FP8 | triton | 0.960 | 256 @ 461.8 | ~3694 | 462 |

Prior offline (`bench_one_batch_server`) anchor was FP8+DPA 3412.9 @ B=1016 — the new online numbers are
+8–10% and confirm the anchor. MXFP4 old GEMV path was 497 → **#28231 Marlin = ~3717 = 7.4×.**

## The two SM120 env vars (mandatory on latest main — no source patch)

Latest-main sglang is **doubly-broken** for DeepSeek-V4 on SM120 (RTX PRO 6000); both are fixed by
maintainer-supported env vars:

| Blocker | Cause | Fix |
|---|---|---|
| Decode crash | `flash_mla_sm120.py` imports `flashinfer.mla._sparse_mla_sm120`, renamed away in flashinfer 0.6.12 | `SGLANG_SM120_FLASHMLA_BACKEND=triton` (in-tree SM120 sparse-MLA decode) |
| Prefill crash | AOT `sgl_kernel.sparse_prefill_fwd` is SM90a/SM100f-only; sparse-prefill env defaults on | `SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=0` (dense SM120 prefill; chunk 1024/fwd < 11673 gate) |

A flashinfer shim is infeasible (the 0.6.12 replacement is an SM100-only TRTLLM-GEN kernel). No stock image
avoids these for the #28231 MXFP4 path; `v0.5.13.post1-cu130` runs FP8/NVFP4 decode without them but is
pre-#28231 (MXFP4 = slow GEMV). FP8 1k/8k is identical on both images (~3610) — cross-image parity confirmed.

## Why they tie / where the ceiling is

- **256/rank is the ceiling** (= `--max-running-requests 2048` ÷ dp_size 8). Decode is memory-bandwidth-bound
  (loading MoE expert weights each step); per-request rate at 256/rank is only ~1.83 tok/s (past the BW knee).
  Going 192→256/rank added just +2%; pushing MAXRUN to 384/rank goes **prefill-bound** with no decode gain.
- **Pool is NOT the binder** (full-token usage 0.05, swa usage ≤0.65, ~10 GB/GPU free) → `--mem-fraction-static`
  is not a useful lever here. The three checkpoints differ only in MoE-expert dtype; since decode is
  weight-bandwidth-bound, the smaller-expert FP4 checkpoints edge out FP8 by ~1.6% (compounds to ~4% at 8k/64k).
- **KV dtype axis CLOSED**: DSV4 hard-locks `fp8_e4m3` (`deepseek_v4_hook.py` assertion; bf16 rejected).
- Spec decode is a net loss on DSV4 (EAGLE topk-1 only); not used.

## Config (ship = NVFP4; FP8/MXFP4 are drop-in alternates — see launch script footer)

TP=8 + DP-attention, `--moe-a2a-backend none`, `--moe-runner-backend flashinfer_cutlass` (NVFP4),
`--kv-cache-dtype fp8_e4m3`, `--mem-fraction-static 0.85`, `--cuda-graph-max-bs 384`,
`--context-length 9472`, `--max-running-requests 2048`, + the two SM120 env vars + the NCCL/GLOO block.
Full run data: `runs/20260704_dsv4_flash_sota_humanize/benchmark/results.md` in the gcp-kimi repo.
