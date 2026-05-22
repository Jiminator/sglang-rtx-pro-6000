# Qwen3.6-35B-A3B-FP8 — Tuning Report

**As of 2026-05-20.** Ship config: **16× single-GPU TP=1 + SMG `round_robin`**. Median TTFT **273 ms**, mean **294 ms**, P99 **368 ms**, total throughput **26,323 tok/s**. See [`results/benchmark_results.md`](results/benchmark_results.md) and [`launch_worker.sh`](launch_worker.sh).

## Model context

`Qwen/Qwen3.6-35B-A3B-FP8` uses **FP8 block-scaled quantization** (W8A8, block_n=128). Total ~35 GB; only **3 B active params per token** (sparse MoE 32-of-8). The model fits on a single GPU at TP=1, so **maximum fan-out (16× TP=1) wins** for TTFT because each replica is barely loaded (conc=10 / 16 replicas = 0.625 active per replica) and per-prefill compute is uniform.

Architecture: hybrid Gated Delta Networks (linear attention) + sparse MoE, native 262K context.

## Workload (TTFT-focused, fixed)

| Field | Value |
|---|---|
| Dataset | random, **50 prompts** |
| ISL (deterministic) | 10,000 tokens (`--random-range-ratio 1.0`) |
| OSL (target) | 500 tokens |
| `--max-concurrency` | 10 |
| `--request-rate` | inf |
| Seed | 1 (default) |
| Bench backend | `--backend sglang-oai` |

This workload **differs from the Qwen3.5 entry** (which uses 200 prompts × conc 40 × ISL 20K). Qwen3.6 was tuned against a smaller interactive-chat target where single-prefill latency dominates.

## Hardware & topology baseline

| Field | Value |
|---|---|
| Cluster | 2× GCP `g4-standard-384`, 16× RTX PRO 6000 Blackwell SM12 |
| Interconnect | 2× 200 Gbps gVNIC TCP, **no NVLink, no RDMA, no PCIe-oneshot AR** |
| Image | `lmsysorg/sglang:dev-cu13` |
| sglang version | `0.0.0.dev1+gedb1b3f8f` |
| `FLASHINFER_DISABLE_VERSION_CHECK` | `1` |
| Parallelism (ship) | **16× single-GPU TP=1 + SMG router** (one replica per GPU) |
| Mandatory env vars | full NCCL/GLOO set + **`NCCL_NCHANNELS=16`** (no-op at TP=1 but harmless) |

## What we shipped

Single-GPU replicas with per-replica fusion levers on (these contribute zero AR work since TP=1, but the QK/MoE/glue fusions still help the single-rank kernel mix):

| Layer | Flag | Mechanism |
|---|---|---|
| 0 | base TP=1 single-GPU | `--tp 1` per replica; SMG router fans out across 16 replicas |
| 1 | `--chunked-prefill-size 16384` + `--max-prefill-tokens 16384` | Large enough to fit 10K-ISL prompt in one chunk |
| 2 | `--mem-fraction-static 0.7` | Conservative; leaves headroom for cuda-graph capture |
| 3 | `--disable-radix-cache` | Random workload has no prefix to match; saves bookkeeping overhead |
| 4 | `--enable-fused-qk-norm-rope` | Closes a glue-layer fusion |
| 5 | `--enable-fused-moe-sum-all-reduce` | Closes another glue-layer fusion (no AR at TP=1 but harmless) |
| 6 | `--enforce-piecewise-cuda-graph` | Captures the 10K prefill as piecewise cuda graphs |
| 7 | `--enable-mixed-chunk` | Allows prefill+decode overlap; noise-level benefit here but harmless |

## TTFT measurements (median, 50 × conc=10)

| Shape | Median TTFT | Notes |
|---|---:|---|
| **16× TP=1 + SMG round_robin (ship)** | **273-275 ms** | Two-run mean ≈ 274 ms; minimum queueing at 0.625 conc/replica |
| 16× TP=1 + `--enable-mixed-chunk` | 273 ms | Within noise of plain 16× TP=1 |
| 8× TP=2 + SMG | 308 ms | 1.25 conc/replica adds ~60 ms queue tail |
| 4× TP=4 + lever stack + SMG | 294 ms | 2.5 conc/replica; worse than 16× TP=1 |
| 1× TP=4 + lever stack (single replica, conc=10) | **1810 ms** | Single replica drowns under conc=10 — queueing-bound |
| 1× TP=2 (single replica, conc=10) | 1781 ms | Same shape |

Single-prefill compute floor (conc=1) on TP=1 = 249 ms; TP=4 = 230 ms. Difference between single-prefill floor and 16× TP=1 ship (273 ms) is ~25 ms of steady-state queueing tail at conc=10.

## torch.profiler trace decomposition — TP=1, single 10K-prefill EXTEND step

(Trace lives in the project bundle at `_raw/qwen36-35ba3b-2026-05-15/16x-tp1-baseline/profile/`.)

| Bucket | GPU ms | % of 235.8 ms wall |
|---|---:|---:|
| FP8 W8A8 dense GEMM (q/k/v/o + MLP) | 52.9 | 22.4 |
| MoE expert GEMM (`fused_moe_kernel`) | 49.7 | 21.1 |
| GDN / mamba kernels | 47.6 | 20.2 |
| flashinfer prefill attention | 27.7 | 11.7 |
| Pointwise/elementwise glue | 18.0 | 7.6 |
| MoE sum-reduce | 10.5 | 4.4 |
| Activation (SwiGLU) | 5.9 | 2.5 |
| FP8 per-token-group quant | 4.8 | 2.0 |
| RMSNorm (both kinds) | 8.8 | 3.7 |
| Aux GEMM, MoE routing | 5.3 | 2.2 |
| GPU idle/sync bubbles | 4.3 | 1.8 |

**98 % GPU-bound** — compute-bound, not Python-bound.

## Closed dimensions — do NOT rerun without new source-code reason

| Lever | Result | Root cause |
|---|---|---|
| `--tp 8` | launch error | `output_size of gate's and up's weight = 64 is not divisible by weight quantization block_n = 128` |
| `--attention-backend fa3` | rejected on SM12 | `project_fa3_rejected_blackwell` |
| `--attention-backend fa4` | rejected on SM12 | `AssertionError: Paged KV not supported on SM 12.0` |
| `--moe-runner-backend flashinfer_trtllm` | crash | sm100 (Hopper) only |
| `--moe-runner-backend flashinfer_cutlass` | crash | `AttributeError: Fp8MoEMethod has no attribute runner` |
| `--moe-runner-backend triton_kernel` | crash | `IndexError: start out of range (expected [-512, 512], got 6144)` in weight loader |
| `--moe-runner-backend deep_gemm` | crash | requires `DeepEPMoE` (expert parallelism) |
| `--enable-torch-compile` | no convergence | Inductor autotune > 40 min on TP=4 (650+ matmul shapes × 19 triton configs × 4 ranks); cuda-graph OOM at mfs=0.7 |
| `--enable-pcie-oneshot-allreduce` | crash | `pcie_allreduce.cu:321 get_graph_buffer_ipc_meta` invalid argument; see `project_pcie_oneshot_ar_broken_on_g4` |
| `--enable-symm-mem` + `--enable-flashinfer-allreduce-fusion` | conflict | `project_symm_mem_dominated_by_fi_ar_fusion` |
| `--pp-async-batch-depth ≥ 1` | deadlock | `project_pp_async_batch_depth_deadlocks` |

## Hardware floor

Per-GPU FP8 dense ~1,500 TFLOPS; pure GEMM work per 10K-token prefill = 60 TFLOPS = **40 ms** ideal compute. Plus GDN scan (~25 ms ideal), flashinfer prefill attn (~20 ms ideal), glue (~25 ms ideal) = **~110 ms single-prefill compute roofline** on this hardware. With 16× fan-out and conc=10, steady-state queueing tail adds ~20 ms → **median TTFT hardware roofline ≈ 130 ms**.

Current ship at 273 ms = **~2.1× the hardware roofline**. The remaining gap is mostly:

- FP8 GEMM at ~40 % of peak (block_n=128 dequant overhead; better MoE backends all crashed)
- GDN scan at ~2× ideal (no fused mamba-scan kernel ships for Blackwell SM12)
- Glue ops not fully fused (torch.compile autotune doesn't converge in budget)

Sub-100 ms TTFT requires hardware changes (NVLink for AR, NVFP4 checkpoint, or smaller model) — see `_raw/qwen36-35ba3b-2026-05-15/REPORT.md` for the impossibility proof.

## Production bundle

Full deployment recipe + impossibility-proof artifacts live in the project repo at `_raw/qwen36-35ba3b-2026-05-15/`. This directory in `sglang-rtx-pro-6000/` is the radixark-bench-style shipping recipe.
