# DeepSeek-V4-Flash (MXFP4 experts) on SM120

`deepseek-ai/DeepSeek-V4-Flash` — the public checkpoint. Routed experts are **MXFP4**
(E2M1 + E8M0 group-32 block scales); attention, shared experts, and MTP are FP8.

> Not to be confused with `nvidia/DeepSeek-V4-Flash-NVFP4` (genuine NVFP4, group-16 E4M3 scales) —
> see [`../nvfp4/`](../nvfp4/), which has **no working MoE backend on SM120**. This folder is the
> MXFP4 checkpoint, which **does** run correctly on SM120 (gsm8k 0.900) but is MoE-bound.

## Verdict

Runs correctly on SM120 but is **MoE-compute-bound** — the FP8 checkpoint is far faster at every
workload. **Ship [`../fp8/`](../fp8/) for production.** This folder documents the best achievable with
the MXFP4 weights and **no source-code changes**.

| Workload | MXFP4 best | FP8 best | gap |
|---|---|---|---|
| 1K/8K offline | **497.6 tok/s** (62.2/GPU) — [`1k8k/`](1k8k/) | 3412.9 (427/GPU) | 6.9× |

## Root cause (one line)

The only working SM120 MXFP4 MoE kernel is a per-(token,expert)-slot triton GEMV
(`_mxfp4_slot_gemv_kernel`) that reloads each expert's weights per token — weight-bandwidth-bound, so
both prefill (~64 tok/s/rank) and decode (~62 tok/s/rank aggregate ×8 = ~497, flat vs batch) are slow.
All faster FP4 MoE paths require SM100 (TMEM/tcgen05). No framework (SGLang, vLLM, TRT-LLM) has a fast
SM120 MXFP4 MoE kernel — the gap is external. The order-of-magnitude lever is a grouped MXFP4 GEMM
(source change, out of scope here).

## Contents
- [`1k8k/`](1k8k/) — 1K/8K offline batch-scaling sweep + the shipped max config (497.6 tok/s).
