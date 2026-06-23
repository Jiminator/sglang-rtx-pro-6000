# DeepSeek-V4-Flash (MXFP4 experts) on SM120

`deepseek-ai/DeepSeek-V4-Flash` — the public checkpoint. Routed experts are **MXFP4**
(E2M1 + E8M0 group-32 block scales); attention, shared experts, and MTP are FP8.

> Not to be confused with `nvidia/DeepSeek-V4-Flash-NVFP4` (genuine NVFP4, group-16 E4M3 scales) —
> see [`../nvfp4/`](../nvfp4/), which has **no working MoE backend on SM120**. This folder is the
> MXFP4 checkpoint, which **does** run correctly on SM120 (gsm8k 0.900) but is MoE-bound.

## Verdict — depends on image version

- **On a #28231 image (`dev-cu13` or newer): MXFP4 is GOOD.** Upstream commit #28231 "Use Marlin for
  SM120 MXFP4 MoE" routes SM120 MXFP4 through the Marlin kernel → **~3573 tok/s (447/GPU), gsm8k 1.000**
  at 1K/8K, which edges past the FP8 checkpoint (3413). **Serve MXFP4 on a #28231 image** — no patch.
- **On the older pinned `v0.5.13.post1`: MXFP4 is the loser** — its only SM120 MXFP4 path is a slow
  per-slot triton GEMV (~497 tok/s, 6.9× below FP8). On that image, ship [`../fp8/`](../fp8/).

| Workload | MXFP4 (v0.5.13.post1, GEMV) | MXFP4 (#28231, Marlin) | FP8 (v0.5.13.post1) |
|---|---|---|---|
| 1K/8K offline | 497.6 (62/GPU) | **~3573 (447/GPU)** | 3412.9 (427/GPU) |

See [`1k8k/REPORT.md`](1k8k/REPORT.md) and [`1k8k/results/upstream_marlin_28231/`](1k8k/results/upstream_marlin_28231/).

## Root cause (one line)

`v0.5.13.post1`'s SM120 MXFP4 MoE was a per-(token,expert)-slot triton GEMV (`_mxfp4_slot_gemv_kernel`)
that reloaded each expert's weights per token → weight-bandwidth-bound (prefill ~64, decode ~62
tok/s/rank). Upstream #28231 replaced it with the **Marlin** tensor-core grouped GEMM (the per-slot file
was deleted), fixing both prefill (~459 tok/s/rank) and decode (~446/rank). An independent grouped-triton
GEMM experiment confirmed the same direction (5.0×) before #28231 was found; Marlin is faster.

## Contents
- [`1k8k/`](1k8k/) — 1K/8K: the upstream-Marlin recommendation (~3573 tok/s) + the pinned-image
  launch-space study (497.6 tok/s) + the grouped-triton experiment notes.
