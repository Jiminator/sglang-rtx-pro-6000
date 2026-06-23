# DeepSeek-V4-Flash (MXFP4) — 1K/8K offline

**Checkpoint:** `deepseek-ai/DeepSeek-V4-Flash` — routed experts **MXFP4** (E2M1 + E8M0 group-32);
attention / shared-experts / MTP in FP8. **HW:** 1× g4, 8× RTX PRO 6000 (SM120). **Workload (fixed):**
ISL 1024 / OSL 8192, offline, `bench_one_batch_server`, batch maxed.

> ## ⭐ TL;DR — UPGRADE THE IMAGE (2026-06-22)
> The "497 tok/s ceiling" below was specific to the **pinned `v0.5.13.post1`** image, whose only working
> SM120 MXFP4 MoE path is a slow per-slot triton GEMV. **Upstream commit #28231 "Use Marlin for SM120
> MXFP4 MoE" (merged ~2026-06-19) fixes this**: it routes SM120 MXFP4 through the Marlin kernel. Measured
> on a fresh `lmsysorg/sglang:dev-cu13` (commit g6779ca8d7, contains #28231) on our RTX PRO 6000:
>
> | path | image | aggregate | /GPU | gsm8k | vs GEMV |
> |---|---|---|---|---|---|
> | per-slot GEMV | v0.5.13.post1 | 497.6 | 62 | 0.90 | 1× |
> | grouped-triton GEMM (our experiment, reverted) | v0.5.13.post1 +patch | 2491.9 | 311 | 0.95 | 5.0× |
> | **Marlin (upstream #28231)** | **dev-cu13** | **~3573** | **447** | **1.000** | **7.2×** |
>
> Marlin saturates at 128 req/rank (flat ~3543–3573 across ratio 0.15→0.30; raising the SWA pool doesn't
> help). **It edges past the FP8 checkpoint's 3413 — MXFP4 is no longer the loser.** **Recommendation:
> serve DSV4-Flash MXFP4 on a #28231 image** (`dev-cu13` or newer). No source patch needed. Evidence:
> `results/upstream_marlin_28231/`. The rest of this report documents the pinned-image launch-space study.

---

**Pinned-image study below — `lmsysorg/sglang:v0.5.13.post1-cu130`, NO source-code changes.**

## Result

| Config | `swa-full-tokens-ratio` / `mem-fraction-static` | decode batch | gen/rank | **Aggregate tok/s** | /GPU |
|---|---|---|---|---|---|
| FP4+DPA r0.10 | 0.10 / 0.85 | 180/rank (1440) | 61.37 | 490.96 | 61.4 |
| **FP4+DPA r0.15 (shipped)** | **0.15 / 0.90** | **214/rank (1712)** | **62.20** | **497.60** | **62.2** |
| FP4+DPA r0.20 | 0.20 / 0.90 | 224/rank (1792) | 61.49 | 491.92 | 61.5 |

**Max = 497.6 tok/s aggregate (62.2/GPU) @ 214 req/rank.** Measured as steady-state decode
`gen throughput × 8 ranks` once the prefill ramp drains (decode-dominant at OSL 8192). Reproduces the
prior log-derived ~497.

## Headline finding: batch-scaling is saturated

Aggregate throughput is **flat at ~491–498 tok/s across 180→224 req/rank** (a ~1.4% spread = noise).
Per-rank gen throughput is pinned at ~61–62 tok/s regardless of batch — textbook MoE **compute
saturation**. Enlarging the SWA pool (`--swa-full-tokens-ratio`, the only batch lever since
`max_running_requests` is force-capped at 256/rank by `DeepseekV4ForCausalLM`) admits more concurrent
requests but **buys no throughput**. The batch lever is exhausted.

## Why ~497 is the ceiling (no-source-edit launch space)

The only working SM120 MXFP4 MoE kernel is the per-(token,expert)-slot triton GEMV
`_mxfp4_slot_gemv_kernel` (`fused_moe_triton/mxfp4_moe_sm120_triton.py`), reached via
`--moe-runner-backend marlin` → `_dsv4_mxfp4_backend="sm120_triton"`. It **reloads each expert's weight
block for every token routed to it** — zero weight reuse across the token dimension → weight-bandwidth-
bound. The same kernel bottlenecks **prefill too** (~64 tok/s/rank input throughput; the prefill ramp to
214/rank takes ~50 min). All faster FP4 MoE paths are unavailable on SM120 (no TMEM/tcgen05):
`flashinfer_mxfp4` SM90/100-only, marlin **CUDA** mxfp4 NaNs, `deepep`/MegaMoE need deep_gemm
(hard-disabled at `sm_version==120`).

Cross-framework gate: **no framework has a fast SM120 MXFP4 MoE kernel** — vLLM falls back to Marlin
W4A16 on SM120 (its MegaMoE/humming/trtllm fast paths gate on SM100; the only fast SM120 4-bit path,
FlashInfer b12x, is NVFP4-only), and TensorRT-LLM is effectively unsupported on SM120 (trtllm-gen
cubins are SM100/103-only). So the gap is **external to all frameworks**, not an SGLang deficiency.

## Context: FP4 vs FP8 at 1K/8K

| Checkpoint | 1K/8K best (aggregate) | /GPU |
|---|---|---|
| `sgl-project/DeepSeek-V4-Flash-FP8` + DPA (shipped) | **3412.9** | 427 |
| `deepseek-ai/DeepSeek-V4-Flash` MXFP4 + DPA (this) | 497.6 | 62.2 |

**FP8 is 6.9× faster.** For production at 1K/8K, ship FP8 ([`../../fp8/1k8k/`](../../fp8/1k8k/)). This
bundle is the documented launch-space ceiling for the MXFP4 weights. The order-of-magnitude lever would
be a grouped/sorted MXFP4 GEMM kernel (load each expert tile once, reuse across its tokens) — a
**source-code** change, which is out of scope for this result.

## Correctness (CLAUDE.md rule 11)

gsm8k 50q on the MXFP4 marlin→sm120_triton path: **accuracy 0.900, invalid 0.000** (see
`results/correctness_gsm8k.txt`). The shipped r0.15 config uses identical MXFP4 numerics — `ratio`/`mfs`
change only KV memory layout, not compute — so the gate covers it.

## Files
- `launch_FP4_dpa_1k8k.sh` — the shipped (r0.15) launch command + rationale.
- `results/curve_r15.txt`, `SUMMARY_r{10,15,20}.txt` — the batch-scaling decode plateaus.
- `results/get_server_info_r15.json`, `srv_decode_excerpt_r15.txt`, `gpu_mem_r15.txt` — server state.
- `results/correctness_gsm8k.txt` — the rule-11 gate.

## Runtime-space exhausted (all no-source-edit levers closed)

| Lever | Verdict |
|---|---|
| Batch-scaling (enlarge SWA pool via `--swa-full-tokens-ratio` → 256/rank hard cap) | **saturated** at ~497 (flat 491–498 across 180→224/rank) |
| `--enable-deepseek-v4-fp4-indexer` | **dead** — source-gated to SM100 (`server_args.py:4166`), rejected on SM120 |
| DeepGemm MegaMoE FP4 (`SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE` …) | **dead** — deep_gemm hard-disabled at `sm_version==120` |
| #25569 fused-MoE Triton autotune | **no-op** — tunes the generic fused_moe, not the sm120 MXFP4 slot kernel |
| `--dsa-decode-backend tilelang` (vs auto) | **−38%** at matched batch (Blackwell+fp8-KV dequant tax) — gsm8k 0.95 |
| `--dsa-topk-backend flashinfer` (vs sgl-kernel) | **−13%** at matched batch — gsm8k 0.95 |

The auto-resolved DSA backend is already optimal. **497.6 tok/s is the launch/runtime-space ceiling for
the MXFP4 checkpoint on 1K/8K.** The only remaining lever is a source-code MXFP4 grouped-GEMM kernel
(out of scope). Ship FP8 for production at 1K/8K.
