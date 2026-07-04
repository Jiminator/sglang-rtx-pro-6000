# DeepSeek-V4-Flash NVFP4 — SHIPPED (1K/8K + 8K/64K winner on SM120, latest main)

`nvidia/DeepSeek-V4-Flash-NVFP4` (hybrid FP8 + NVFP4 MoE, 157 GB). **On latest-main SGLang (SM120 / RTX
PRO 6000) this is the 1K/8K throughput leader and ties for the 8K/64K lead.** This supersedes the earlier
"NVFP4 runs but is slower, not shipped" verdict below — that predated upstream **#28231** (SM120→Marlin)
and conflated the `deepseek-ai/DeepSeek-V4-Flash` MXFP4 checkpoint (old per-slot GEMV, ~497 tok/s) with
this ModelOpt-NVFP4 checkpoint.

| Workload | tok/s | /GPU | vs prior offline anchor | bundle |
|---|---:|---:|---|---|
| 1K/8K (ISL 1024 / OSL 8192) | **~3753** | **469** | +10% vs 3413 | [`1k8k/`](1k8k/) |
| 8K/64K (ISL 8192 / OSL 65536) | **~1934** | **242** | 3.5× vs 551.9 | [`8k64k/`](8k64k/) |

gsm8k 0.960 both (bench_serving, `--random-range-ratio 1.0`). All three DSV4-Flash checkpoints tie within
~1.6% at 1K/8K and ~3% at 8K/64K (FP4 experts edge out FP8); NVFP4 is top/tied at both **and** the smallest
checkpoint → the single-checkpoint pick. MXFP4-Marlin (`../mxfp4/`) marginally tops it at 8K/64K (1941);
FP8 (`../fp8/`) is the conservative fallback.

## The two mandatory SM120 env vars (latest-main dev-cu13; no source patch)

`SGLANG_SM120_FLASHMLA_BACKEND=triton` (decode) + `SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=0` (prefill). Latest
main is doubly-broken for DSV4 on SM120 without them (flashinfer 0.6.12 dropped the `_sparse_mla_sm120`
decode symbol; the AOT sparse-prefill kernel is SM90a/SM100f-only). MoE runner = `flashinfer_cutlass`.
KV hard-locked to `fp8_e4m3`. Full mechanism + image saga + why 256/rank (1K/8K) & 64/rank (8K/64K) are the
ceilings: [`1k8k/TUNING_REPORT.md`](1k8k/TUNING_REPORT.md), [`8k64k/TUNING_REPORT.md`](8k64k/TUNING_REPORT.md).

---

## (Superseded 2026-06 note — kept for history)

> NVFP4 runs on SM120 but is slower than FP8 — ship FP8. On SM120 the fast FP4 MoE kernels were unavailable
> (`flashinfer_mxfp4` SM100-only; marlin CUDA mxfp4 NaN'd), so SGLang routed MoE to a triton per-slot GEMV
> (`_mxfp4_slot_gemv_kernel`), making FP4 decode dual-bound (DSA 41% + MoE GEMV 36%). **Upstream #28231
> (SM120→Marlin, in dev-cu13 now) removed that GEMV; the Marlin path is 7.4× faster, which is why FP4 now
> wins.** The old proxy was FP4 173 vs FP8 240 tok/s @ B=32.
