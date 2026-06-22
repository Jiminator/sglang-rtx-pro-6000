# DeepSeek-V4-Flash NVFP4 — runs on SM120, but slower than FP8 (not shipped)

`deepseek-ai/DeepSeek-V4-Flash` (NVFP4/mxfp4 experts, FP8 base).

**Verdict: NVFP4 runs on SM120 but is slower than the FP8 checkpoint — ship [`../fp8/`](../fp8/).**

## Why (profiled mechanism)

On SM120 the fast FP4 MoE kernels are unavailable (no TMEM/tcgen05): `flashinfer_mxfp4` is SM100-only
and faults at cuda-graph capture, and the **marlin CUDA mxfp4 kernel produces NaN**. SGLang therefore
routes the MoE to a **triton GEMV** — `_mxfp4_slot_gemv_kernel` in
`fused_moe_triton/mxfp4_moe_sm120_triton.py`.

> Note: the startup log `"SM120 detected: using PyTorch MXFP4 MoE fallback"` is a **misnomer** — it is
> a real Triton kernel, not a PyTorch reference loop. But it is still ~19× the cost of FP8's MoE.

Consequently FP4 decode is **dual-bound**: DSA attention **41%** + MoE GEMV **36%**, vs FP8 which is
DSA-only (MoE just 1.7%). The attention cost is identical between the two (both use fp8 KV + fp8
attention projections — only the *experts* differ fp4 vs fp8), so the extra 36% MoE tax is pure loss.
DP-attention can't fix it (it parallelizes attention, not the MoE tax). Proxy comparison: FP4 173 vs
FP8 240 tok/s @ batch 32.

## If revisited

The lever would be a fast SM120 mxfp4 MoE kernel (currently none exists in-tree). Until then, FP8 +
DP-attention is the single-node winner. See [`../fp8/TUNING_REPORT.md`](../fp8/TUNING_REPORT.md).
