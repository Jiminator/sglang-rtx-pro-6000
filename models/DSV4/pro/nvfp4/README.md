# DeepSeek-V4-Pro NVFP4 — not yet benchmarked on this cluster

Placeholder. Not yet run on the RTX PRO 6000 (SM120) node.

Caveat carried over from Flash: on SM120 there is **no fast FP4 MoE kernel** (marlin CUDA NaNs,
flashinfer_mxfp4 is SM100-only), so FP4 falls back to a triton GEMV that is ~19× the cost of FP8's
MoE. On Flash this made FP4 slower than FP8 — expect the same on Pro unless a fast SM120 mxfp4 kernel
lands. See [`../../flash/nvfp4/README.md`](../../flash/nvfp4/README.md).
