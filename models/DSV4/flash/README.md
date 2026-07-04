# DeepSeek-V4-Flash on RTX PRO 6000 (SM120) — serving recipe + throughput (latest main, 2026-07-04)

Single node, 8× RTX PRO 6000 Blackwell (SM120, PCIe, no NVLink), **TP=8 + DP-attention**. Measured with
`sglang.bench_serving` (`--random-range-ratio 1.0`, no zipfian) → steady-state decode plateau from server
logs. Image: **latest-main `lmsysorg/sglang:dev-cu13`** (has upstream #28231).

## Throughput (aggregate output tok/s; /GPU = ÷8). All gsm8k 0.960.

| Checkpoint | MoE runner | **1K/8K** (1024/8192) | **8K/64K** (8192/65536) |
|---|---|---:|---:|
| **nvidia/DeepSeek-V4-Flash-NVFP4** 🥇 | flashinfer_cutlass | **3753** (469/GPU) | 1934 (242/GPU) |
| deepseek-ai/DeepSeek-V4-Flash (MXFP4, #28231) | marlin | 3717 (465) | **1941** (243/GPU) |
| sgl-project/DeepSeek-V4-Flash-FP8 | triton | 3694 (462) | 1881 (235) |

All three tie within ~1.6% (1K/8K) / ~3% (8K/64K) — decode is memory-bandwidth-bound, so the smaller-expert
FP4 checkpoints edge out FP8. **NVFP4 is top or tied at both and the smallest checkpoint → the pick.**
Bundles: [`nvfp4/`](nvfp4/) (shipped), [`mxfp4/`](mxfp4/), [`fp8/`](fp8/). vs prior offline anchors:
1K/8K 3413 (+10%), 8K/64K 551.9 (3.5×). **#28231 makes MXFP4 = 7.4× the old SM120 per-slot GEMV (497).**

## ⚠️ Latest-main is doubly-broken on SM120 — fix with TWO env vars (NO source patch)

| Blocker | Root cause | Fix (env var) |
|---|---|---|
| **Decode** `ModuleNotFoundError` | `flash_mla_sm120.py` imports `flashinfer.mla._sparse_mla_sm120`; flashinfer 0.6.12 renamed it away (the 0.6.12 replacement is an SM100-only TRTLLM-GEN kernel → a shim is infeasible) | `SGLANG_SM120_FLASHMLA_BACKEND=triton` |
| **Prefill** `Sparse Attention Forward Kernel only on SM90a/SM100f` | AOT `sgl_kernel.sparse_prefill_fwd` has no SM120 build; the sparse-prefill env defaults on so it always fires | `SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=0` (dense SM120 prefill; per-forward chunk 1024 < 11673 gate) |

Both are maintainer-supported env vars. `v0.5.13.post1-cu130` runs FP8/NVFP4 decode without them but is
pre-#28231 (MXFP4 = slow GEMV); FP8 is identical on both images.

## Fixed facts
- **KV dtype hard-locked to `fp8_e4m3`** (`deepseek_v4_hook.py` assertion; bf16 rejected at arg-parse).
- **MoE runner is per-checkpoint**: NVFP4→`flashinfer_cutlass`, MXFP4→`marlin` (#28231), FP8→`triton`.
- **Ceilings** (decode is BW-bound, past the knee): 1K/8K = 256 running-req/rank (= `--max-running-requests`
  2048 ÷ dp 8; pool stays empty, mfs won't help; MAXRUN=3072→prefill-bound, no gain). 8K/64K = ~64/rank
  (CONC=512); higher concurrency floods the 8192-token prefill backlog (prefill-swamped, no decode plateau).
- Spec decode is a net loss on DSV4 (EAGLE topk-1 only). Not used.

See per-workload `TUNING_REPORT.md` under each checkpoint dir. Full run data + the image-diagnosis ledger:
`runs/20260704_dsv4_flash_sota_humanize/` in the gcp-kimi repo.
