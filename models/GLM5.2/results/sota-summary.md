# GLM-5.2-NVFP4 SM120 — SOTA summary & lever sweep

ISL 1,024 / OSL 8,192, single node (8× RTX PRO 6000 SM120). Throughput = output tok/s per GPU.

## Headline ladder
| # | Config | Metric | tok/s/GPU | Δ vs stock | gsm8k |
|---|---|---|---:|---:|---:|
| 0 | stock bf16 dense (DP-attn, mfs0.94, cg=batch) | one-shot b88 | 147.0 | — | 0.92 |
| 1 | + fp8_e4m3 KV (glm-opt) | one-shot b143 | 162.2 | +10% | 0.96 |
| 2 | + minimal cuda-graph buckets | one-shot b159 | 174.5 | +19% | – |
| 3 | + mfs 0.955 | one-shot b167 | 179.8 | +22% | – |
| 4 | + chunked-prefill shrink (512/rank) | one-shot b175 | 184.7 | +26% | – |
| 5 | + chunked-prefill 256/rank, mfs0.97 | one-shot b191 | **194.6** | **+32%** | 0.97 |
| 6 | **EAGLE 3-step / 4-draft spec decode** | **sustained** | **~300** | **+104%** | **0.94** |

Steady-state (bench_serving) cross-check: non-spec @conc191 ≈ 207 tok/s/GPU; EAGLE 3-step @conc100 = 300, 4-step = 306 (3-step wins whole-run + ITL + gsm8k → the pick).

## The three stacking memory levers (non-spec, glm-opt fp8)
Decode is KV-pool-bound, so every byte freed for the pool buys batch:
1. **fp8_e4m3 KV** — 1.57× pool per mfs (vs bf16). Works only on the glm-opt branch (stock has no SM120 DSA fp8 decode kernel).
2. **minimal per-worker cuda-graph buckets** `--cuda-graph-bs "8 16 24 32"` — frees ~2.2 GB of graph private pool. ⚠️ buckets clamp to `max_running_requests/dp_size`; size at *global_batch ÷ dp_size* or you hit a 6× padding penalty.
3. **chunked-prefill shrink** `--chunked-prefill-size` (→128–256/rank under DP-attn) — shrinks the fixed cutlass fused-MoE prefill workspace, freeing room for a higher mfs.

Ceiling: mfs0.975 (the cutlass MoE prefill workspace OOMs on the post-decode probe). The one-shot peak is decode-activation-bound.

## EAGLE speculative decoding (the SOTA)
- Official MTP/NEXTN head: **accept-length ≈ 3.9 / 4** (well-trained), ~0.95 accept rate.
- Inverts the older "spec loses at max batch" finding — that was **NEXTN 1-step** (accept ~2). **EAGLE multi-step** (accept ~4) more than pays for the halved batch ceiling.
- Depth: 3-step ties 4-step on steady-state throughput but wins whole-run + ITL + gsm8k → **3 steps / 4 draft tokens** is the recommendation.
- Pin `--speculative-moe-runner-backend flashinfer_cutlass --speculative-moe-a2a-backend none` — otherwise modelopt_fp4 + EAGLE auto-routes the nextn MoE to `deep_gemm`/`deepep`, which are dead on SM120.
- `--speculative-eagle-topk > 1` (tree spec) is hard-blocked: flashinfer-MLA supports topk=1 only for spec on SM120 DSA.

## Closed / dead dimensions (do not re-run without a new reason)
| Lever | Verdict on SM120 |
|---|---|
| fp8 KV on **stock** dev-cu13 | dead — no SM120 DSA fp8 decode kernel (TllmGenFmhaRunner SM100-only) |
| moe-runner `flashinfer_cutedsl` / `flashinfer_trtllm` | dead — SM100-only kernels |
| moe-runner `marlin` | numerically broken (gsm8k 0.02) |
| `--fp4-gemm-backend flashinfer_cudnn` | no-op (+0.1%) — decode is not GEMM-bound |
| `--flashinfer-allreduce-fusion` | no-op — PCIe SM120 has no multicast substrate |
| `--enable-dp-lm-head` | harmful (pool −17%) |
| `--enable-two-batch-overlap` | config-blocked (index_topk_freq=4) |
| `--num-continuous-decode-steps` 2/4 | no-op — not scheduler-bound |
| EAGLE `topk>1` (tree) | blocked — flashinfer-MLA topk=1 only |
| multi-node PP=2 | needs a non-upstream patch; loses (75.9 tok/s/GPU, ~52% of single-node) |
| `--chunked-prefill-size` *enlarge* | OOMs the cutlass MoE workspace; *shrinking* is the lever instead |

## OOM tuning (continuous serving / over-subscription)
The one-shot SOTA (mfs0.97) OOMs at concurrency above the pool ceiling — a queued request's prefill MHA-K activation (~264 MiB) collides with the full decode pool. For serving past the ceiling: **mfs 0.92 + chunked-prefill 1024 (128/rank)** leaves ~7.6 GB headroom (the doc's 5–8 GB target) so over-subscription queues gracefully. (Lowering mfs is the opposite of the usual KV-OOM fix — this is an *activation* OOM at a full pool, not a pool-reservation OOM.)
