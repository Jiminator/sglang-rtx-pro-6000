# GLM-5.2-NVFP4 — Tuning Report (1K/8K, single node, SM120, latest main)

**As of 2026-07-05.** Workload **ISL 1024 / OSL 8192**, no-zipfian (`--random-range-ratio 1.0`), online
**`sglang.bench_serving`** as load generator → steady-state decode plateau read from server logs (at OSL 8192
requests don't complete in-window, so #running-req pins at the admission ceiling; aggregate = per-rank
gen-throughput × dp_size 8). Checkpoint `nvidia/GLM-5.2-NVFP4` (`GlmMoeDsa`), snapshot `b0b2b68`. Image:
**stock latest-main `lmsysorg/sglang:dev-cu13`** (sglang `0.0.0.dev1+gb28bc1060`, transformers 5.12.1).
Correctness gate: gsm8k on the idle server before load.

## Headline

**fp8 KV + DP-attention = ~2680 output tok/s (~335/GPU) @ 52 running-req/rank, gsm8k 0.900** — the top single
number, and **+108% over the bf16 plateau (~1264, 158/GPU)**. The EAGLE-3 spec variant (~2645, 330/GPU)
**ties within noise** and wins correctness (gsm8k 0.940) + latency (accept-len 4.0). See
[`launch_glm52_nvfp4_1k8k.sh`](launch_glm52_nvfp4_1k8k.sh).

## The fp8-on-stock unblock

The prior campaign's fp8 SOTA was **glm-opt-branch-only** — on stock dev-cu13 the fp8 DSA decode path crashed
(no SM120 DSA decode kernel: `TllmGenFmhaRunner` SM100-only, etc.). **On latest main the crash is gone** — the
trtllm DSA decode path now runs on SM120, so fp8 KV works on the stock image. Mechanism: fp8_e4m3 is 1.81–2.5×
denser than bf16 for the DSA KV pool, so at a given `--mem-fraction-static` the pool holds ~2.5× more tokens →
more concurrent sequences survive at saturation → higher decode plateau. Decode here is **KV-pool-bound**, so
every memory-freeing lever converts directly into pool capacity.

## Minimal-env ablation (A1a)

The **only** env var fp8 needs is `SGLANG_DISABLE_DSA_INDEXER_FUSION=1` (confirmed: gsm8k 0.940 with just it).
The two DSV4 env vars — `SGLANG_SM120_FLASHMLA_BACKEND` and `SGLANG_OPT_FLASHMLA_SPARSE_PREFILL` — are **inert**
for GLM: GLM DSA dispatches through a different code path than DeepSeek-V4's flash-MLA. Setting them changes
nothing. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is still required for boot (indexer buffer
fragmentation under DP-attention).

## Hill-climb (each lever frees memory → higher mfs → bigger pool → higher plateau)

| Step | Config | pool/rank | agg tok/s | /GPU |
|---|---|---:|---:|---:|
| baseline | bf16 KV, mfs 0.94 | 91.5K | ~1264 | 158 |
| + fp8 KV | fp8_e4m3, mfs 0.94 | ~180K | ~1888 | 236 |
| + mincg + chunked-prefill-shrink | fp8_e4m3, mfs 0.97 | ~220K | ~2616 | 327 |
| **+ mfs ceiling** | **fp8_e4m3, mfs 0.975** | **229.7K** | **~2680** | **335** |

The **minimal per-worker cuda-graph buckets** (`--cuda-graph-bs "16 32 48 64"`) and **chunked-prefill shrink**
(`--chunked-prefill-size 2048`) each free the fixed cutlass MoE prefill workspace + cuda-graph capture memory,
unlocking the next mfs notch. **mfs 0.975 is the ceiling** — 0.98 OOMs at boot.

## Spec vs non-spec

Tied at the plateau (~335 non-spec vs ~330 spec) because both are pool-bound at mfs~0.97–0.975 — spec's
draft+verify buffers halve the pool (128K/rank), and it runs at only 14–16 running-req/rank vs non-spec's 52,
but the ~4× per-request speedup (accept-len 4.0 on the well-trained MTP head) nearly compensates. Spec's real
edge is **correctness (gsm8k 0.940 vs 0.920 base) and latency** (lower ITL at matched load). Non-spec is the
simplest max-throughput config; spec is the latency/correctness pick.

## Closed levers

- **KV dtype:** fp8_e4m3 wins decisively; bf16 is the conservative fallback only (pool-bound, plateau decays).
- **mfs:** 0.975 ceiling (0.98 OOMs).
- **MoE runner:** `flashinfer_cutlass` is the only viable NVFP4 runner on SM120 (marlin gsm8k 0.02; cutedsl/trtllm no SM120 build).
- **cuda-graph / chunked-prefill:** tuned to their memory-freeing minimum; larger values eat pool with no plateau gain.
- **Spec topk:** `--speculative-eagle-topk` must be 1 (tree spec blocked on SM120 flashinfer-MLA).

## Config

TP=8 + DP-attention (dp8), `--moe-a2a-backend none --ep-size 1 --moe-runner-backend flashinfer_cutlass`,
`--attention-backend flashinfer`, `--kv-cache-dtype fp8_e4m3`, `--disable-shared-experts-fusion`,
`--mem-fraction-static 0.975`, `--chunked-prefill-size 2048`, `--cuda-graph-bs "16 32 48 64"`,
`--context-length 9472`, `--max-running-requests 1024`, + `SGLANG_DISABLE_DSA_INDEXER_FUSION=1` +
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + the NCCL/GLOO block. Full run data:
`runs/20260705_glm5.2_sota_humanize/` in the gcp-kimi repo.
