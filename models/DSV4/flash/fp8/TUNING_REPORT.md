# DeepSeek-V4-Flash FP8 — Tuning Report (8K/64K, single node, SM120)

**As of 2026-06-21.** Ship config: **TP=8 + DP-attention**, triton MoE, FP8.
True 8K/64K workload (ISL 8192 / OSL 65536): **551.9 output tok/s @ batch 33 (≈69.0 / GPU)**,
ITL ~60 ms — **2.45× over pure TP=8**. See [`results/benchmark_results.md`](results/benchmark_results.md)
and [`launch_FP8_dpa.sh`](launch_FP8_dpa.sh).

## Model context

`sgl-project/DeepSeek-V4-Flash-FP8` — `DeepseekV4ForCausalLM`, 43 layers, MoE (256 routed experts,
top-6, 1 shared), **MLA with a single KV head** (head_dim 512), and **DSA (DeepSeek Sparse Attention,
`index_topk=512`)**. Mixed precision: FP8 base, FP8 KV. YaRN to 1M context (native 64K). The FP8
checkpoint is the one we ship; the FP4 checkpoint runs but is slower on SM120 (see
[`../nvfp4/README.md`](../nvfp4/README.md)).

## Workload (fixed)

| Field | Value |
|---|---|
| Benchmark | `sglang.bench_one_batch_server` (offline, synchronized fixed batch) |
| ISL | 8,192 tokens |
| OSL | 65,536 tokens |
| Batch | filled to KV-pool capacity (33 with DP-attention, 38 pure-TP) |

The 64K output length is load-bearing: KV footprint scales with output length (≈73,728 tok/req at
OSL 65536), so the KV pool caps the batch. At 8K/64K the pool holds **33** concurrent sequences under
DP-attention — that is the max batch for this shape, not a tuning knob.

## Hardware & topology baseline

| Field | Value |
|---|---|
| Node | 1× GCP `g4-standard-384`, 8× RTX PRO 6000 Blackwell **SM120** |
| Interconnect | intra-node PCIe, **no NVLink, no RDMA** |
| Image | `lmsysorg/sglang:v0.5.13.post1-cu130` |
| sglang version | v0.5.13.post1 |
| Parallelism (ship) | **TP=8 + DP-attention** (`--dp-size 8 --enable-dp-attention`), no spec |
| Mandatory env | full NCCL/GLOO set + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

## What we shipped

| Flag | Mechanism |
|---|---|
| `--tp 8 --dp-size 8 --enable-dp-attention` | **the win** — see mechanism below; `dp_size` must equal `tp_size` (dp<tp short-circuits the allreduce and hangs) |
| `--moe-a2a-backend none` | deepep needs deep_gemm, hard-disabled on SM120 (`deep_gemm_wrapper/configurer.py:22`) |
| `--moe-runner-backend triton` | auto force-selects marlin, whose `Fp8MoEMethod` lacks `self.runner` → AttributeError at cuda-graph capture; triton is the only working MoE runner on SM120 |
| `--mem-fraction-static 0.85` | maxes the KV pool (higher OOMs at this ctx) |
| `--cuda-graph-max-bs 64` | graphs the decode batch (B=33 ≤ 64) |
| `--context-length 73728` | 8192 + 65536 |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | avoids fragmentation OOM at high ctx |

## Why DP-attention wins (mechanism)

Profiled (FP8 pure-TP decode, **GPU-attributed** — works on the v0.5.13.post1-cu130 image):
`_tiled_sparse_decode_kernel` (the DSA sparse-attention decode kernel,
`flash_mla_sm120_triton.py`) = **83%** of decode GPU time; MoE only **1.7%**; NCCL AR **4%**. The
decode is **DSA-attention-bound**, so MoE-config tuning is worthless — the lever is *how the
attention is parallelized*.

Pure TP=8 awkwardly TP-shards the **single** MLA KV head across 8 ranks. DP-attention instead gives
each of 8 workers its own sequences with the full KV, so the bottleneck DSA kernel runs as **8
independent streams** (≈near-linear) while the tiny MoE is left alone. Per-step decode goes ~2.8×
faster (ITL 169 → 60 ms); aggregate is **2.45×** despite a *smaller* batch (DPA replicates KV per
rank → pool 2.62M vs TP 2.99M tokens → B=33 vs 38).

## Results (true 8K/64K)

| Config | batch | output tok/s | /GPU | ITL | E2E latency |
|---|---:|---:|---:|---:|---:|
| **FP8 + DP-attention** (ship) | 33 | **551.9** | **69.0** | ~60 ms | 3,999 s |
| FP8 pure-TP (baseline) | 38 | 225.5 | 28.2 | ~169 ms | 11,582 s |

Raw `bench_one_batch_server` JSONLs in [`results/`](results/).

## Closed dimensions (do-not-rerun, all SM120-verified on v0.5.13.post1)

| Variant | Verdict |
|---|---|
| **DP-attention** | **WIN, 2.45×** → shipped |
| FP4 checkpoint | runs (triton `_mxfp4_slot_gemv_kernel`, *not* a PyTorch loop) but **slower** — decode dual-bound (DSA 41% + a 36% MoE GEMV tax). FP8 avoids the tax. See [`../nvfp4/README.md`](../nvfp4/README.md). |
| EAGLE / MTP spec | **net loss** — DSA-bound, so draft forwards re-run the dominant kernel ~3× for ~2.9× tokens. `--max-running-requests 256` sticks but doesn't change the verdict. |
| `--moe-a2a-backend deepep` | dead — deep_gemm hard-disabled at `sm_version==120`. |
| cutlass / marlin / flashinfer_* MoE | dead on SM120 (HashTopK not-impl / `self.runner` TODO / SM100-only). triton is the only working runner. |
| `--dp-size 4` (dp<tp) | crashes — "short-circuiting allreduce will lead to hangs"; `dp_size` must equal `tp_size`. |
| `dev-cu13` image | identical to v0.5.13.post1 (no improvement). |

## Remaining lever (source/kernel-level, not yet done)

Decode is still DSA-attention-bound *within each DP worker*. The one lever left is the SM120 DSA
decode kernel `_tiled_sparse_decode_kernel` (`flash_mla_sm120_triton.py`, ~17% occupancy, narrow
3-config autotune): widen its `@triton.autotune` (add `BLOCK_T=64`) or wire the in-tree alternative
`triton_sparse_attn_decode_dsv4`. That is the only path above 69/GPU at this workload.
