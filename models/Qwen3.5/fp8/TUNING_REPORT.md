# Qwen3.5-397B-A17B-FP8 — Tuning Report

**As of 2026-05-18.** Ship config: **v38** — 2× SMG TP=8 + fused QK-norm-RoPE + fused MoE sum allreduce + NCCL_NCHANNELS=16. Median TTFT **1180 ms**, mean **2256 ms**, P99 **14320 ms**, total throughput **14,182 tok/s**. See [`results/benchmark_results.md`](results/benchmark_results.md) and [`launch_worker.sh`](launch_worker.sh).

## Model context

`Qwen/Qwen3.5-397B-A17B-FP8` uses **FP8 block-scaled quantization** (W8A8, block_n=128). Weights are ~400 GB; the model fits on a single G4 node at TP=8 with mfs=0.8, leaving ~5-10 GB / GPU for KV pool + activation. **Single-node TP=8 is the natural shape**; replication (2× SMG) gives 2× throughput at near-identical TTFT vs cross-node PP=2.

The model has a hybrid Gated Delta Networks + sparse MoE architecture (32 experts, 8 active per token) and supports native 262K-token context.

## Workload (TTFT-focused, fixed)

| Field | Value |
|---|---|
| Dataset | random, **200 prompts** |
| ISL / OSL targets | 20,000 / 1,000 |
| `--random-range-ratio` | **0.0** (default; variable lengths → effective ISL ~10K avg, OSL ~500 avg) |
| `--max-concurrency` | 40 |
| `--request-rate` | inf |
| Seed | 1 (default) |
| Bench backend | `--backend sglang` (native sglang protocol) |

This workload **differs from the throughput-focused workload** used by the other entries in this repo (1536 prompts × conc=512 × ISL=1024 / OSL=8192). The TTFT goal targets a production interactive-chat shape.

## Hardware & topology baseline

| Field | Value |
|---|---|
| Cluster | 2× GCP `g4-standard-384`, 16× RTX PRO 6000 Blackwell SM120 |
| Interconnect | 2× 200 Gbps gVNIC TCP, **no NVLink, no RDMA** |
| Image | `lmsysorg/sglang:dev-cu13` |
| sglang version | `0.0.0.dev1+gedb1b3f8f` (image-bundled, no pip pin) |
| `FLASHINFER_DISABLE_VERSION_CHECK` | `1` |
| Parallelism (ship) | **2× single-node TP=8 replicas + SMG router** |
| `--enable-flashinfer-allreduce-fusion` | on (best intra-node AR path) |
| `--enforce-piecewise-cuda-graph` | on (captures decode steps as cuda graphs) |
| Mandatory env vars | full NCCL/GLOO set + **`NCCL_NCHANNELS=16`** |

## What we shipped (v38)

Layered hill-climb on the SMG fan-out base:

| Layer | Flag | Mechanism | Δ Mean TTFT |
|---|---|---|---|
| 0 | base v6 single-node | TP=8 NEXTN spec + chunked=20480 | 4,470 ms (TTFT-focused workload, ratio=0 estimate from sweep) |
| 1 | 2× SMG replicas | `sglang-router round_robin` over 2× single-node TP=8 | ↓ to v21: 2,667 ms (-40 %) |
| 2 | Drop spec, add mixed-chunk + chunked-prefill=4096 | enables prefill-chunk-interleaved-decode | ↓ to v24: 2,345 ms (-12 %) |
| 3 | `--enable-fused-qk-norm-rope` + `--enable-fused-moe-sum-all-reduce` | fuses small ops, eliminates 698 ms of `Command Buffer Full` driver stalls per prefill (vs v24) | ↓ to v30: 2,262 ms (-3.5 %) |
| 4 | `NCCL_NCHANNELS=16` env var | more parallel NCCL channels, shaves P99 -4.6 % | ↓ to v38: 2,256 ms (-0.3 %) |

**Total stack delta vs baseline**: ~50 % mean TTFT reduction from the single-node-with-spec baseline.

## Profile-confirmed prefill wall decomposition (v30, TP-0 trace)

Per 20K-token prefill (5 chunks of 4K with mixed-chunk):

| Bucket | Time | % |
|---|---:|---:|
| NCCL allreduce GPU kernels (`ncclDevKernel_AllReduce_Sum_bf16_RING_LL`, ~500 calls) | **656 ms** | **54 %** |
| GPU compute (fused_moe, w8a8 fp8 matmul, FI prefill attn, GDN linear-attn, norm/act) | 388 ms | 32 % |
| `flashinfer/prefill.py:plan` blocking `cudaMemcpyAsync` (host-side book-keeping, fires 2× per req) | **164 ms** | **14 %** |
| `Command Buffer Full` driver stalls | **0 ms** | 0 % (vs 698 ms in pre-fused v21) |

The **656 ms NCCL allreduce is the hardware floor** on PCIe-only Blackwell. PCIe-oneshot AR is broken on g4 + SM120 per `project_pcie_oneshot_ar_broken_on_g4`; NVLS needs NVLink; symm_mem conflicts with FI AR fusion.

## Closed dimensions — do NOT rerun without new source-code reason

| Lever | Result | Root cause |
|---|---|---|
| `--enforce-piecewise-cuda-graph` on PP=2 | scale regression (16 s mean @ conc=40) | PP send/recv breaks captured graph composability |
| `--chunked-prefill-size` 2048 | +16 % mean | per-chunk NCCL AR cost dominates compute on small chunks |
| `--chunked-prefill-size` 40K/81K | +13-18 % mean | scheduler partial-chunks multiple reqs, spreading each prefill across batches |
| `--chunked-prefill-size` 20480 (= ISL) | +13 % mean despite -2 % median | chunking removal hurts queue dynamics for tail reqs |
| `--prefill-max-requests` 1/2/3 | +18-41 % mean | smaller batches finish faster but tail-of-burst explodes |
| `--max-running-requests` 8 | mean +1075 % | server admission queue replaces prefill queue; total wait unchanged |
| `--enable-prefill-context-parallel` | +86 % mean | cross-rank attention sync exceeds saved kernel work |
| `--attention-backend triton` | +105 % mean | flashinfer kernel is faster on Qwen3.5 (Kimi-K2.5 memory was model-specific) |
| `--page-size 8` | +9 % mean | flashinfer paged attention is optimized for page=1 |
| `--cuda-graph-bs` custom-tightened list | +77 % mean | extra captures ate KV pool, deepened prefill queue |
| `--mem-fraction-static 0.85` + `--num-reserved-decode-tokens 64` | mean +3 %, P99 -5 % | tail compresses but mean stays floor-bound |
| `--enable-attn-tp-input-scattered` | mean +7 % | scattered TP AR doesn't compose cleanly with FI AR fusion |
| NEXTN spec (v6 / v21 / v26) | -7 % mean median tied, +6 % mean | spec helps decode (TPOT) but adds prefill setup cost |
| NGRAM spec | best median + throughput but **+60 % mean** | KV-pool pressure from speculation widens prefill queue tail |
| Stack v38 + v37 + v39 levers | +2 % mean | individual P99 wins don't compose for mean |
| PD-Disaggregation (mooncake) | watchdog timeout | cross-node mooncake KV transfer hangs at 20K-ISL on gVNIC TCP |
| PD-Disaggregation (NIXL/UCX + tmpfs staging) | 222 s mean TTFT | functional but bandwidth-bound at this workload |
| PP=2 + `--enforce-piecewise-cuda-graph` | small-bench OK, full-bench mean +500 % | piecewise CG breaks PP send/recv at concurrency |

## Promising but unmeasured / hardware-blocked

| Idea | Mechanism | Expected delta | Cost / risk |
|---|---|---:|---|
| NVLink / NVSwitch hardware | unlocks PCIe-oneshot AR or NCCL NVLS → halves 656 ms AR floor | **-15 to -25 % mean TTFT** | requires new GPUs (H200, B200, or NVLink'd Blackwell) |
| RDMA hardware | makes PD-Disaggregation viable; mooncake/NIXL KV transfer at line rate | enables conc-decoupled scaling | requires GCP A-series or different cloud |
| Lower bench concurrency (workload change) | linear reduction of prefill queue depth | mean TTFT ~1.5 s @ conc=20 | not a server change |
| Third node + `--enable-prefill-context-parallel` cross-node | shards attention sequence across nodes | +5-15 % possibly | requires 3rd node + CP-friendly attention backend |
| Custom Blackwell-compatible PCIe-oneshot AR implementation | fix the `get_graph_buffer_ipc_meta` crash in `pcie_allreduce.cu:321` | -15 to -25 % AR | upstream sglang kernel fix needed |

## Closed structural / hardware constraints

| Constraint | Source |
|---|---|
| 656 ms NCCL AR per 20K prefill is the hardware floor on PCIe-only Blackwell | this investigation, v30 profile |
| `--pp-async-batch-depth ≥ 1` deadlocks on PP=2 + cross-node TCP | `project_pp_async_batch_depth_deadlocks` |
| `--enable-pcie-oneshot-allreduce` is dead code on g4 + SM120 + CUDA 13 | `project_pcie_oneshot_ar_broken_on_g4` |
| `--enable-symm-mem` + `--enable-flashinfer-allreduce-fusion` conflict at runtime | `project_symm_mem_dominated_by_fi_ar_fusion` |
| `--enforce-piecewise-cuda-graph` auto-disables under PP=2 | sglang source: `_handle_piecewise_cuda_graph` |
| Triton attention backend regression on Qwen3.5 | this investigation |

## Methodology corrections

Two non-trivial errors were discovered mid-investigation:

1. **`--random-range-ratio 1.0` vs default 0.0**: introduced `1.0` deterministically in mid-session bench commands. This doubled the effective workload vs the prior-session anchor (which used default 0.0 = variable [0, 20K] uniform ≈ 10K avg). All comparisons across sessions need to control for this. After reverting to default, baseline reproduced within 1 %.

2. **Stale SMG router process**: router pid sometimes persisted across multiple worker rounds, leading to `no_available_workers` errors. Fix: kill by explicit PID before each new launch, or include in the launch script orchestration.

## Suggested next-session experiments

1. **Re-test under `--random-range-ratio 1.0` (fixed ISL=20K)** — confirms behavior under a deterministic workload; informs production sizing.
2. **Wait for sglang upstream PCIe-oneshot AR fix** — would directly halve the 656 ms AR floor when available.
3. **`--enable-prefill-context-parallel` on a 3-node cluster** if available — sharding the sequence dim across more ranks reduces per-rank attention work.
4. **Try EAGLE3 spec with reduced draft tokens** — NEXTN/NGRAM both hurt TTFT; EAGLE3 with `--speculative-num-draft-tokens 1-2` might be the smallest-blast-radius spec.
