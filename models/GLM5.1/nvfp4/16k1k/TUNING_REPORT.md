# GLM-5.1-NVFP4 — Long-context (ISL=16,384 / OSL=1,024) Tuning Report

**As of 2026-05-11.** Ship config: **4,833.78 tok/s total** (output 277.43) — see [`results/benchmark_results.md`](results/benchmark_results.md) and [`launch_node1.sh`](launch_node1.sh), [`launch_node2.sh`](launch_node2.sh).

**Concurrency-scaling sweep** of this same ship config (no flag changes — only `--max-concurrency` and `--num-prompts` varied, num-prompts = 3 × concurrency): see [`concurrency_sweep/`](concurrency_sweep/) — 10 data points from conc=1 to conc=512, with per-metric figures and the full bench summary tables.

## Workload-specific context

This is a **prefill-dominant** workload: each request has 16,384 input tokens and only 1,024 output tokens (16:1 ratio). Profile of the ship config:

- 6,738 prefill batches vs 798 decode batches (8.4 : 1 prefill-dominant)
- Peak KV pool token usage = 0.82 (well-utilized, mfs is well-tuned)
- Mean TTFT = 118 s, mean TPOT = 2,440 ms — TPOT is huge because decode tokens are *blocked behind ongoing prefill chunks* on PP=2
- Median TTFT (13 s) vs mean TTFT (118 s) → bimodal: ~25 % fast-handoff requests vs a long-backlog tail

The workload definition is fixed; only server-side config varies between experiments.

## Hardware & topology baseline

| Field | Value |
|---|---|
| Cluster | 2× GCP `g4-standard-384`, 16× RTX PRO 6000 Blackwell SM120 |
| Interconnect | 2× 200 Gbps gVNIC TCP, **no NVLink, no RDMA** |
| Image | `lmsysorg/sglang:dev-cu13` |
| Parallelism (ship) | 2-node TP=8 PP=2 + DPA dp_size=8 |
| KV dtype | bfloat16 |
| FP4 GEMM / MoE runner | flashinfer_cutlass |
| Attention backend | flashinfer |
| Mandatory env vars | NCCL_P2P_LEVEL=SYS, NCCL_MIN_NCHANNELS=8, NCCL_ALLOC_P2P_NET_LL_BUFFERS=1, NCCL_NVLS_ENABLE=0, NCCL_CUMEM_ENABLE=0, NCCL_IB_DISABLE=1, NCCL_SOCKET_IFNAME=enp128s4,ens4, GLOO_SOCKET_IFNAME=ens4 (without them: throughput drops 10-15 %, TPOT regresses ~16 %) |

## What we shipped (the only confirmed win on this workload)

| Lever | Mechanism | Delta vs anchor (4,574 tok/s) |
|---|---|---:|
| `--enable-dynamic-chunking` + `SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR=0.65` + `--chunked-prefill-size 8192` | Quadratic runtime predictor shrinks chunk size as prefix grows so per-chunk runtime stays equalized across PP=2 stages. Effective `chunked_prefill_size` is auto-capped to 1024 by DPA (build-scoped). The initial 8192 seeds the predictor; SGLANG smoothing factor 0.65 was best of the {0.65, 0.75} sweep. | **+5.66 % (4,833.78 tok/s)** |

The dynamic-chunking flag was originally designed for *exactly* this PP + long-ISL profile — see the lmsys "Chunked Pipeline" blog (2026-01-15).

## Closed dimensions — do NOT rerun without new source-code reason

Each row is fully diagnosed; do not re-test unless something upstream materially changes.

| Lever attempted | Mechanism aimed at | Result | Root cause |
|---|---|---|---|
| `--mem-fraction-static 0.92` (vs ship 0.90) | grow KV pool by ~5 % | **runtime OOM** during first chunked-prefill batch | 16K-token chunk's transient activation peak (3.05 GiB allocation) exceeds the 1.97 GiB free remaining after raising mfs. mfs=0.90 leaves 9.81 GiB free per GPU — the structural floor. |
| `SGLANG_PP_LAYER_PARTITION=38,40` | balance later/earlier stage workload on PP=2 | **−0.38 %** (noise) | 78 hidden layers split evenly at 39/39 by default; ±1-layer asymmetry is within noise. No tail-latency improvement either. |
| `--enable-two-batch-overlap` (TBO) | overlap two micro-batches per PP stage to hide P2P comm | **not bootable on this stack** | TBO requires `moe_a2a_backend != none` + `ep_size > 1`. nixl & mooncake A2A backends explicitly `NotImplementedError("Normal mode is not supported")` — only low_latency dispatch implemented. flashinfer A2A misses `ep_size` setter for `DeepseekV2MoE` in init path (`deepseek_v2.py:547-555` omits `is_flashinfer()` from the EP branch); `op_dispatch_a` later crashes with `AttributeError: 'DeepseekV2MoE' object has no attribute 'ep_size'` during cudagraph capture. Upstream source bug — would need patch. |
| `--enable-single-batch-overlap` (SBO) | overlap within a single batch via dual streams | **no-op on this stack** | All three `SboFlags` benefit paths are gated off: `combine_down_gemm` requires `flashinfer_cutedsl` runner (we use `flashinfer_cutlass`); `dispatch_shared_one_stream` requires `not is_blackwell()`; `combine_shared_two_stream` requires shared-experts NOT disabled (but `--disable-shared-experts-fusion` is mandatory for GLM-NVFP4 loader, per `project_glm_nvfp4_shared_experts_fusion_required`). |
| `--enable-mixed-chunk` (at mfs=0.90) | interleave decode tokens inside prefill chunks to reduce decode starvation | **runtime OOM** at T+2:40 (3.05 GiB needed, free 1.97 GiB) | mixed-chunk piggybacks decode tokens onto each prefill chunk; at ISL=16,384 with 512 concurrent in-flight, the `_concat_and_cast_mha_k` attention scratch buffer (~`chunk × hidden × num_in_flight`) overflows free GPU memory. |
| `--enable-mixed-chunk` at mfs=0.86 (retry, +3 GiB headroom) | same goal, with more activation headroom | **runtime OOM** at T+2:14 (3.09 GiB needed, free 3.04 GiB — exceeded by 50 MiB) | Same allocation site, slightly different chunk shapes. The ~3 GiB freed by lowering mfs was nearly fully consumed by mixed-chunk's larger scratch peak. Structural; mfs ceiling does not bracket the peak. |
| PD-Disaggregation NIXL UCX-TCP (symmetric TP=4 PP=2 dp=4 + fp8_e4m3 KV) | separate prefill & decode for decode isolation | **−80.7 %** total throughput (282 / 1,536 completed in 40.5 min) | Two compounding handicaps: (a) per-side KV pool ~510K tokens at fp8 → only ~30 concurrent decode seqs (vs ship's effective ~50); (b) inter-node KV-state copy over gVNIC TCP for ISL=16,384 is huge — fights NCCL collectives for fabric bandwidth. Mean TTFT 239 s, p99 590 s. Architectural fit for RDMA/NVLink only. |
| `--pp-async-batch-depth ≥ 1` | overlap P2P sends with later forward batches | **deadlocks** | Confirmed elsewhere on this stack (PP=2 + DPA + cross-node TCP): all 16 ranks watchdog-timeout at 300 s. Closed dimension. |
| Asymmetric TP between PD prefill (TP=4) and PD decode (TP=8) | bigger decode KV pool while letting prefill PP=2 | **NIXL KVReceiver Exception** | NIXL transfer assumes per-rank KV layout matches; TP mismatch produces incompatible shards. Must be symmetric. |
| Single-node TP=8 PP=1 for PD-Disagg, mfs ≤ 0.90 | smaller per-GPU weight footprint than PP=2 in-node | **boot OOM** ("Not enough memory") | GLM-5.1 weights at TP=8 single-node leave insufficient room for KV pool at standard mfs. Must use TP=4 PP=2 within node (or accept much smaller KV). |

## Promising but unmeasured (open levers, ranked)

These are reasonable next-step experiments. Each predicts a specific delta and has a defined success criterion.

| Idea | Mechanism | Expected delta | Cost | Risk / notes |
|---|---|---:|---|---|
| **1. fp8_e4m3 KV on the unified ship** | Halves KV memory per token → effective KV pool grows from 877K to ~1.75M tokens. Could let us run mfs=0.88 (free 3 GiB / GPU) and still beat the ship via more concurrency in the long tail. | **+3 to +8 %** | 1 flag (`--kv-cache-dtype fp8_e4m3`) on existing topology. ~50 min bench. | We confirmed fp8_e4m3 boots and runs cleanly on flashinfer attention during the PD-Disagg test (mean TPOT was actually fine: 147 ms). Prior `project_dev_cu13_nsa_uplift` warning of 2.5× regression was NSA-tilelang-specific, not flashinfer. Worth a clean run. |
| **2. `--enable-dynamic-chunking` smoothing-factor finer sweep** | Smoothing factor 0.65 won over 0.75 by ~2 %. Try {0.55, 0.60, 0.70} and {`--chunked-prefill-size` 4096, 6144}. | **+0.5 to +2 %** | 3-5 short benches × 50 min each. | Tuned around the existing win; diminishing returns. Worth the budget only after #1 closes. |
| **3. Mooncake transfer for PD-Disagg with HiCache spill** | Re-test PD with the *opposite* transfer choice + L2 KV cache. Mooncake writes through to L3 eagerly; NIXL POSIX only on eviction. On long-ISL workload with KV-transfer saturation, eager write-through might paradoxically *help* by spreading transfers over time. | **sign uncertain (-15 to +5 %)** | Equivalent to redoing Run 4 PD with `--disaggregation-transfer-backend mooncake` + `--enable-hierarchical-cache`. | Likely still loses to unified ship; do only after #1 and #2 close. |
| **4. Triton attention (not flashinfer) + dynamic chunking** | Triton attention has a different decode kernel scheduler that may interleave better with PP boundaries. Per the `project_kimi_nvfp4_blackwell_attention_default` memory, triton was 2.7× faster on Kimi-K2.5-NVFP4 vs forced flashinfer. May or may not apply to GLM. | **sign uncertain (+ or − 10 %)** | 1 flag (`--attention-backend triton`); ~50 min bench. | The dev-cu13 image has a cudagraph buffer regression on triton attention (see KimiK2.5/int4 notes); may require the sglang 0.5.10.post1 pin to actually be useful. |
| **5. `--enforce-piecewise-cuda-graph` instead of the default disabled state** | The ship config has `disable_piecewise_cuda_graph: True`. Enabling piecewise graphs would create per-stage decode cuda graphs; could overlap decode with prefill within a stage. | **+1 to +3 %** | 1 flag; ~50 min bench. | Currently disabled because of memory pressure interactions with `--enable-dp-attention`; needs careful mfs revisit. |
| **6. Decode-side `--stream-interval > 1`** | Default 1; sends a streaming token per server step. Raising it to 4 or 8 reduces per-token serialization overhead at the cost of perceived latency (TTFT for first chunk vs. first token). Long-OSL workloads benefit; ours has OSL=1024 so the gain is modest. | **+0.5 to +1 %** | 1 flag; ~50 min bench. | Quick easy ablation. Trade-off only matters for clients that depend on token streaming. |
| **7. NSA prefill context-parallel** (`--enable-nsa-prefill-context-parallel`) | DSv3.2-specific (uses NSA lightning indexer) — GLM-5.1 doesn't have NSA so not applicable. | n/a | n/a | Listed for completeness — closed dimension on GLM. |

## Where the bottleneck lives now

After dynamic chunking, the workload is still **PP-bubble-limited + decode-starved by prefill**:

- Median TTFT 13 s vs Mean 118 s — bimodal: requests that land early get quick service, late ones queue 10× longer.
- TPOT mean 2,440 ms is enormous — decode is starved while prefill churns through chunks. Mixed-chunk would have addressed this but OOMs.
- Without TBO / SBO / flashinfer normal-mode A2A on the stack, we can't pipeline-overlap MoE dispatch with prefill compute.

The biggest single-flag lever left is **fp8 KV** (idea #1). It directly addresses the long-TTFT tail by allowing more in-flight requests per PP stage's KV pool. After that, gains compound in single-digit percentages.

## Closed structural / hardware constraints (not flag-tunable)

| Constraint | Source |
|---|---|
| `--enable-pcie-oneshot-allreduce`: dead code on g4 + SM120 + CUDA 13 — JIT-builds but `get_graph_buffer_ipc_meta` crashes at runtime. | `project_pcie_oneshot_ar_broken_on_g4` |
| `--enable-symm-mem`: dominated by FI-AR-fusion on PCIe Blackwell. Never stack. | `project_symm_mem_dominated_by_fi_ar_fusion` |
| `fa3` attention backend: rejected on Blackwell SM12 (Ampere/Hopper only). | `project_fa3_rejected_blackwell` |
| `--disable-shared-experts-fusion`: MANDATORY for `lukealonso/GLM-5.1-NVFP4` (loader crash without it). | `project_glm_nvfp4_shared_experts_fusion_required` |
| PD-Disagg requires symmetric TP between prefill/decode; NIXL KV layout depends on per-rank shard. | This investigation. |
| `dev-cu13` image ships no NCCL defaults — env vars must be set explicitly via wrapper. | `feedback_nccl_env_vars_required` |

## Hardware upgrades that would change the picture

- **NVLink / NVSwitch**: would unlock PCIe-oneshot AR (or NCCL NVLS path), making intra-node AR much faster — possibly +5-15 % on this workload.
- **RDMA (Mellanox IB / RoCE)**: would make PD-Disaggregation viable by removing the gVNIC TCP saturation issue. PD might land +10 to +15 % on this prefill-heavy workload with RDMA.
- **More nodes (≥ 3)**: would unlock prefill context parallelism (`--enable-prefill-context-parallel`) which shards each prefill across DP groups for additional throughput.

## Suggested experiment order for next session

1. **fp8_e4m3 KV** on the existing ship (single flag). If it helps, lock it in as the new anchor.
2. **piecewise cudagraph** on top (alone or stacked with #1).
3. **stream-interval** sweep — low-risk small win.
4. **smoothing-factor finer sweep** ({0.55, 0.60, 0.70}) — diminishing returns territory.

Stop after the first two clear losses (≥ ±0.5 %) to bound budget at ~4 hours.
