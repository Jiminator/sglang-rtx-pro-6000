# DeepSeek-V3.2-NVFP4 — Tuning Report

**As of 2026-05-11.** Three deployable configurations live under this folder; the table below is the verdict on each.

| Config | Folder | Output tok/s | Total tok/s | Status |
|---|---|---:|---:|---|
| **2× replica behind SMG router** | [`smg-2x/`](smg-2x/) | **4,395.44** | **4,931.62** | **Ship (use this)** |
| PD-Disaggregation (NIXL UCX-TCP) | [`pd-disagg/`](pd-disagg/) | 3,730.39 | 4,185.45 | Inert on this cluster (−15.1 %) |
| Single-node EAGLE speculative decoding | [`sglang-ds32-job-v2.yaml`](sglang-ds32-job-v2.yaml) | 2,675.33 | 3,012.42 | Lower-throughput alternative — useful when low TPOT matters more than raw tokens/s |

Workload throughout: 1,536 prompts, ISL=1,024 / OSL=8,192, max-concurrency 512, seed 1, `--apply-chat-template`. **The ship config is +47 % over the original prior anchor (2,980 bf16 → 4,395 NVFP4 SMG-2x).**

## Hardware & topology baseline

| Field | Value |
|---|---|
| Cluster | 2× GCP `g4-standard-384`, 16× RTX PRO 6000 Blackwell SM120 |
| Interconnect | 2× 200 Gbps gVNIC TCP, **no NVLink, no RDMA** |
| Image | `lmsysorg/sglang:dev-cu13` (sglang `0.0.0.dev1+g3c3f0bd55`) |
| Per-replica parallelism (ship) | single-node TP=8 + DPA dp_size=8 |
| Router | sglang-router, `--policy cache_aware`, two workers `http://10.0.1.2:8000` + `http://10.0.1.4:8000` |
| Mandatory env vars | NCCL_P2P_LEVEL=SYS, NCCL_MIN_NCHANNELS=8, NCCL_ALLOC_P2P_NET_LL_BUFFERS=1, NCCL_NVLS_ENABLE=0, NCCL_CUMEM_ENABLE=0, NCCL_IB_DISABLE=1, NCCL_SOCKET_IFNAME=enp128s4,ens4, GLOO_SOCKET_IFNAME=ens4 (without them: throughput drops 10–15 %, TPOT regresses ~16 %) |

## What we shipped — `smg-2x/` at mfs=0.85

The journey: original baseline (PP=2 cross-node, 3,743 tok/s) → SMG 2-replica topology (3,947 under-tuned, then **4,377 with full NCCL env vars** = "Run B' v4 corrected anchor") → mfs=0.85 (**4,395, +0.4 % within noise**, but with **−74 % mean TTFT and −94 % p99 TTFT** vs mfs=0.82). The mfs=0.85 lever is the key win: tied throughput, dramatically better TTFT tail.

Mandatory server flags (per memory `feedback_nccl_env_vars_required` and the ship REPORT):

```
--quantization modelopt_fp4
--disable-shared-experts-fusion
--reasoning-parser deepseek-v3
--tool-call-parser deepseekv32
--nsa-prefill-backend tilelang
--nsa-decode-backend tilelang
--page-size 64
--kv-cache-dtype bfloat16        ← MUST be explicit; fp8_e4m3 is 2.5× slower on NSA+tilelang
--tokenizer-path deepseek-ai/DeepSeek-V3.2
--schedule-policy fcfs            ← random workload only; lpm/cache_aware here would build a useless radix tree
--enable-dp-attention --dp-size 8 --enable-dp-lm-head --enable-fused-moe-sum-all-reduce
--attention-backend flashinfer
--moe-runner-backend flashinfer_cutlass
--enable-flashinfer-allreduce-fusion --disable-custom-all-reduce
--mem-fraction-static 0.85
```

## Closed dimensions — do NOT rerun without new source-code reason

Each row is fully diagnosed. Re-test only if upstream sglang / image / hardware materially changes.

### Server-side ablation outcomes

| Dimension | Values tried | Result | Root cause |
|---|---|---|---|
| **NCCL/GLOO env vars** | absent vs full set | **MUST be set** — full set adds ~10–15 % throughput; without them, TPOT regresses 16 %. dev-cu13 ships **no NCCL defaults** — they're load-bearing. | feedback `feedback_nccl_env_vars_required` |
| `--mem-fraction-static` (with NCCL env correct) | 0.82, 0.85, 0.86, 0.88 | 0.82-0.85 tied throughput; **0.85 dominates on TTFT tail**; 0.86+ OOMs FusedMoE workspace mid-bench (KV-pool sizing is fine; the dispatch workspace is what blows). | `run-g-staircase-2026-05-08` |
| `--schedule-policy` (server) | `lpm` (default), `fcfs` | `fcfs` strictly better on random workload (`lpm` maintains an unused radix tree). | feedback `feedback_fcfs_for_random_workloads` |
| `--policy` (sglang-router) | `cache_aware`, `round_robin` | `cache_aware` slight edge (+1.4 %) on the corrected NCCL-env baseline. Supersedes older "round_robin equivalent" advice. | Run B' v3 vs v4 |
| `--kv-cache-dtype` | `auto` (default fp8_e4m3) vs `bfloat16` | **bfloat16 MANDATORY** on Blackwell+NSA+tilelang. fp8_e4m3 dequant path is ~2.5× slower per step. | project `project_dev_cu13_nsa_uplift` |
| `--enable-symm-mem` + `--enable-flashinfer-allreduce-fusion` | stacked vs FI-AR-fusion alone | **FI-AR-fusion alone wins** — symm-mem is dominated on PCIe Blackwell with no NVL substrate. | project `project_symm_mem_dominated_by_fi_ar_fusion` |
| `--pp-async-batch-depth ≥ 1` (only matters if PP>1) | `0`, `≥1` | **deadlocks** on PP=2 + DPA + cross-node TCP — all 16 ranks watchdog-timeout at 300 s. (N/A on ship since ship is PP=1 per replica.) | project `project_pp_async_batch_depth_deadlocks` |
| `--enable-pcie-oneshot-allreduce` | enabled | **dead code** on g4 + SM120 + CUDA 13: JIT-builds but `get_graph_buffer_ipc_meta` crashes at runtime. | project `project_pcie_oneshot_ar_broken_on_g4` |
| `--chunked-prefill-size` (with DPA) | 16,384 requested, 4,096 requested | server **silently caps** to 2,048 (sglang 0.5.10.post1) or 4,096 (dev-cu13). Cap is build-scoped. Always verify via `/get_server_info`. | project `project_dpa_chunked_prefill_auto_override` |

### Architectural / multi-server topology outcomes

| Topology / feature | Setup | Result | Root cause |
|---|---|---|---|
| **PD-Disaggregation NIXL UCX-TCP** | 1 prefill + 1 decode, symmetric TP=8 each, mfs=0.85 | **−15.1 %** output throughput (3,730 vs 4,395 ship) | Cross-node KV-state copy over gVNIC TCP fights NCCL collectives; decode TPOT regresses by ~15 ms (95→111) because KV transfers block decode batches at sub-step granularity. Mean TTFT explodes 4 s → 18 s with p99 tail at 259 s. Architectural fit only with RDMA/NVLink. |
| **PD-Disaggregation Mooncake** (alt transfer backend) | same topology, `--disaggregation-transfer-backend mooncake` | BOOT_FAILURE (Run J Mooncake variant did not produce a measurement on this workload — see `_raw/dsv3nvfp4/2node-pd-disagg-mooncake-2026-05-09/BOOT_FAILURE.md` history) | Mooncake's TCP backend is more sensitive to MTU + connection-setup ordering on this cluster. Not pursued since NIXL already established PD as non-competitive. |
| **PD-Disagg with asymmetric TP** (prefill TP=4 PP=2, decode TP=8 PP=1) | tried on GLM-NVFP4 sibling investigation; structural | `NIXL KVReceiver Exception` — KV layout per-rank depends on TP, must be symmetric | This investigation (cross-applicable). |
| **HiCache + NIXL POSIX / tmpfs** (`--hicache-storage-backend nixl`) | enabled on ship topology | **−2.74 %** (4,257 vs 4,395) on random workload; only modest TTFT tail improvement | NIXL POSIX/tmpfs is "deferred write" (writes to L3 only on L2 eviction). Random workload generates near-zero hits, so the L1↔L2 transfer overhead is pure cost. Useful on shared-prefix workloads. | Run H (corrected) |
| **HiCache + Mooncake** (`--hicache-storage-backend mooncake`) | enabled on ship topology | **−5.25 %** (4,147 vs 4,395) | Mooncake is "eager write-through" — writes 467 GB / 222K puts to L3 over TCP on every L1 write, even when L3 is empty. Strictly worse than NIXL on random workload; sign would flip on shared-prefix workloads. | Run I (corrected) |
| `--cpu-offload-gb > 0` on DSv3.2 | `--cpu-offload-gb 10` | **CRASH** — `torch.bmm` "mat2 on cpu" inside MLA-absorb | OffloaderV1's `functional_call` only re-binds registered parameters; DSv3.2 MLA's post-load `self.w_kc` is an instance attribute, stays on CPU. Deterministic. Use `--mem-fraction-static` bump for more KV headroom instead. | project `project_offloader_mla_incompat` |
| `--hicache-io-backend direct` + `page_first_direct` | enabled | **CRASH** — same `offloader.py:144 functional_call` bug class as cpu-offload-gb | Same root cause. Default to `--hicache-mem-layout layer_first --hicache-io-backend kernel` for HiCache on DSv3.2. | project `project_hicache_io_backend_direct_crash` |
| `pip install nixl` / `pip install mooncake-transfer-engine` | inside dev-cu13 | **libcudart conflict** — wrecks cuDNN dl_iterate_phdr | dev-cu13 image already ships `nixl-cu13` + `mooncake-transfer-engine-cuda13`. **Never pip install upstream — they pull cu12 wheels.** Always `pip list` first. | project `project_mooncake_libcudart_conflict` |
| io_uring inside Docker (NIXL POSIX `use_uring=true`) | enabled | fails with "Failed to initialize io_uring instance: Success" | Docker's default seccomp blocks `io_uring_setup`. Use `use_uring=false` or pass `--security-opt seccomp=unconfined` to docker run (our wrapper does the latter). | project `project_iouring_blocked_docker_seccomp` |

## Promising but unmeasured (open levers, ranked)

| Idea | Mechanism | Expected delta | Cost | Notes |
|---|---|---:|---|---|
| **1. EAGLE speculative decoding on the 2× SMG ship** | The existing single-node EAGLE config (this folder's [`sglang-ds32-job-v2.yaml`](sglang-ds32-job-v2.yaml)) uses EAGLE with `num-steps=3, topk=1, num-draft-tokens=4` and gets 2,675 output tok/s on 8 GPUs (332 tok/s/GPU). The ship 2× SMG gets 4,395 / 16 = 275 tok/s/GPU. **EAGLE would buy back per-replica per-token throughput**. Stack on top of mfs=0.85 + cache_aware router. | **+10 to +25 %** | Adds `--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --speculative-moe-runner-backend flashinfer_cutlass` to the worker_launch.sh. ~40 min bench. | EAGLE acceptance rate on DSv3.2 was 2.87 (from the EAGLE config's results) — every accepted token saves a forward pass. Highest expected uplift; this is the largest unexplored lever for DSv3.2-NVFP4 throughput. |
| **2. `--enable-mixed-chunk` on the 2× SMG ship** | Interleave decode tokens inside prefill chunks. The single-node EAGLE config (also under this folder) enables this flag. Workload is decode-heavy (ISL=1,024 / OSL=8,192) so the scratch peak should fit (unlike GLM-5.1 16K where it OOMs). | **+2 to +5 %** | 1 flag; ~40 min bench. | Independent of EAGLE; can stack with #1 if both win. |
| **3. `--flashinfer-cutlass-moe-fp4-allgather` off** | The existing EAGLE config sets `--fp4-gemm-backend flashinfer_cudnn` (different from ship's `flashinfer_cutlass`). cuDNN backend may be faster on Blackwell SM120 for the FP4 GEMM. | **−5 to +5 %** | Swap one flag; ~40 min bench. | Sign uncertain — `flashinfer_cutlass` is what the corrected anchor measured. cudnn may help or hurt; quick to bound. |
| **4. NSA prefill context parallel** (`--enable-nsa-prefill-context-parallel`) | DSv3.2-specific — shards NSA lightning indexer's prefill across DP groups | **+0 to +3 %** | needs `nnodes ≥ 3` to give a useful CP split. We only have 2 nodes. | Closed for now on this cluster; revisit if a 3rd node appears. |
| **5. `nsa-prefill-backend` alternatives** | currently `tilelang` (the only one we tested); other backends might exist in newer sglang builds | **sign unknown** | swap one flag; ~40 min bench each | Low priority — `tilelang` is what unlocked the 2.5× win, swapping risks losing it. |
| **6. Re-enable PD-Disagg if we ever get RDMA** | RDMA fabric would eliminate the gVNIC TCP saturation that killed Run J | **+5 to +15 %** (theoretical) | Free once hardware lands | Listed for completeness; not actionable today. |
| **7. `--mem-fraction-static 0.86` retry under different concurrency** | mfs=0.86 OOM'd at 512-concurrency; might survive at lower concurrency where activation peaks are smaller | **+0 to +2 %** | swap mfs + max-concurrency; ~40 min bench | Workload args are normally fixed but this would be a diagnostic to find the actual memory ceiling. Modest expected gain; low priority. |
| **8. Push concurrency past 512** | peak KV pool usage at mfs=0.85 is well below saturation — there's headroom for more in-flight requests | **+5 to +15 %** | requires changing the (otherwise fixed) bench's `--max-concurrency`. Diagnostic only. | Worth doing once as a ceiling probe. |

## Where the bottleneck lives now

After 2× SMG + mfs=0.85 + corrected NCCL env vars:

- **Output throughput is roughly 275 tok/s/GPU.** EAGLE on the single-node config achieves 334 tok/s/GPU — there's headroom from speculative decoding (lever #1 above).
- **Inter-node fabric is otherwise idle** — the 2× SMG topology has each replica self-contained on one node. Cross-node traffic is only the router's HTTP forwarding (small). This is the architectural reason SMG wins over PD-Disagg.
- **KV pool is well-utilized** but not saturated. The mean TTFT of 4.1 s indicates the queue is short; p99 of 16 s indicates a moderate burst tail that doesn't need further KV growth.
- **Mean TPOT 96 ms** is roughly the per-token cost on this hardware without speculative decoding. EAGLE could cut this materially.

## Closed structural / hardware constraints (not flag-tunable)

| Constraint | Source |
|---|---|
| PD-Disaggregation requires symmetric TP; NIXL KV layout depends on per-rank shard. | This investigation. |
| `--enable-pcie-oneshot-allreduce` is dead code on g4 + SM120 + CUDA 13. | `project_pcie_oneshot_ar_broken_on_g4` |
| `--enable-symm-mem` dominated by FI-AR-fusion on no-NVLink Blackwell. | `project_symm_mem_dominated_by_fi_ar_fusion` |
| `fa3` attention backend rejected on Blackwell SM12 (Ampere/Hopper only). | `project_fa3_rejected_blackwell` |
| `--disable-shared-experts-fusion` mandatory for DSv3.2 NVFP4 (loader). | DSv3.2 standard |
| `dev-cu13` image ships no NCCL defaults — env vars must be set explicitly via wrapper. | `feedback_nccl_env_vars_required` |
| `--cpu-offload-gb` and `--hicache-io-backend direct` both crash on DSv3.2's MLA-absorb post-load attribute pattern. | `project_offloader_mla_incompat`, `project_hicache_io_backend_direct_crash` |
| `--chunked-prefill-size` silently capped to 2,048-4,096 under DPA (build-scoped). Confirm via `/get_server_info`. | `project_dpa_chunked_prefill_auto_override` |

## Hardware upgrades that would change the picture

- **NVLink / NVSwitch**: unlocks PCIe-oneshot AR (or NCCL NVLS), +5–15 % on intra-node AR work. Also makes intra-node KV transfer fast enough for PD-Disagg to potentially win.
- **RDMA (Mellanox IB / RoCE)**: makes PD-Disaggregation viable by removing gVNIC TCP saturation. The architectural decode-isolation premise would actually convert to throughput here.
- **Third node**: unlocks `--enable-nsa-prefill-context-parallel` (DSv3.2-specific CP shard across DP groups). +0–3 %.
- **More GPU memory per card (≥ 128 GB)**: would let us push mfs higher without FusedMoE workspace OOM — possibly +2-5 %.

## Suggested experiment order for next session

1. **EAGLE on 2× SMG** (lever #1) — biggest expected uplift, single flag-set addition. If +10 %+, ship as new anchor.
2. **`--enable-mixed-chunk`** (lever #2) — independent of EAGLE; stack on whatever wins.
3. **`--fp4-gemm-backend flashinfer_cudnn`** (lever #3) — quick ablation; bound the cuDNN-vs-cutlass question.
4. **Concurrency ceiling probe** (lever #8) — diagnostic; informs whether further KV-pool growth would help in practice.

Stop after #1 and #2 if both ship cleanly to bound budget at ~2 hours. Skip #4-#7 until 1-3 close.
