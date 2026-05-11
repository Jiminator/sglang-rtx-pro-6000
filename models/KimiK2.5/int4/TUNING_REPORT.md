# Kimi-K2.5-INT4 — Tuning Report

**As of 2026-05-11.** Ship config: **3,206.10 tok/s output, 3,597.20 total**, 1,536 / 1,536 completed — see [`results/benchmark_results.md`](results/benchmark_results.md) and [`launch_node.sh`](launch_node.sh).

## Model context

`moonshotai/Kimi-K2.5` uses **compressed-tensors INT4 group=32** with FP8 KV cache. Weights are **~555 GB on disk** (64 shards). This is the model's binding constraint on this hardware: at single-node TP=8 each GPU holds ~70 GB of weights out of 96 GB RTX PRO 6000 capacity — only ~26 GB left for KV pool + cudagraph + activation. The PP=2 cross-node split is the **only** topology that gives adequate KV-pool headroom for the bench's 512-concurrency target.

## Workload (fixed)

| Field | Value |
|---|---|
| Dataset | random, 1,536 prompts |
| ISL / OSL | 1,024 / 8,192 (decode-dominant, mirror of DSv3.2 standard workload) |
| Concurrency | max-concurrency 512, request rate 9999 |
| Seed | 1 |
| `--apply-chat-template` | yes |

## Hardware & topology baseline

| Field | Value |
|---|---|
| Cluster | 2× GCP `g4-standard-384`, 16× RTX PRO 6000 Blackwell SM120 |
| Interconnect | 2× 200 Gbps gVNIC TCP, **no NVLink, no RDMA** |
| Image | `lmsysorg/sglang:dev-cu13` |
| **sglang version (pinned)** | **`0.5.10.post1`** (pip-installed inside the container, overriding image's `0.0.0.dev1+gcbc2bee54`) |
| `FLASHINFER_DISABLE_VERSION_CHECK` | `1` (bypasses cubin/wrapper drift after the pin) |
| Parallelism (ship) | 2-node TP=8 PP=2 + DPA dp_size=8 |
| `--kv-cache-dtype` | `fp8_e5m2` (halves KV memory vs bf16) |
| `--page-size` | 1 (default; not page=64 like DSv3.2) |
| `--chunked-prefill-size` | 16,384 (auto-capped to 2,048 by DPA) |
| Mandatory env vars | full NCCL/GLOO set (see top-level repo's `scripts/docker_run_sglang_worker.sh`) |

## Why the sglang 0.5.10.post1 pin is load-bearing

The `dev-cu13` image's bundled sglang dev build has a **cudagraph buffer regression on triton attention** that drops K2.5-INT4 throughput by ~5 %. Pinning `sglang==0.5.10.post1` via `pip install` at container start avoids the regression but introduces version drift between sglang's flashinfer wrapper (0.6.7.post3) and the image's pre-built cubins (0.6.8.post1) — handled by `FLASHINFER_DISABLE_VERSION_CHECK=1`. Once the upstream regression is fixed, the pin can be dropped.

## What we shipped

The ship config is the baseline of a layered hill-climb on the pinned-sglang version, applying 5 standard optimizations:

| Layer | Flag | Mechanism |
|---|---|---|
| 1 | `--enable-dp-attention --dp-size 8` | DPA across 8 attention-DP ranks per node |
| 2 | `--enable-dp-lm-head` | DP LM head — avoids the AR-after-softmax on each step |
| 3 | `--enable-fused-moe-sum-all-reduce` | Fuses the MoE-output reduce into the FFN downcast |
| 4 | `--disable-shared-experts-fusion` | Compatibility with K2.5 checkpoint shape |
| 5 | `--disable-custom-all-reduce` + (no `--enable-flashinfer-allreduce-fusion`) | NCCL AR is the working AR path on PCIe Blackwell; FI-AR-fusion is silent no-op + has wrapper/cubin drift after the pin |

Result: **3,206.10 tok/s output, 3,597.20 total, 1,536/1,536, mean TTFT 4.7 s, mean TPOT 134 ms**.

## Closed dimensions — do NOT rerun without new source-code reason

| Lever attempted | Mechanism aimed at | Result | Root cause |
|---|---|---|---|
| 2× SMG replica with **DPA dp_size=8** (single-node per replica, mfs=0.92, max-rr=128, cgmb=128) | replace cross-node PP=2 with two independent single-node servers behind a router; goal is to remove PP P2P overhead | **418.66 tok/s, 389/1,536 completed (−87 %)** | KV pool collapses to ~8,569 tokens / DP rank (~68K / cluster). Per-request KV state at ISL+OSL=9,216 means the pool barely fits one in-flight request per DP rank. 75 % of requests time out. |
| 2× SMG replica with **dp_size=1, no DPA** (pure TP=8 attention sharding) | recover KV pool by avoiding DPA's per-DP-rank duplication | **1,805.61 tok/s, 1,536/1,536 completed (−43.7 %)** | KV pool stabilizes at 227K tokens / replica (455K cluster). Bench completes, but mean TTFT = **11.4 min**; pool is ~5× undersized for the bench's 2.3M-token peak demand. Per-token TPOT actually improved (67 vs 134 ms) but the queue cost destroys throughput. |
| ktransformers CPU-expert offload (offload MoE experts to CPU to free GPU memory for KV) | mitigation for the single-node weight footprint problem above | **infrastructure-blocked** | (1) kt-kernel pre-built CUDA wheels target SM 80/86/89/90 only — Blackwell SM 12.0 not in the support matrix. (2) Official ktransformers image is CUDA 12.1; cluster requires CUDA 13. (3) `pip install kt-kernel` inside dev-cu13 pulls torch 2.9.1 (incompatible with image's 2.11.0) → `ImportError: libtorch_cuda.so: undefined symbol: ncclDevCommDestroy`. (4) `pip install sglang-kt` breaks the same way. Would require a custom container build with sm_120 codegen — multi-hour-to-days infra project. |
| `--enable-flashinfer-allreduce-fusion` | overlap AR with later compute | dropped from ship | Silent no-op on PCIe Blackwell (no NVL multicast substrate, no NVLS, no working PCIe-oneshot — see `project_pcie_oneshot_ar_broken_on_g4`). Also, flashinfer wrapper 0.6.7.post3 (from the pin) vs image's 0.6.8.post1 cubins → drift risk at the AR kernel call. Net: stack with no AR fusion is the safe ship. |
| `--mem-fraction-static ≥ 0.86` | grow KV pool further | OOM on FusedMoE workspace | At PP=2 dp=8 each GPU holds ~35 GB weights + ~50 GB KV — leaving ~10 GB for activation. Pushing mfs higher squeezes activation; the FusedMoE expert dispatch workspace is what blows first. mfs=0.85 is the structural floor. |

## Promising but unmeasured (open levers, ranked)

| Idea | Mechanism | Expected delta | Cost | Risk / notes |
|---|---|---:|---|---|
| **1. Custom kt-kernel build with `TORCH_CUDA_ARCH_LIST="12.0"`** | enable MoE expert offload to CPU via kt-kernel AMX-INT4. Frees ~10-20 GB GPU memory per node → grows KV pool by 25-50 %. **If sm_120 codegen actually has compatible primitives**, this could enable a single-node 2× SMG replica strategy that recovers KV pool. | **+10 to +25 %** | High: hours of CMake/CUDA toolchain work to build kt-kernel from source. May still fail at the kernel-codegen level if Blackwell-specific intrinsics aren't compatible. | Largest theoretical lever; highest implementation risk. Out of scope for a single bench session. |
| **2. Wait for triton-attention cudagraph fix in dev-cu13's bundled sglang** | drop the `sglang==0.5.10.post1` pin and simplify the ship config | **+0 to +3 %** (recovers the 5 % we currently spend on the pin, modulo updated kernel performance) | Zero — passive monitoring. | Sometime in sglang main branch. Re-test once a new dev-cu13 image lands. |
| **3. Push concurrency past 512** | current ship's peak KV usage is only 14 %; lots of pool headroom remains | **+5 to +15 %** | Workload args are normally fixed (`max-concurrency 512`); this would require a separate bench. | Worth doing as a diagnostic to bound the max useful concurrency. If it lands +10 % at conc=768, that confirms the ship is under-saturated and motivates other levers (more concurrency = more overlap). |
| **4. `--enable-prefill-context-parallel`** | shards each prefill across DP groups, releasing more per-GPU work | **+1 to +5 %** | requires nnodes ≥ 3 to give a meaningful CP shard. We only have 2 nodes. | Closed for now on this cluster; revisit if we get a 3rd node. |
| **5. `--kv-cache-dtype fp8_e4m3` (vs current fp8_e5m2)** | different fp8 mantissa/exponent; potential accuracy boost or kernel speedup | **−1 to +1 %** | 1 flag; ~35 min bench. | Mantissa/exponent trade-off usually doesn't change throughput materially. Low priority. |
| **6. Mixed-chunk scheduling** (`--enable-mixed-chunk`) | interleave decode tokens inside prefill chunks | **+2 to +5 %** | 1 flag; ~35 min bench. | DSv3.2 ships with this flag enabled in the K8s manifest. K2.5-INT4 hasn't tested it on the pinned-sglang stack. May OOM as it did on GLM-5.1 at long-ISL — but this workload's ISL is only 1,024, so the scratch peak should fit. Worth a try. |
| **7. `--enable-two-batch-overlap`** | overlap two micro-batches per PP stage | **+3 to +8 %** | requires `moe_a2a_backend != none` + `ep_size > 1`. Same source-code limitations as GLM-NVFP4. | Bound to the same upstream A2A backend issues; unlikely to be testable until sglang patches the flashinfer A2A init path. |

## Closed structural / hardware constraints (not flag-tunable)

| Constraint | Source |
|---|---|
| Kimi-K2.5 weights at 555 GB total — single-node TP=8 leaves no usable KV-pool headroom. PP=2 cross-node is mandatory. | This investigation. |
| `--pp-async-batch-depth ≥ 1` deadlocks on PP=2 + DPA + cross-node TCP. | `project_pp_async_batch_depth_deadlocks` |
| `dev-cu13` image's bundled sglang has triton-attention cudagraph regression. Must pin to 0.5.10.post1 for now. | This investigation. |
| `--enable-pcie-oneshot-allreduce` is dead code on g4 + SM120 + CUDA 13. | `project_pcie_oneshot_ar_broken_on_g4` |
| kt-kernel pre-built wheels do not target sm_120. Custom build required. | This investigation. |

## Hardware upgrades that would change the picture

- **NVLink / NVSwitch**: would unlock PCIe-oneshot AR (or NCCL NVLS path) for ~+5-15 % intra-node throughput. Would also make single-node TP=8 with all 555 GB weights feasible (NVLink lets ranks share GPU memory at higher bandwidth, though doesn't add capacity).
- **More memory per GPU (≥ 128 GB)**: would let us fit K2.5-INT4 at TP=8 single-node *with* meaningful KV pool headroom, unlocking the 2× SMG strategy without ktransformers. Hopper H200 (141 GB) or B200 (192 GB) would suffice.
- **Third node**: would unlock `--enable-prefill-context-parallel`.
- **Custom Blackwell-compatible kt-kernel build**: software-only; could be done on existing hardware with a multi-day build effort.

## Suggested experiment order for next session

1. **`--enable-mixed-chunk`** on the existing ship — if it doesn't OOM at this ISL (it's only 1,024, much smaller than the GLM 16K case where it failed), it should land a modest win. ~35 min.
2. **Push concurrency to 768 or 1024** as a diagnostic — bounds the effective ceiling and informs whether KV-pool is the limit or something else.
3. **Watch for sglang upstream**: drop the 0.5.10.post1 pin once the triton-attention cudagraph fix lands.
4. **kt-kernel custom build** (separate research project): commit a day to build from source with `TORCH_CUDA_ARCH_LIST="12.0"`, then re-test 2× SMG with CPU expert offload.

Stop after #1 and #2 to bound the budget at ~2 hours.
