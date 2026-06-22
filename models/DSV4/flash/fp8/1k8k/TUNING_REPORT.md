# DeepSeek-V4-Flash FP8 — Tuning Report (1K/8K, single node, SM120)

**As of 2026-06-22.** Workload **ISL 1024 / OSL 8192**, offline `bench_one_batch_server`, batch maxed
to the KV pool for every config. Ship config: **TP=8 + DP-attention**, triton MoE, FP8.

## Headline

**FP8 + DP-attention = 3412.9 output tok/s @ batch 1016 (≈426.6 / GPU)** — **7.1× over pure TP=8**
(481.4 tok/s). See [`results/benchmark_results.md`](results/benchmark_results.md) and
[`launch_FP8_dpa_1k8k.sh`](launch_FP8_dpa_1k8k.sh).

## Results (offline `bench_one_batch_server`, ISL 1024 / OSL 8192, batch maxed)

| Config | batch | output tok/s | /GPU | **tok/s per req** | note |
|---|---:|---:|---:|---:|---|
| **FP8 + DP-attention** | 1016 | **3412.9** | **426.6** | 3.36 | **ship** |
| FP8 + DPA, swa-ratio 0.16 | 1120 | 3357.9 | 419.7 | 3.00 | past the knee — no gain |
| FP8 + DPA, swa-ratio 0.20 | 1160 | 3336.7 | 417.1 | 2.88 | past the knee — slight drop |
| FP8 2× TP=4 replicas | 72 | 678.1 | 84.8 | 9.41 | small-TP is fast/req but pool-starved |
| FP8 TP=8 pure | 146 | 481.4 | 60.2 | 3.30 | single SWA pool caps batch |
| FP8 DPA + EAGLE/MTP spec | 624 | ~250* | ~31 | 0.40 | **13.6× net loss** — draft+verify overhead at batch; *log-derived |
| FP4 DP-attention (marlin) | ~1336 | ~497* | ~62 | 0.37 | MoE-compute-saturated; *log-derived (bench >100min) |
| FP4 TP=8 (marlin) | 185 | 266.7 | 33.3 | 1.44 | mxfp4 MoE tax + single pool |

(All 8 GPUs, single node, so /GPU == aggregate/8. *FP4-DPA bench exceeded the 100-min cap; value is
the steady-state aggregate decode rate from the server log, 0 retractions, batch 167/worker.)

## Why DP-attention wins — and what the lever really is

The 7.1× is **batch-scaling on a memory-bandwidth-bound decode, not a compute speedup.** Evidence:
pure TP=8 (batch 146) and DP-attention (batch 1016) have **identical per-request throughput**
(3.30 vs 3.36 tok/s/req). DPA didn't make the math faster — it ran **7× more requests**. The check is
linear: `481 × (1016/146) = 3348 ≈ 3413`.

Decode is **memory-bandwidth-bound**: loading the MoE expert weights each step dominates, *independent*
of how many tokens ride along. So throughput scales ~linearly with batch until the bandwidth→compute
knee (~1016 here). Pure TP=8 at batch 146 was loading all the weights every step but only processing
146 tokens — **the GPU was starved, not saturated.**

**What caps the batch is the hybrid attention memory.** DeepSeek-V4-Flash splits KV into a large
full-attention pool + a small **SWA/DSA-sparse pool** (`swa_full_tokens_ratio` ≈ 0.1 of memory,
~1792 sparse-tokens/seq at OSL 8192). That small pool — not `max_total_num_tokens` — is the batch
ceiling. Pure TP=8 has **one** SWA pool → batch 146. **DP-attention gives each of the 8 GPUs its own
full attention pool → ~1016.** That 8× SWA-pool capacity is the entire lever.

## Saturation — is 3413 the ceiling?

Yes. The `swa_full_tokens_ratio` sweep added batch past 1016 (→1120→1160) and got **no throughput
gain** — per-request rate fell proportionally (3.36 → 3.00 → 2.88). So ~1016 is the
bandwidth→compute knee and **3413 is the real ceiling** for this config at 1K/8K. Raising the ratio is
neutral-to-negative; **ship the default ratio (0.1).**

## Workload (fixed)

| Field | Value |
|---|---|
| Benchmark | `sglang.bench_one_batch_server` (offline, fixed batch) |
| ISL | 1,024 | 
| OSL | 8,192 |
| Batch | maxed to the SWA pool per config (`0.1 × pool / ~2048` per worker, ×`dp_size` for DPA) |

## Hardware & topology baseline

| Field | Value |
|---|---|
| Node | 1× GCP `g4-standard-384`, 8× RTX PRO 6000 Blackwell **SM120** |
| Interconnect | intra-node PCIe, **no NVLink** |
| Image | `lmsysorg/sglang:v0.5.13.post1-cu130` (sglang v0.5.13.post1) |
| Parallelism (ship) | **TP=8 + DP-attention** (`--dp-size 8 --enable-dp-attention`), no spec |
| Mandatory env | full NCCL/GLOO set + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

## Closed dimensions (mechanism-backed)

| Variant | Verdict |
|---|---|
| **DP-attention** | **WIN, 7.1×** → ship. 8 per-worker SWA pools lift the batch ceiling 146→1016 on a BW-bound decode. |
| FP8 TP=8 pure | batch-starved by its single SWA pool (146) → 481. Not compute-saturated. |
| FP8 2× TP=4 | per-req **2.8× faster** than TP=8 (smaller PCIe-AR domain) but only 2 small SWA pools → batch 72 → 678. |
| FP4 TP=8 (marlin) | 266 — single SWA pool **and** the mxfp4 MoE GEMV tax halves per-req rate (1.44). FP4 needs `--moe-runner-backend marlin` (triton routes mxfp4 through the FP8 path → hidden-size crash). |
| **FP4 DP-attention** | ~497 (62/GPU) — DPA lifts FP4's batch to ~1336 but the mxfp4 MoE GEMV becomes **compute-bound** (per-req collapses to 0.37). **6.9× below FP8 DPA** → FP4 conclusively loses at 1K/8K. |
| FP4 2×TP=4 / 4×TP=2 | skipped — fewer SWA pools than FP4-DPA **and** the same MoE tax → strictly worse than the 497 above. |
| `swa_full_tokens_ratio` 0.16 / 0.20 | neutral-to-negative; decode saturated at batch ~1016. |
| EAGLE/MTP spec (FP8 DPA) | **13.6× net loss** (~250 vs 3413). Spec shrinks the pool (2.62M→1.6M → batch 624) **and** the draft+verify forwards add huge per-step cost at batch — per-req rate collapses to 0.40 despite 2-token acceptance. Spec is a latency lever, not throughput; consistent with the 8K/64K finding. (DSV4 supports only `--speculative-algorithm EAGLE`.) |

## Open / interesting lead

C's per-request rate (9.41) is ~2.8× TP=8's — the no-NVLink **PCIe all-reduce cost of TP=8** is real.
A topology with both small TP domains *and* many large SWA pools could in principle beat DPA, but FP8
doesn't fit at TP=2 and FP4 carries the MoE tax. Worth a profiling pass to confirm the BW-bound /
AR-cost split — the one remaining lever that could push past 3413.

All planned configs are complete; the sweep is closed.
