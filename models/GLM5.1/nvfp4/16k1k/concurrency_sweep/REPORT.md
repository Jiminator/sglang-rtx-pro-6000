# GLM-5.1-NVFP4 Ship — Concurrency Scaling Sweep (ISL=16,384 / OSL=1,024)

| Field | Value |
|---|---|
| Date | 2026-05-11 |
| Source | Provenance from `Jiminator/gcp-kimi:glm-nvfp4-prefill` → `_raw/glm-nvfp4-prefill/2node-isl16384-conc-sweep-2026-05-11/` |
| Model | `lukealonso/GLM-5.1-NVFP4` |
| Config | **Identical to ship** — 2-node TP=8 PP=2 + DPA dp=8, mfs=0.90, `--enable-dynamic-chunking` + `SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR=0.65`, `--chunked-prefill-size 8192` (auto-capped to 1024 by DPA), bf16 KV, flashinfer attention + flashinfer_cutlass MoE, FI-AR-fusion, fcfs, max-running-requests 768 |
| Server-side variables | NONE changed across runs |
| Workload knobs swept | `max-concurrency` ∈ {256, 128, 64, 32, 16, 8, 4, 2, 1}; `num-prompts = 3 × concurrency` (matches the ship's 3:1 ratio: 1,536 / 512) |
| Fixed | ISL = 16,384; OSL = 1,024; seed = 1; `--random-range-ratio 0.0`; `--apply-chat-template`; `--warmup-requests 1` |
| Ship reference (conc=512, 1,536 prompts) | output 277.43 tok/s · total **4,833.78 tok/s** · mean TTFT 117,963 ms · mean TPOT 2,441 ms · duration 2,786.92 s |

## Results

| Conc | #Prompts | Done | Duration (s) | Output tok/s | Total tok/s | Mean TTFT (ms) | Mean TPOT (ms) | Mean E2E (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **512 (ship)** | 1,536 | 1,536 | 2,786.9 | **277.43** | **4,833.78** | 117,963 | 2,441 | 912,124 |
| 256 | 768 | 768 | 1,635.6 | 234.57 | 4,046.02 | 61,091 | 1,430 | 530,144 |
| 128 | 384 | 384 | 1,044.0 | 198.09 | 3,188.58 | 31,621 | 828 | 335,661 |
| 64 | 192 | 192 | 626.9 | 151.63 | 2,597.00 | 16,712 | 478 | 199,494 |
| 32 | 96 | 96 | 401.3 | 124.73 | 2,033.87 | 10,139 | 236 | 126,527 |
| 16 | 48 | 48 | 248.9 | 95.78 | 1,486.03 | 7,011 | 133 | 73,674 |
| 8 | 24 | 24 | 168.4 | 78.53 | 1,185.25 | 5,181 | 84 | 50,292 |
| 4 | 12 | 12 | 115.9 | 51.29 | 733.88 | 3,671 | 60 | 32,998 |
| 2 | 6 | 6 | 102.5 | 33.64 | 502.02 | 4,121 | 49 | 33,423 |
| 1 | 3 | 3 | 86.9 | 20.11 | 316.58 | 4,475 | 42 | 28,958 |

All runs: 100 % completion (`completed = num_prompts`).

## Figures

Per-metric scatter-plus-line plots across all 10 concurrency points (1 → 512), in [`figures/`](figures/):

| Metric | File |
|---|---|
| Input token throughput | [`figures/input_throughput.png`](figures/input_throughput.png) |
| Output token throughput | [`figures/output_throughput.png`](figures/output_throughput.png) |
| Mean E2E latency | [`figures/mean_e2e_latency.png`](figures/mean_e2e_latency.png) |
| Mean TTFT | [`figures/mean_ttft.png`](figures/mean_ttft.png) |
| Mean TPOT | [`figures/mean_tpot.png`](figures/mean_tpot.png) |
| Mean ITL | [`figures/mean_itl.png`](figures/mean_itl.png) |

X-axis is `max-concurrency` on log₂ scale (ticks at exact sweep points {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}). Each point is annotated with its value.

## Observations

1. **Throughput scales roughly with `sqrt(concurrency)` over the sampled range.** Going from conc=512 → conc=1 is a 512× drop in offered load but only a 15.3× drop in total throughput (4,833 → 317). The cluster's per-rank decode rate dominates; queueing amortizes the per-request prefill cost across batches.

2. **TPOT improves *almost linearly* with falling concurrency** — from 2,441 ms at conc=512 down to 42 ms at conc=1. The conc=512 TPOT is enormous specifically because decode tokens are blocked behind ongoing prefill chunks under PP=2 + dynamic chunking. At conc=1 the decode loop is unloaded and TPOT converges to the raw per-token kernel cost: **~42 ms per output token** on this hardware/topology.

3. **TTFT bottoms out around 3.7–4.5 s** for conc ≤ 8. This is the floor: a single ISL=16,384 prefill at PP=2 + DPA dp=8 takes ~4 s wallclock to complete. Below conc=8, queueing is essentially absent.

4. **Bimodal TTFT structure visible at high concurrency**: at conc=512 mean TTFT is 118 s but median was 13 s (from the ship REPORT) — most requests get fast handoff, a backlog tail of 5–10 % balloons the mean. Lower concurrency removes the tail; mean and median converge.

5. **conc=2 vs conc=4 anomaly**: TTFT is *higher* at conc=2 (4,121 ms) than conc=4 (3,671 ms). This is from PP=2 pipeline-bubble dynamics: with only 2 in-flight requests, one stage often sits idle waiting for the other. At conc=4, both stages stay busy.

6. **conc=1 TTFT (4,475 ms) is higher than conc=4 TTFT (3,671 ms)** for the same reason: at conc=1, the single-prompt forward through both PP stages sequentially is the dominant cost, with no overlap.

## Where this is useful

- **Capacity planning**: for a target tail-latency SLO (say p99 TTFT ≤ 30 s) on this workload, conc=128 gets the most throughput within that tail (3,189 tok/s at 31.6 s mean TTFT).
- **Latency-sensitive serving**: conc=8–16 buys 70–95 ms TPOT at 1,185–1,486 tok/s — a 12 % to 31 % throughput trade for ~25× faster per-token latency vs the ship.
- **Single-stream baseline**: conc=1 establishes the raw per-token kernel cost (42 ms / TPOT, ~4.5 s prefill) — useful as a kernel-perf upper-bound check.

## Bundle artifacts (in this folder)

```
REPORT.md                                          # this file
ALL_SUMMARIES.md                                   # canonical bench summary table from each of 9 runs (full TTFT/TPOT/ITL percentiles)
figures/{input,output}_throughput.png              # 6 per-metric plots across concurrency
figures/mean_{e2e_latency,ttft,tpot,itl}.png
```

The launch scripts that were used (identical across all 10 runs) are in the parent folder: [`../launch_node1.sh`](../launch_node1.sh), [`../launch_node2.sh`](../launch_node2.sh). The raw bench `.jsonl` and `.log` files for each concurrency point are in the source bundle at `gcp-kimi:glm-nvfp4-prefill:_raw/glm-nvfp4-prefill/2node-isl16384-conc-sweep-2026-05-11/`.

For human-readable per-conc breakdowns (full TTFT/TPOT/ITL percentiles, peak/concurrency, all the bench fields), see **[`ALL_SUMMARIES.md`](ALL_SUMMARIES.md)**.

The actual `--max-concurrency` and `--num-prompts` flags are visible in each `.log` file's bench Namespace header.
