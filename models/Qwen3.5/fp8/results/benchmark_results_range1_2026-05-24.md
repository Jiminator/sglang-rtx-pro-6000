# Qwen3.5-397B-A17B-FP8 — Range=1.0 Fresh Paired Baseline (2026-05-24)

Benchmark of the unchanged v38 ship config under the **deterministic fixed-ISL workload** (`--random-range-ratio 1.0`), captured as a paired-seed baseline on `S_panel = {1, 2, 3}`. Same cluster, same image, same launch script as the v38 anchor in `benchmark_results.md`; only the bench's `--random-range-ratio` flips from `0.0` (variable) to `1.0` (fixed ISL=20K, fixed OSL=1K).

- 200 prompts, ISL exactly 20,000, OSL exactly 1,000, `--random-range-ratio 1.0`
- `--max-concurrency 40`, `--request-rate inf`
- Topology: 2× single-node TP=8 + SMG round-robin router (unchanged v38)
- All v38 flags unchanged: `--chunked-prefill-size 4096`, `--max-prefill-tokens 32768`, `--mem-fraction-static 0.8`, `--enable-mixed-chunk`, `--enable-fused-qk-norm-rope`, `--enable-fused-moe-sum-all-reduce`, `--enable-flashinfer-allreduce-fusion`†, `--enforce-piecewise-cuda-graph`, `--disable-radix-cache`
- env unchanged: full NCCL/GLOO block via `scripts/docker_run_sglang_worker.sh` + `NCCL_NCHANNELS=16`

† `--enable-flashinfer-allreduce-fusion` is accepted at ServerArgs parse time but **silently falls back to NCCL ring AR on Blackwell SM120** (dispatcher gates SM90/SM100 only at `layers/communicator.py:158-169`). The flag has been a no-op on this hardware all along; the actual AR is NCCL ring with the env block doing the heavy lifting. See "New closed findings" below.

## Result

**Median TTFT: 1,827.5 ms** (paired median-of-medians across seeds 1, 2, 3 — **below the 2,000 ms TTFT SLO with 172.5 ms margin**).

| Seed | median TTFT (ms) | mean TTFT (ms) | std TTFT (ms) | P99 TTFT (ms) | mean TPOT (ms) | output_throughput (tok/s) | total_throughput (tok/s) | completed | duration (s) |
|:---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| 1 | **1,827.5** | 4,584.5 | 6,747.2 | 28,736.5 | 51.7 | 706.4 | 14,834.1 | 200/200 | 283.1 |
| 2 | **1,721.7** | 4,517.4 | 6,781.9 | 28,812.6 | 52.0 | 703.3 | 14,770.0 | 200/200 | 284.4 |
| 3 | **1,923.4** | 4,631.9 | 6,609.3 | 28,061.4 | 51.6 | 706.6 | 14,837.7 | 200/200 | 283.1 |
| **median-across** | **1,827.5** | 4,584.5 | — | 28,736.5 | 51.7 | **706.4** | 14,834.1 | 200/200 ×3 | ~283.5 |

**Cross-seed std-dev on medians: 100.9 ms.** Per the hill-climb plan's DEC-7 rule, expansion to a 5-seed panel only triggers if std-dev ≥ 200 ms — not needed.

## How this compares to the range=0.0 anchor

The v38 anchor in `benchmark_results.md` is at `--random-range-ratio 0.0` (variable [0, 20K] uniform; effective ISL ~10K average). The range=1.0 version doubles the effective prefill compute per prompt.

| Metric | range=0.0 anchor (`benchmark_results.md`) | range=1.0 paired (this file) | Δ |
|---|---:|---:|---:|
| median TTFT | 1,180 ms | 1,827 ms | +55 % |
| mean TTFT | 2,256 ms | 4,584 ms | +103 % |
| P99 TTFT | 14,320 ms | 28,737 ms | +101 % |
| total_throughput | 14,182 tok/s | 14,834 tok/s | +5 % |

The throughput is slightly higher at range=1.0 because every prompt's prefill is the same large size — fewer wasted scheduler slots than the variable-length range=0.0 distribution. TTFT degrades as expected when every prompt is at the upper end of the ISL distribution.

## How this compares to the prior 5-trial range=1.0 baseline

The prior 5-trial range=1.0 baseline (`_raw/qwen35-v38-fixed-workload-2026-05-22/REPORT.md`) gave per-trial medians `{1758, 1795, 1904, 2127, 2381}` ms with a 5-trial mean-of-medians 1,993 ms. The fresh 3-seed paired baseline `{1722, 1828, 1923}` lies in the lower half of that distribution. The "2,044 ms single-trial reconfirm" cited in the hill-climb DRAFT.md was an upper-tail sample of the same distribution; paired sampling shows v38 is reliably SLO-compliant.

## New closed findings — also added to TUNING_REPORT.md

Two flags retired pre-sweep via Codex source-verification:

| Flag | Status | Why |
|---|---|---|
| `--enable-single-batch-overlap` | **closed inert** for Qwen3.5 | SBO's runtime hookup is DeepSeek-MoE-only (`models/deepseek_v2.py:1028`). Qwen3.5's MoE forward at `models/qwen2_moe.py:461` never tests it. Setting the flag has zero effect on Qwen3.5. |
| `--enable-flashinfer-allreduce-fusion` | **accepted-but-no-op on SM120** | Dispatcher predicate `(is_sm90_supported() or is_sm100_supported())` at `layers/communicator.py:158-169` excludes SM120 (`utils/common.py:254-267` treats SM120/SM100/SM90 as separate capability predicates). v38's launch script carries the flag but the actual AR is NCCL ring. The "Best intra-node AR path" annotation in the original TUNING_REPORT.md is incorrect for this hardware — see TUNING_REPORT.md addendum. |

## Implication for the AR floor

The 656 ms NCCL-AR-per-20K-prefill floor described in TUNING_REPORT.md "Profile-confirmed prefill wall decomposition" is, in light of the FI-AR-fusion-noop finding, **the actual operating point**, not "a fallback below FI fusion". On this hardware, NCCL ring with `NCCL_NCHANNELS=16` is the realized AR path. Any future tuning that targets the AR floor needs to look at NCCL primitive choice or hardware (NVLink, RDMA), not at FlashInfer AR fusion.

## Provenance

- Bundle: `_raw/qwen35-hillclimb-2026-05-24/v0_baseline/`
- Terminal artifact: `_raw/qwen35-hillclimb-2026-05-24/TARGET_ALREADY_MET.md`
- Cross-cuts: `_raw/qwen35-hillclimb-2026-05-24/CLOSED_DIMENSIONS_ADDENDUM.md`
- Sister docs on the parent branch `qwen3.5`: `docs/02_model/MODEL.md`, `docs/04_results_and_profiling/CURRENT_STATE.md`.

## Methodology

- Paired-seed baseline: each variant would be compared seed-for-seed to v0, with `paired_median_delta(vN) = median over s of (median_TTFT(vN, seed s) − median_TTFT(v0, seed s))` against a predeclared `T = 50 ms` decision boundary. No variant ran in this bundle (target already met), so the verdict apparatus was idle — but it is the methodology to use for any follow-up.
- Pre-bench gates (per AC-2 of `_raw/qwen35_hillclimb_handoff/PLAN.md`): workers ready, `/get_server_info` reflects diff, `nvidia-smi` snapshot, 1-prompt smoke valid. All four passed before every seed bench.
- Bundle layout (per AC-1/AC-2): launch_worker.sh, router_launch.sh, docker_run_wrapper_invocation.sh, server_info from both workers + router, gpu_mem pre+postbench, prebench smoke, results.json, bench.log, server logs both nodes, router log, INTERPRETATION.md.
