# GLM-5.2-NVFP4 on RTX PRO 6000 (SM120) — serving recipe + throughput (latest main, 2026-07-05)

Single node, 8× RTX PRO 6000 Blackwell (SM120, PCIe, no NVLink), **TP=8 + DP-attention (dp8)**. Measured
with `sglang.bench_serving` (loadgen, `--random-range-ratio 1.0`, no zipfian) → steady-state decode plateau
read from server logs (aggregate = per-rank gen-throughput × dp_size 8; /GPU = ÷8). Checkpoint:
[`nvidia/GLM-5.2-NVFP4`](https://huggingface.co/nvidia/GLM-5.2-NVFP4) (`GlmMoeDsa` DSA arch), snapshot
`b0b2b68`. Image: **stock latest-main `lmsysorg/sglang:dev-cu13`** (sglang `0.0.0.dev1+gb28bc1060`,
transformers 5.12.1). No source patch, no custom branch.

## Headline: fp8_e4m3 KV DSA decode now works on STOCK latest main

The prior campaign's fp8 SOTA was **glm-opt-branch-only** — on stock dev-cu13 the fp8 DSA decode path
crashed (no SM120 DSA decode kernel: `TllmGenFmhaRunner` SM100-only, etc.). **On latest main that crash is
gone** — the trtllm DSA decode path itself now runs on SM120, so fp8 KV works on the stock image with a
single env var. fp8's ~1.8–2.5× larger KV pool lifts the concurrent-sequence ceiling at saturation, which
is what raises the decode plateau. bf16 is now a conservative fallback, not the ceiling.

**The only env var needed for fp8 is `SGLANG_DISABLE_DSA_INDEXER_FUSION=1`** (ablation A1a: gsm8k 0.940).
The DSV4 vars (`SGLANG_SM120_FLASHMLA_BACKEND`, `SGLANG_OPT_FLASHMLA_SPARSE_PREFILL`) are **inert** for GLM —
GLM DSA uses a different dispatch. Also pass `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for boot and
the load-bearing NCCL/GLOO block for perf.

## Throughput (aggregate output tok/s; /GPU = ÷8)

| Config | KV | mfs | pool/rank | **1K/8K** (1024/8192) | **8K/64K** (8192/65536) | gsm8k |
|---|---|---:|---:|---:|---:|---:|
| bf16 baseline | bfloat16 | 0.94 | 91.5K | ~1264 (158/GPU) | ~984 (123/GPU) | 0.920 |
| **non-spec fp8** 🥇 | fp8_e4m3 | 0.975 | 229.7K | **~2680 (335/GPU)** | **~1081 (135/GPU)** | 0.900 |
| **spec+fp8 (EAGLE-3)** | fp8_e4m3 | 0.97 | 128K | ~2645 (330/GPU) | — | **0.940** |

**Winner: fp8 KV, both workloads.** 1K/8K non-spec (335/GPU) ≈ spec (330/GPU) — **tied within noise** (both
pool-bound at mfs~0.97–0.975); spec carries the correctness (0.940 vs 0.920) and latency (accept-len 4.0)
edge. fp8 is **+108% over the bf16 plateau (158→335/GPU)** at 1K/8K and **+10%** at 8K/64K (via the
concurrency-ceiling lift: 16 running-req/rank vs bf16's 9). Bundles: [`nvfp4/1k8k/`](nvfp4/1k8k/),
[`nvfp4/8k64k/`](nvfp4/8k64k/). The zai-org full-FP8 checkpoint is [`fp8/`](fp8/) (**2-node BLOCKED**).

> Prior glm-opt-branch bundle (superseded): offline `bench_one_batch_server` one-shot 194.6 tok/s/GPU
> (non-spec) / ~300 sustained (EAGLE), 1K/8K only. Different harness — **not comparable** to these
> `bench_serving` plateau numbers. Old Pareto pngs kept under [`results/`](results/) as history.

## Fixed facts

- **MoE runner = `flashinfer_cutlass`** for NVFP4 (cutedsl/trtllm have no SM120 build; marlin gives gsm8k
  0.02 = garbage). This is the only viable NVFP4 MoE runner on SM120.
- **Required flags:** `--attention-backend flashinfer` + `--kv-cache-dtype fp8_e4m3` +
  `--disable-shared-experts-fusion` (loader crashes without it) + `--moe-a2a-backend none --ep-size 1`.
- **`dp_size` must equal `tp_size`** (=8). DP-attention gives each GPU its own KV pool.
- **Pool is the lever.** Decode is KV-pool-bound: each memory-freeing lever (fp8 KV, minimal per-worker
  cuda-graph buckets, chunked-prefill shrink) buys a higher `--mem-fraction-static` notch → bigger pool →
  more concurrent seqs → higher plateau. **mfs 0.975 is the ceiling** (0.98 OOMs at boot).
- **Spec = EAGLE topk-1 only.** `--speculative-eagle-topk >1` (tree spec) is hard-blocked on SM120
  (flashinfer-MLA is topk=1 only). Pin `--speculative-moe-runner-backend flashinfer_cutlass
  --speculative-moe-a2a-backend none` to dodge the dead deep_gemm/deepep auto-route.

## FP8 checkpoint (zai-org/GLM-5.2-FP8) — 2-node BLOCKED

The full-FP8 704 GB checkpoint needs 2 nodes at TP=16. **Blocked on latest main by a cross-node warmup
deadlock.** Distributed init + weight load succeed (~83.7 GB/GPU), then the first warmup forward hangs
permanently at `multimem all-gather disabled (Failed to send fd: No such file or directory)` — sglang's
CUDA symm-mem multimem collective does a CUDA-IPC fd exchange over a Unix socket that cannot work across two
nodes with no shared FS; it disables multimem then deadlocks on the next collective (the 2400s watchdog
never fires — ranks spin). 5 launch configs all identical. No-code path exhausted. See [`fp8/README.md`](fp8/README.md)
for the correct recipe (for when the substrate is fixed) and the remaining source-patch / PP-across-nodes options.

## Full campaign data

`runs/20260705_glm5.2_sota_humanize/` in the gcp-kimi repo (hill-climb ledger, ablations, correctness gates,
per-config server logs). Per-workload stories: [`nvfp4/1k8k/TUNING_REPORT.md`](nvfp4/1k8k/TUNING_REPORT.md),
[`nvfp4/8k64k/TUNING_REPORT.md`](nvfp4/8k64k/TUNING_REPORT.md).

**Attribution:** GLM-5.2-NVFP4 SM120 tuning campaign — RadixArk serving team.
