# Qwen3.5-397B-A17B on GCP G4

Optimized configurations and TTFT-focused benchmarks for [Alibaba's Qwen3.5-397B-A17B-FP8](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8) on GCP G4 instances using SGLang.

## Model Overview

Qwen3.5-397B-A17B is a hybrid Gated-Delta-Networks + sparse Mixture-of-Experts model from the Qwen3.5 family (397B total / 17B active per token). Supports up to 262K-token native context.

## Serving Configuration

The model is served using a **2-replica SMG fan-out** (each replica is a single-node TP=8 worker; an `sglang-router` round-robins requests between the two nodes):

- **Tensor Parallelism**: 8 (per replica, single-node)
- **Pipeline Parallelism**: 1
- **Replicas**: 2 (one per G4 node)
- **Router policy**: `round_robin`
- **Quantization**: FP8 block-scaled (W8A8)
- **KV Cache**: auto (model default)
- **Serving Image**: `lmsysorg/sglang:dev-cu13`
- **sglang version**: `0.0.0.dev1+gedb1b3f8f`

## TTFT-focused tuning

This entry's goal differed from the other models in the repo: optimize **first-token latency (TTFT)** rather than steady-state throughput. The workload mirrors a TTFT-sensitive production target — 200 prompts × `--max-concurrency 40` × ISL avg ~10K / OSL avg ~500 (random-range-ratio default 0.0).

The investigation tested **41 server configs** including parallelism shapes (PP=2 cross-node, 2×SMG, PD-Disaggregation with both mooncake and NIXL transports), chunked-prefill-size sweep, speculative decoding (NEXTN, NGRAM), context parallelism, multiple attention/MoE backends, page-size, cuda-graph batch lists, fused kernels, and NCCL channel tuning.

**Key finding**: Mean TTFT is bounded by NCCL allreduce on PCIe-only Blackwell (656 ms per 20K-token prefill, 54 % of the per-prefill wall — measured via torch.profiler). Without NVLink, the mean-TTFT floor is **~2.26 s** on this hardware for this workload.

## Benchmark Results

The following benchmarks were conducted on a cluster of 2× `g4-standard-384` instances (16× RTX PRO 6000 Blackwell GPUs).

| Metric | Result |
|--------|--------|
| Median TTFT | **1180 ms** |
| Mean TTFT | **2256 ms** |
| P99 TTFT | 14320 ms |
| Median TPOT | 53.84 ms |
| Output Throughput | 654.45 tok/s |
| Total Throughput | 14182 tok/s |
| Peak Output Throughput | 1500 tok/s |
| Successful Requests | 200 / 200 |

See [`fp8/results/benchmark_results.md`](fp8/results/benchmark_results.md) for the full bench output and [`fp8/TUNING_REPORT.md`](fp8/TUNING_REPORT.md) for the sweep history.

**Attribution**: Tuning investigation by **Jimmy Shong**.

## Usage

Both nodes run the same `launch_worker.sh`; the router runs on one node and points to both worker endpoints:

```bash
# On both nodes (worker)
NCCL_NCHANNELS=16 ./fp8/launch_worker.sh

# On node-1 only (router)
./fp8/router_launch.sh
```

NCCL/GLOO env vars (load-bearing for PCIe Blackwell) are passed via the top-level repo's `scripts/docker_run_sglang_worker.sh` wrapper.
