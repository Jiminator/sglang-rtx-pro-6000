# OSS Model Benchmarks on GCP G4

Optimized GKE configurations and benchmarks for serving LLMs on GCP G4 instances using SGLang.

## Infrastructure
- **GPU**: NVIDIA RTX PRO 6000 Blackwell (SM120)
- **Architecture Details**: [Technical Specifications: GCP G4](./gcp_g4_specs.md)
- **Serving Framework**: [SGLang](https://github.com/sgl-project/sglang) (`dev-cu13`, `0.5.10.post1`)

## Performance Benchmarks (Latest)

| Model | Quantization | Setup | Output Throughput (tok/s) | Total Throughput (tok/s) | Peak Throughput (tok/s) | TPOT (ms) |
|-------|--------------|-------|---------------------------|--------------------------|-------------------------|-----------|
| [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) | FP8 | 2 Nodes (16x RTX 6000) | 2962.79 | 3324.21 | 4951.00 | 149.29 |
| [DeepSeek-V3.2](https://huggingface.co/nvidia/DeepSeek-V3.2-NVFP4) | NVFP4 | 1 Node (8x RTX 6000) | 2675.33 | 3012.42 | 2046.00 | 106.03 |
| [GLM-5.1](https://huggingface.co/zai-org/GLM-5.1-FP8) | FP8 | 2 Nodes (16x RTX 6000) | 2785.55 | 3125.35 | 4092.00 | 155.26 |
| [GLM-5.1](https://huggingface.co/lukealonso/GLM-5.1-NVFP4) | NVFP4 | 1 Node (8x RTX 6000) | 1462.73 | 1641.16 | 950.00 | 107.02 |
| [GLM-5.2](https://huggingface.co/nvidia/GLM-5.2-NVFP4) | NVFP4 | 1 Node (8x RTX 6000)‖ | 2145.76 | 2413.98 | n/a | 43.15 |
| [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2-FP8) | FP8 | 2 Nodes (16x RTX 6000)¶ | 2190 | 2464 | n/a | ~215 |
| [Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5) | INT4* | 2 Nodes (16x RTX 6000) | 3069.15 | 3443.55 | 6889.00 | 147.45 |
| [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8) | FP8 | 2 Nodes (16x RTX 6000)† | 654.45 | 14182.22 | 1500.00 | 56.46 |
| [DeepSeek-V4-Flash](https://huggingface.co/sgl-project/DeepSeek-V4-Flash-FP8) | FP8 | 1 Node (8x RTX 6000)‡ | 551.92 | 608.47 | n/a | ~60 |
| [DeepSeek-V4-Flash](https://huggingface.co/sgl-project/DeepSeek-V4-Flash-FP8) | FP8 | 1 Node (8x RTX 6000)§ | 3412.91 | 3482.25 | n/a | n/a |

*Benchmarks conducted using `inf` request rate and 512 max concurrency. Tests utilized a random dataset with 1024 input tokens and 8192 output tokens (1536 total prompts). The load generator was isolated on a dedicated CPU-only node pool to ensure zero interference with GPU performance.*

*\*Kimi-K2.5 uses native INT4 quantization for model weights and FP8 for the KV cache to optimize memory efficiency and inference speed.*

*†Qwen3.5 was tuned for **TTFT** rather than steady-state throughput, using a different workload (200 prompts × max-concurrency 40 × ISL target 20K / OSL target 1K, random-range-ratio 0.0 → variable lengths). Result: **median TTFT 1180 ms, mean TTFT 2256 ms**. Throughput numbers in this row reflect that workload and are not directly comparable to the other entries' steady-state numbers. Topology is 2× single-node TP=8 replicas behind an `sglang-router` (SMG fan-out), not cross-node PP. See [`models/Qwen3.5/fp8/TUNING_REPORT.md`](./models/Qwen3.5/fp8/TUNING_REPORT.md).*

*‡DeepSeek-V4-Flash was tuned for a **long-context** workload (ISL 8192 / OSL 65536) and measured with **offline** `bench_one_batch_server` at the KV-pool-capped batch (33), single node. Numbers are not directly comparable to the other rows' online 1K/8K results. The win is **DP-attention** (`--dp-size 8 --enable-dp-attention`), **2.45× over pure TP=8** — DeepSeek-V4-Flash decode is DSA-sparse-attention-bound (83% of decode), and DP-attention runs that kernel as 8 independent per-worker streams. "Total" is overall (in+out) throughput; offline runs have no peak/TPOT (≈60 ms ITL). See [`models/DSV4/flash/fp8/8k64k/TUNING_REPORT.md`](./models/DSV4/flash/fp8/8k64k/TUNING_REPORT.md).*

*§DeepSeek-V4-Flash on a **1024 / 8192** workload, **offline** `bench_one_batch_server`, batch maxed to the KV pool (1016), single node. Not comparable to the online rows. The win is again **DP-attention**, **7.1× over pure TP=8** (481 tok/s): decode is memory-bandwidth-bound so throughput scales with batch, and DP-attention's 8 per-worker SWA pools lift the batch ceiling from 146 (single pool) to 1016 — per-request rate is identical, so the gain is pure batch-scaling. FP4 loses (mxfp4 MoE compute-bound at batch). See [`models/DSV4/flash/fp8/1k8k/TUNING_REPORT.md`](./models/DSV4/flash/fp8/1k8k/TUNING_REPORT.md).*

*‖GLM-5.2-NVFP4, single node, **EAGLE 3-step speculative decoding** (the throughput SOTA) on the SGLang `glm-opt` branch — fp8 KV + DP-attention, 1K/8K, `bench_serving` sustained at max-concurrency 100 (its KV-pool knee). accept-length ≈ 3.9, gsm8k 0.94. Output 2145.76 tok/s ≈ **268 tok/s/GPU** whole-run (≈300/GPU at the steady-state saturation window). EAGLE Pareto-dominates the non-spec config across all concurrency; non-spec fp8 max-batch one-shot = 194.6 tok/s/GPU, stock bf16-dense baseline = 147. See [`models/GLM5.2/README.md`](./models/GLM5.2/README.md).*

*¶GLM-5.2-**FP8** (zai-org full-FP8, 704 GB) — the checkpoint for customers who require **FP8 weights**. Needs **2 nodes**: TP=8 × **PP=2** + DP-attention, stock `dev-cu13` (v0.5.15), **no source patch** (the PP boundary is fixed by `SGLANG_PP_LAYER_PARTITION=38,40`). 1K/8K `bench_serving` steady-state plateau = **137 tok/s/GPU** (2,190 output tok/s; per-DP-rank decode pinned at 273 tok/s × 8 ÷ 16); total/TPOT derived from the ≈215 ms steady-state ITL, no peak. **Throughput-inferior to the GLM-5.2-NVFP4 row above** — on SM120 it's forced onto dense MLA (DSA sparse decode is SM100-only) and cannot use speculative decoding (3 independent SM120 walls: PP-assert / flashinfer-draft-`topk_indices` / DSA-logits-SM100-only). 8K/64K = 24.85 tok/s/GPU. Use full-FP8 only when FP8 weights are mandated. See [`models/GLM5.2/fp8/README.md`](./models/GLM5.2/fp8/README.md).*

## Project Structure

- `models/`: Model-specific SGLang job configurations and benchmarks.
  - `DeepSeekv3-2/`: Configs for DeepSeek-V3 and V2.5.
    - `fp8/`: Optimized 2-node FP8 serving setup.
    - `nvp4/`: Native FP4 serving using `modelopt_fp4` with EAGLE speculative decoding.
  - `GLM5.1/`: Optimized configurations and results for GLM-5.1.
    - `fp8/`: 2-node FP8 serving optimization.
    - `nvfp4/`: 1-node native FP4 serving with NEXTN speculative decoding.
  - `KimiK2.5/`: Configurations for Kimi-K2.5.
  - `Qwen3.5/`: TTFT-focused tuning for Qwen3.5-397B-A17B.
    - `fp8/`: 2× SMG TP=8 with fused QK-norm-RoPE and fused MoE-sum-allreduce; NCCL_NCHANNELS=16. Mean TTFT 2.26 s.
  - `DSV4/`: DeepSeek-V4 single-node, `flash`/`pro` × `fp8`/`nvfp4`.
    - `flash/fp8/1k8k/`: **shipped** — TP=8 + DP-attention, 3412.9 tok/s @ B=1016 (7.1× over pure TP).
    - `flash/fp8/8k64k/`: **shipped** — TP=8 + DP-attention, 551.9 tok/s @ B=33 (2.45× over pure TP).
    - `flash/nvfp4/`: FP4 runs but loses on SM120 (mxfp4 MoE compute-bound) — notes only.
    - `pro/`: not yet benchmarked on this cluster.
- `gkecluster/`: Infrastructure-as-Code for GKE provisioning.
  - `createCluster_template.sh`: Automated script to provision VPC, networking, and GKE clusters optimized for Blackwell G4.
  - `createCluster_README.md`: Detailed setup and usage instructions for the GKE template.
- `benchmarking_scripts/`: Global benchmark definitions and performance scripts.
  - `benchmark-dsv2.yaml`: Load generator config for DeepSeek.
  - `benchmark-glm51.yaml`: Load generator config for GLM-5.1.
- `gcp_g4_specs.md`: Detailed hardware and infrastructure specifications.

## Key Updates (April 2026)
- **Native FP4 Support**: Successfully validated DeepSeek-V3.2 and GLM-5.1 on single-node setups using NVFP4 quantization, achieving high efficiency on Blackwell architecture.
- **Speculative Decoding**: Integrated EAGLE for DeepSeek-V3.2 and NEXTN for GLM-5.1 NVFP4 to optimize token generation speeds.
- **GLM-5.1 Optimization**: Completed both FP8 (2-node) and NVFP4 (1-node) serving optimizations.
- **Distributed SGLang**: Standardized 2-node configurations for ultra-large models using `pipeline-parallel-size 2` and `tensor-parallel-size 8`.

## GKE Infrastructure Setup

The `gkecluster` directory contains a comprehensive template for provisioning a GKE environment optimized for SGLang:
- **Custom VPC**: High MTU (8896) for optimized multi-node traffic.
- **Multi-Networking**: Specialized network interfaces for distributed inference.
- **Blackwell Node Pools**: Automated creation of `g4-standard-384` pools with 8x RTX PRO 6000 Blackwell GPUs.
- **Benchmarking Isolation**: Dedicated node pools for load generators to ensure clean performance metrics.

## Viewing Detailed Benchmark Results

Detailed performance logs, including TTFT/TPOT latency distributions and throughput metrics, are located within each model's `results` directory:

- [DeepSeek-V3.2 (FP8): models/DeepSeekv3-2/fp8/results/benchmark_results.md](./models/DeepSeekv3-2/fp8/results/benchmark_results.md)
- [DeepSeek-V3.2 (NVFP4): models/DeepSeekv3-2/nvp4/results/benchmark_results.md](./models/DeepSeekv3-2/nvp4/results/benchmark_results.md)
- [GLM-5.1 (FP8): models/GLM5.1/results/benchmark-results.md](./models/GLM5.1/results/benchmark-results.md)
- [GLM-5.1 (NVFP4): models/GLM5.1/nvfp4/results/benchmark_results.md](./models/GLM5.1/nvfp4/results/benchmark_results.md)
- [GLM-5.2 (NVFP4, EAGLE spec + Pareto curves): models/GLM5.2/results/concurrency-sweep.md](./models/GLM5.2/results/concurrency-sweep.md)
- [GLM-5.2 (FP8, 2-node PP=2 — full-FP8 checkpoint): models/GLM5.2/fp8/README.md](./models/GLM5.2/fp8/README.md)
- [Kimi-K2.5 (FP8): models/KimiK2.5/results/benchmark_results.md](./models/KimiK2.5/results/benchmark_results.md)
- [Qwen3.5-397B-A17B (FP8, TTFT-focused): models/Qwen3.5/fp8/results/benchmark_results.md](./models/Qwen3.5/fp8/results/benchmark_results.md)
- [DeepSeek-V4-Flash (FP8, 1K/8K): models/DSV4/flash/fp8/1k8k/results/benchmark_results.md](./models/DSV4/flash/fp8/1k8k/results/benchmark_results.md)
- [DeepSeek-V4-Flash (FP8, 8K/64K): models/DSV4/flash/fp8/8k64k/results/benchmark_results.md](./models/DSV4/flash/fp8/8k64k/results/benchmark_results.md)

## Usage

For detailed instructions on deploying models and running benchmarks, see the [Benchmarking Guide](./benchmarking_guide.md).

Each model directory also contains a dedicated `README.md` with specific optimization details and attribution.

## Contributing

This repository is updated as new optimization techniques (e.g., native FP4 serving) and models are validated on the G4 architecture.
