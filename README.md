# OSS Model Benchmarks on GCP G4

📖 **[Link to Documentation](https://shivajid.github.io/sglang-rtx-pro-6000/)** 


Optimized GKE configurations and benchmarks for serving LLMs on GCP G4 instances.

## Infrastructure
- **GPU**: NVIDIA RTX PRO 6000 Blackwell (SM120)
- **Architecture Details**: [Technical Specifications: GCP G4](./gcp_g4_specs.md)
- **Serving Framework**: [SGLang](https://github.com/sgl-project/sglang) (`dev-cu13`, `0.5.10.post1`)

## Performance Benchmarks (Latest)

| Model | Quantization | Setup | Output Throughput (tok/s) | Total Throughput (tok/s) | Peak Throughput (tok/s) | TPOT (ms) |
|-------|--------------|-------|---------------------------|--------------------------|-------------------------|-----------|
| [deepseek-ai/DeepSeek-V3.2](./models/DeepSeekv3-2/fp8/results/benchmark_results.md) | FP8 | 2 Nodes (16x RTX 6000) | 2962.79 | 3324.21 | 4951.00 | 149.29 |
| [nvidia/DeepSeek-V3.2-NVFP4](./models/DeepSeekv3-2/nvp4/results/benchmark_results.md) | NVFP4 | 1 Node (8x RTX 6000) | 2675.33 | 3012.42 | 2046.00 | 106.03 |
| [zai-org/GLM-5.1-FP8](./models/GLM5.1/results/benchmark-results.md) | FP8 | 2 Nodes (16x RTX 6000) | 2785.55 | 3125.35 | 4092.00 | 155.26 |
| [lukealonso/GLM-5.1-NVFP4](./models/GLM5.1/nvfp4/results/benchmark_results_1node.md) | NVFP4 | 1 Node (8x RTX 6000) | 1490.31 | 1672.11 | 734.00 | 73.82 |
| [lukealonso/GLM-5.1-NVFP4](./models/GLM5.1/nvfp4/results/benchmark_results_2node.md) | NVFP4 | 2 Nodes (16x RTX 6000) | 3075.85 | 3451.06 | 4606.00 | 141.36 |
| [moonshotai/Kimi-K2.5](./models/KimiK2.5/results/benchmark_results.md) | INT4* | 2 Nodes (16x RTX 6000) | 3152.79 | 3537.39 | 4793.00 | 136.52 |
| [nvidia/Kimi-K2.5-NVFP4](./models/KimiK2.5/nvfp4/results/benchmarks_2node.yaml) | NVFP4 | 2 Nodes (16x RTX 6000) | 3237.46 | 3632.39 | 5535.00 | 137.89 |
| [moonshotai/Kimi-K2.6](./models/KimiK2.6/results/benchmark_results.md) | INT4* | 1 Node (8x RTX 6000) | 1459.26 | 1637.28 | 850.00 | 82.43 |
| [datalab-to/chandra-ocr-2](./models/datalab2-ocr/benchmark_results.md)** | BF16| 1 Node (1x RTX 6000)| 2600.67 | 5267.08 | 4603.00| 32.47 |

**[openai/whisper-large-v3](./models/whisper-v3-large/results/benchmark_results.md)** - Since this is ASR model, we did not apply the standard ISL/OSL of 1K/8K and concurrancy of 512.

*Table last updated: May 22, 2026*
 
*Benchmarks conducted using `inf` request rate and 512 max concurrency. Tests utilized a random dataset with 1024 input tokens and 8192 output tokens (1536 total prompts). The load generator was isolated on a dedicated CPU-only node pool to ensure zero interference with GPU performance.*

*\*Kimi-K2.5 and Kimi-K2.6 use native INT4 quantization and KV cache optimization to improve memory efficiency and inference speed.*

**\** datalab-to/chandra-ocr-2 is an VLM model. We have run an image benchmark different for the rest of the models **

## [moonshotai/Kimi-K2.6](./models/KimiK2.6/agent_benchmark/README.md) Agentic Benchmark
[Detailed Configuration & Results](./models/KimiK2.6/agent_benchmark/)

Evaluates performance under real-world agentic traces with long sequences and high prompt volume. This benchmark uses non-standard traffic profiles and SGLang features like **HiCache** and **EAGLE3 Speculative Decoding** to evaluate performance on complex workloads.

### Benchmark Settings
- **Traffic Profile:** Simulated real-world agentic traces (Replay).
  - **Request Configuration:** Temperature=0.6, Top_P=0.95, Max Tokens=4096.
  - **Dataset:** Kimi K2.6 real-world agentic trace replay (Long-tail steps and sequences).
- **Parallelism:** 64 (Single-Node), 256 (Two-Node).
- **SGLang Features:** HiCache, EAGLE3 Speculative Decoding, SMG Router (Dual-node).
- **Environment:** TP=8 per node, GKE (Single-node), GCE (Dual-node).

| Metric | Single-Node (GKE) | Two-Node (GCE) |
| :--- | :---: | :---: |
| **Requests per Second** | 0.353 | 0.481 |
| **Total Tokens per Second** | 6,550.98 | 8,924.50 |
| **P50 Latency (s)** | 16.13 | 33.96 |
| **P99 Latency (s)** | 699.50 | 952.79 |
| **Prompt Cache Hit Rate** | **81.19%** | 0.00%* |

*\*Note: 0% hit rate on dual-node is a reporting limitation of the SMG router.*

## [Qwen/Qwen3.5-397B-A17B-FP8](./models/Qwen3.5-397B-A17B-FP8/BENCHMARK_REPORT.md) Latency Benchmark
[Detailed Configuration & Results](./models/Qwen3.5-397B-A17B-FP8/)

Focuses on latency characteristics of an ultra-large MoE model, comparing performance with and without **SGLang HiCache**. This benchmark evaluates the model's performance under ultra-large workload scenarios.

### Benchmark Settings
- **Traffic Profile:**
  - **Input Length:** 20,000 tokens
  - **Output Length:** 1,000 tokens
  - **Concurrency:** 40
  - **Number of Prompts:** 2,000
- **Total Tokens:** ~19.7M Input, ~986K Generated.
- **Server Configuration:** TP=8, Chunked Prefill (4096), Max Prefill (32768), Mixed Chunk Enabled.
- **HiCache Config:** `--enable-hierarchical-cache --hicache-ratio=2.0 --hicache-io-backend=kernel`.

| Metric | HiCache (Enabled) | No Radix Cache |
| :--- | :---: | :---: |
| **Median TTFT (ms)** | **1,054.01** | 1,128.88 |
| **Mean TTFT (ms)** | **1,121.17** | 1,371.28 |
| **Median TPOT (ms)** | 101.18 | **90.41** |
| **Mean TPOT (ms)** | 100.59 | **90.45** |

## Project Structure

- `models/`: Model-specific SGLang job configurations and benchmarks.
  - `DeepSeekv3-2/`: Configs for DeepSeek-V3 and V2.5.
  - `GLM5.1/`: Optimized configurations and results for GLM-5.1.
  - `KimiK2.5/`: Configurations for Kimi-K2.5.
  - `KimiK2.6/`: Agentic benchmark results and HiCache configurations.
  - `Qwen3.5-397B-A17B-FP8/`: Latency benchmarks for ultra-large MoE model.
- `gkecluster/`: Infrastructure-as-Code for GKE provisioning.
- `benchmarking_scripts/`: Global benchmark definitions and performance scripts.
  - `agentic_benchmark/`: Scripts for simulating agentic workloads.
- `gcp_g4_specs.md`: Detailed hardware and infrastructure specifications.

## Key Updates (May 2026)
- **Qwen3.5-397B Validation**: Successfully benchmarked the 397B MoE model on a single node using FP8 and HiCache, showing massive TTFT improvements.
- **Agentic Benchmarking**: Introduced agentic trace simulation for Kimi K2.6, achieving over 80% cache hit rate with HiCache.
- **Kimi-K2.5 NVFP4 Validation**: Successfully optimized and benchmarked Kimi-K2.5 using native NVFP4 quantization on a 2-node (16x GPU) setup.
- **Native FP4 Support**: Successfully validated DeepSeek-V3.2 and GLM-5.1 on single-node setups using NVFP4 quantization.

## GKE Infrastructure Setup

The `gkecluster` directory contains a comprehensive template for provisioning a GKE environment optimized for SGLang:
- **Custom VPC**: High MTU (8896) for optimized multi-node traffic.
- **Multi-Networking**: Specialized network interfaces for distributed inference.
- **Blackwell Node Pools**: Automated creation of `g4-standard-384` pools with 8x RTX PRO 6000 Blackwell GPUs.
- **Benchmarking Isolation**: Dedicated node pools for load generators to ensure clean performance metrics.

## Viewing Detailed Benchmark Results

Detailed performance logs, including TTFT/TPOT latency distributions and throughput metrics, are located within each model's `results` directory:

- [Qwen/Qwen3.5-397B-A17B-FP8: models/Qwen3.5-397B-A17B-FP8/BENCHMARK_REPORT.md](./models/Qwen3.5-397B-A17B-FP8/BENCHMARK_REPORT.md)
- [moonshotai/Kimi-K2.6 Agentic: models/KimiK2.6/agent_benchmark/README.md](./models/KimiK2.6/agent_benchmark/README.md)
- [deepseek-ai/DeepSeek-V3.2 (FP8): models/DeepSeekv3-2/fp8/results/benchmark_results.md](./models/DeepSeekv3-2/fp8/results/benchmark_results.md)
- [nvidia/DeepSeek-V3.2-NVFP4 (NVFP4): models/DeepSeekv3-2/nvp4/results/benchmark_results.md](./models/DeepSeekv3-2/nvp4/results/benchmark_results.md)
- [zai-org/GLM-5.1-FP8 (FP8): models/GLM5.1/results/benchmark-results.md](./models/GLM5.1/results/benchmark-results.md)
- [lukealonso/GLM-5.1-NVFP4 (NVFP4): models/GLM5.1/nvfp4/README.md](./models/GLM5.1/nvfp4/README.md)
- [moonshotai/Kimi-K2.5 (INT4): models/KimiK2.5/results/benchmark_results.md](./models/KimiK2.5/results/benchmark_results.md)
- [nvidia/Kimi-K2.5-NVFP4 (NVFP4): models/KimiK2.5/nvfp4/results/benchmarks_2node.yaml](./models/KimiK2.5/nvfp4/results/benchmarks_2node.yaml)
- [moonshotai/Kimi-K2.6 (Standard): models/KimiK2.6/results/benchmark_results.md](./models/KimiK2.6/results/benchmark_results.md)
- [datalab-to/chandra-ocr-2: models/datalab2-ocr/benchmark_results.md](./models/datalab2-ocr/benchmark_results.md)
- [openai/whisper-large-v3: models/whisper-v3-large/results/benchmark_results.md](./models/whisper-v3-large/results/benchmark_results.md)

## Usage

For detailed instructions on deploying models and running benchmarks, see the [Benchmarking Guide](./benchmarking_guide.md).

Each model directory also contains a dedicated `README.md` with specific optimization details and attribution.

## Contributing

This repository is updated as new optimization techniques (e.g., native FP4 serving) and models are validated on the G4 architecture.
