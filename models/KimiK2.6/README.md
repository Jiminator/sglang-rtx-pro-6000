# Kimi-K2.6 on GCP G4

Optimized configurations and benchmarks for [Moonshot AI's Kimi-K2.6](https://huggingface.co/moonshotai/Kimi-K2.6) on GCP G4 instances using SGLang.

## Model Overview
Kimi-K2.6 is the latest iteration of the large-scale Mixture-of-Experts (MoE) model, optimized for advanced reasoning and tool-use capabilities.

## Serving Configuration

### 1-Node Setup (Optimized with Speculative Decoding)
- **Model**: `moonshotai/Kimi-K2.6`
- **Tensor Parallelism**: 8
- **KV Cache**: FP8 (e4m3)
- **Speculative Algorithm**: EAGLE3 (`lightseekorg/kimi-k2.5-eagle3`)
- **Serving Image**: `lmsysorg/sglang:dev-cu13`
- **Key Features**: Hierarchical cache enabled, mixed-chunk scheduling, and specialized Kimi-K2 parsers.

### 2-Node Setup (Distributed)
- **Model**: `moonshotai/Kimi-K2.6`
- **Tensor Parallelism**: 8
- **Pipeline Parallelism**: 2
- **KV Cache**: INT4
- **Serving Image**: `lmsysorg/sglang:v0.5.10.post1`

## Benchmark Results
The following benchmarks were conducted using the 1-node optimized configuration.

| Metric | Result |
|--------|--------|
| Output Throughput | 1459.26 tok/s |
| Total Throughput | 1637.28 tok/s |
| Peak Output Throughput | 850.00 tok/s |
| Mean TPOT | 82.43 ms |
| Median TTFT | 1087.88 s |

> **Note**: The high TTFT observed in these results (1087.88s) is reflective of a high-concurrency (Max 512) sustained load benchmark.

## Usage

### 1-Node Deployment
To deploy the single-node optimized setup:
```bash
kubectl apply -f sglang-kimi-26-1node.yaml
```

### 2-Node Deployment
To deploy the distributed 2-node setup:
```bash
kubectl apply -f sglang-kimi-26-2nd.yaml
```
