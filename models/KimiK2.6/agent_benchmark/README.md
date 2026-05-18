# Kimi K2.6 Agentic Benchmark for SGLang

This repository contains the configuration, scripts, and results for benchmarking the **Kimi K2.6** model using **SGLang**. The benchmark focuses on evaluating performance under agentic workloads, specifically testing features like **Hierarchical Cache (HiCache)** and **EAGLE3 Speculative Decoding**.

## Overview

The benchmark simulates real-world agentic traces from the Kimi K2.6 model. It measures key performance metrics including request throughput, token throughput, and latency percentiles (P50, P90, P95, P99).

### Key Features Tested:
- **SGLang Hierarchical Cache (HiCache):** Optimized KV cache management for long-context and multi-turn agentic interactions.
- **EAGLE3 Speculative Decoding:** Acceleration technique using a draft model (Kimi-K2.5-EAGLE3) to speed up inference.
- **SMG Router:** Load balancing and request routing across multiple SGLang replicas in a multi-node environment.

---

## Deployment Configurations

There are two primary deployment configurations used in this benchmark:

### 1. Single-Node GKE (Google Kubernetes Engine)
- **Environment:** GKE Cluster with NVIDIA L4/A100 GPUs.
- **Setup:** A single SGLang server running with TP=8.
- **Storage:** GCS Fuse for model weights and local SSD/EmptyDir for HiCache storage.
- **Configuration:** Defined in `sglang-debug-hicache.yaml`.
- **Highlights:** Demonstrates high cache hit rates (up to 81%) using HiCache in a single-node setup.

### 2. Dual-Node GCE (Google Compute Engine)
- **Environment:** Two independent GCE instances, each running an SGLang server (TP=8).
- **Setup:** A single **SMG Router** (SGLang Multi-GPU) is used to distribute traffic across both nodes.
- **Acceleration:** EAGLE3 Speculative Decoding enabled on both nodes.
- **Configuration:** Orchestrated via `sglang-2node-hicache-smg.sh`.
- **Highlights:** Achieves higher overall throughput by scaling across multiple nodes, though cache hit reporting via the router may vary.

---

## Benchmark Results

Detailed results are stored in the `results/` directory.

| Metric | Single-Node (GKE) | Dual-Node (GCE) |
| :--- | :---: | :---: |
| **Requests per Second** | 0.353 | 0.481 |
| **Total Tokens per Second** | 6,550.98 | 8,924.50 |
| **Mean Latency (s)** | 100.19 | 133.92 |
| **P50 Latency (s)** | 16.13 | 33.96 |
| **P99 Latency (s)** | 699.50 | 952.79 |
| **Prompt Cache Hit Rate** | **81.19%** | 0.00%* |

*\*Note: The 0% hit rate in the dual-node results is likely due to current limitations in how the SMG router aggregates and reports cache statistics from backend workers.*

---

## Project Structure

```text
.
├── benchmark_scripts/     # Benchmark Python scripts and GKE Job manifests
│   ├── agentic_benchmark.py
│   └── agentic_benchmark_sglang_low_load.py
├── k8s-monitor/           # Real-time monitoring dashboard for GKE workloads
├── results/               # Benchmark output JSON files
├── sglang-2node-hicache-smg.sh  # Launch script for dual-node setup
└── sglang-debug-hicache.yaml    # Kubernetes manifest for GKE setup
```

---

## Architecture

The dual-node setup utilizes a hierarchical routing architecture to maximize throughput while maintaining speculative decoding efficiency.

![Architecture Diagram](results/dual_node_architecture.svg)
*(Note: Please ensure the `dual_node_architecture.svg` file is placed in the `results/` directory to display the diagram above.)*

---

## How to Run

### Running the Benchmark Script
1. Ensure the SGLang server is running.
2. Decompress the data file: `zstd -d benchmark_scripts/data.jsonl.zst`.
3. Run the benchmark:
   ```bash
   python3 benchmark_scripts/agentic_benchmark_sglang_low_load.py http://<SGLANG_IP>:30000 --parallelism 64
   ```

### Deploying the Monitor
The `k8s-monitor` provides a UI for tracking progress:
```bash
kubectl apply -f k8s-monitor/k8s-manifests.yaml
kubectl port-forward svc/k8s-monitor 8080:80
```
Visit `http://localhost:8080` to view real-time metrics.
