# Kimi-K2.6 on GCP G4

Optimized configurations and benchmarks for [Moonshot AI's Kimi-K2.6](https://huggingface.co/moonshotai/Kimi-K2.6) on GCP G4 instances using SGLang.

## Model Overview
Kimi-K2.6 is a ~1T-parameter Mixture-of-Experts (MoE) model with native **INT4** quantization, optimized for advanced reasoning and tool-use capabilities. Architecture shares K2.5's MLA + DeepSeek-style routed-experts.

## Serving Configurations

Three variants ship in this directory. Pick by topology, target workload, and SLO priority.

### Variant A — 1-Node, EAGLE3 (3, 4) + HiCache (single-replica, standard random workload)
- **Model**: `moonshotai/Kimi-K2.6`
- **Topology**: 1× `g4-standard-384` (8x RTX PRO 6000 Blackwell), TP=8, single SGLang replica
- **Quantization**: Native INT4
- **KV Cache**: FP8 (e4m3)
- **Speculative Algorithm**: EAGLE3 with `lightseekorg/kimi-k2.5-eagle3` draft (num_steps=3, num_draft_tokens=4)
- **Key Features**: Hierarchical cache enabled, mixed-chunk scheduling, specialized Kimi-K2 parsers
- **Serving Image**: `lmsysorg/sglang:dev-cu13`
- **Manifest**: `sglang-kimi-26-1node.yaml`

### Variant B — 2-Node, 2× SMG-Routed Replicas + EAGLE3 (4, 6) + HiCache (multi-replica, agentic workload)
- **Model**: `moonshotai/Kimi-K2.6`
- **Topology**: 2× `g4-standard-384` (16x RTX PRO 6000 Blackwell) as **two independent** TP=8 SGLang replicas (NOT a distributed PP=2 instance), fronted by an `sgl-mini-gateway` (SMG) router with `--policy power_of_two`
- **Quantization**: Native INT4
- **KV Cache**: FP8 (e4m3)
- **Speculative Algorithm**: EAGLE3 with `lightseekorg/kimi-k2.5-eagle3` draft (num_steps=**4**, num_draft_tokens=**6**) — sweep-tuned for high-concurrency multi-turn workloads
- **HiCache**: `--hicache-ratio 2.0`, `--hicache-write-policy write_through_selective`
- **Schedule policy**: `lpm`, conservativeness 0.3, chunked-prefill 16384, `--enable-mixed-chunk`
- **Memory Fraction**: 0.85
- **Serving Image**: `lmsysorg/sglang:dev-cu13` (+ `pip install sglang==0.5.10.post1` inside)
- **Required env**: `SGLANG_ENABLE_SPEC_V2=1`, `FLASHINFER_DISABLE_VERSION_CHECK=1`, plus the full NCCL/GLOO socket-pinning block (see manifest)
- **Manifest**: `sglang-kimi-26-2node-smg.yaml`

> Variant B chosen as the high-throughput ship for the **ClientAgenticBenchmark** (long-context multi-turn agentic) workload. Two independent replicas avoid the inter-node TCP all-reduce that PP=2 (Variant C) imposes on every forward step.

### Variant C — 2-Node, Distributed PP=2 (single replica spread across both nodes)
- **Model**: `moonshotai/Kimi-K2.6`
- **Topology**: 2× `g4-standard-384`, TP=8 × PP=2, single distributed SGLang instance
- **Quantization**: Native INT4
- **KV Cache**: FP8 (e5m2)
- **Serving Image**: `lmsysorg/sglang:v0.5.10.post1`
- **Manifest**: `sglang-kimi-26-2nd.yaml`

## Benchmark Results

### Variant A — 1-Node, SGLang random workload (1,536 reqs, conc 512)

| Metric | Result |
|--------|--------|
| Output Throughput | 1,459.26 tok/s |
| Total Throughput | 1,637.28 tok/s |
| Peak Output Throughput | 850.00 tok/s |
| Mean TPOT | 82.43 ms |
| Median TTFT | 1,087.88 s |
| Accept length (EAGLE3) | 3.65 |

> **Note**: High TTFT (1,087.88 s) reflects the high-concurrency (max 512) sustained-load benchmark.

### Variant B — 2-Node 2× SMG, ClientAgenticBenchmark (12,542 reqs, 256 sequences, mean ISL ~18 K)

| Metric | Result |
|--------|--------|
| Wall clock | **7h 14m 51s** (26,091.5 s) |
| Total Throughput | **8,924.5 tok/s** |
| Prompt token throughput | 8,704.4 tok/s |
| Completion token throughput | 220.1 tok/s |
| Requests completed | **12,542 / 12,542 (100 %)** |
| Failures | **0** |
| Mean E2E latency | 133.9 s |
| p50 E2E latency | 34.0 s |
| p99 E2E latency | 952.8 s |

> Workload is **ClientAgenticBenchmark** (multi-turn agentic with tools, p99 ISL 73 K tokens) — **not directly comparable** to Variant A's random `bench_serving` workload. Variant B beats the prior 2-node config (D8 equivalent on the same workload) by 1.7 % on wall, 1.7 % on total tok/s, and 30 % on p99 latency.

### Detailed Logs
- [Full Results (Variant A + Variant B + ablations)](./results/benchmark_results.md)

**Attribution**:
- Variant A (1-node, EAGLE3 (3, 4)): **Shivaji Dutta**.
- Variant B (2-node 2× SMG, EAGLE3 (4, 6) sweep + ClientAgenticBenchmark validation): **Jimmy Shong** (RadixArk).

## Usage

### Variant A — 1-Node
```bash
kubectl apply -f sglang-kimi-26-1node.yaml
```

### Variant B — 2-Node, 2× SMG Replicas (high-throughput ship)
```bash
kubectl apply -f sglang-kimi-26-2node-smg.yaml
```
The router exposes a single OpenAI-compatible endpoint at port 8000; both server replicas are reachable individually via the headless `sglang-replica` service.

### Variant C — 2-Node Distributed PP=2
```bash
kubectl apply -f sglang-kimi-26-2nd.yaml
```
