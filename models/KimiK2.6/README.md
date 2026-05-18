# Kimi-K2.6 on GCP G4

Optimized configurations and benchmarks for [Moonshot AI's Kimi-K2.6](https://huggingface.co/moonshotai/Kimi-K2.6) on GCP G4 instances using SGLang.

## Model Overview
Kimi-K2.6 is a ~1T-parameter Mixture-of-Experts (MoE) model with native **INT4** quantization. Architecture shares K2.5's MLA + DeepSeek-style routed-experts.

## Serving Configuration — 2-Node, 2× SMG-Routed Replicas + EAGLE3 (4, 6) + HiCache

- **Model**: `moonshotai/Kimi-K2.6`
- **Topology**: 2× `g4-standard-384` (16x RTX PRO 6000 Blackwell) as **two independent** TP=8 SGLang replicas (NOT a distributed PP=2 instance), fronted by an `sgl-mini-gateway` (SMG) router with `--policy power_of_two`
- **Quantization**: Native INT4
- **KV Cache**: FP8 (e4m3)
- **Speculative Algorithm**: EAGLE3 with `lightseekorg/kimi-k2.5-eagle3` draft (`--speculative-num-steps 4 --speculative-num-draft-tokens 6 --speculative-eagle-topk 1`) — sweep-tuned for high-concurrency multi-turn workloads
- **HiCache**: `--hicache-ratio 2.0`, `--hicache-write-policy write_through_selective`
- **Schedule policy**: `lpm`, conservativeness 0.3, chunked-prefill 16384, `--enable-mixed-chunk`
- **Memory Fraction**: 0.85
- **Serving Image**: `lmsysorg/sglang:dev-cu13` (+ `pip install sglang==0.5.10.post1` inside)
- **Required env**: `SGLANG_ENABLE_SPEC_V2=1`, `FLASHINFER_DISABLE_VERSION_CHECK=1`, plus the full NCCL/GLOO socket-pinning block (see manifest)

Two independent replicas avoid the inter-node TCP all-reduce that PP=2 imposes on every forward step. The (4, 6) EAGLE3 sweep point beat (3, 4), (5, 6), and the +FI-AR-fusion variant on this cluster.

## Benchmark Results

Workload: **ClientAgenticBenchmark** — 12,542 multi-turn agentic requests across 256 sequences, mean ISL ~18 K tokens, mean OSL ~458 tokens, p99 ISL 73 K tokens. Long-context, prefill-dominant agentic workload (NOT the standard SGLang `bench_serving` random workload).

| Metric | Result |
|--------|--------|
| Wall clock | **7h 14m 51s** (26,091.5 s) |
| Total throughput | **8,924.5 tok/s** |
| Prompt token throughput | 8,704.4 tok/s |
| Completion token throughput | 220.1 tok/s |
| Request throughput | 0.481 req/s |
| Requests completed | **12,542 / 12,542 (100 %)** |
| Failures | **0** |
| Mean E2E latency | 133.9 s |
| p50 E2E latency | 34.0 s |
| p90 E2E latency | 407.4 s |
| p99 E2E latency | 952.8 s |
| Total prompt tokens served | 227,110,098 |
| Total completion tokens generated | 5,742,931 |

### Ablation lineage on the same workload

| Run | Topology | Speculative | HiCache | Wall | tot tok/s | p99 latency | Completion | Failures |
|-----|---|---|:-:|---:|---:|---:|---:|---:|
| Run #2 | PP=2 + DPA + radix + dyn-chunking | none | no | 20h 36m | 3,069 | — | 97.4 % | 326 |
| D7 | 2x SMG `power_of_two` | none | no | 17h 37m | 3,669 | 1,238 s | 99.89 % | 14 |
| D8 | 2x SMG + HiCache + EAGLE3 (3, 4) | EAGLE3 (3, 4) | yes | 7h 22m | 8,772 | 1,363 s | 100 % | 0 |
| **D9 (ship)** | 2x SMG + HiCache + EAGLE3 (4, 6) | **EAGLE3 (4, 6)** | yes | **7h 14m 51s** | **8,924** | **953 s** | **100 %** | **0** |

D9 over D8: −1.7 % wall, +1.7 % tot tok/s, **−30 % p99 latency**.

### Detailed Logs
- [Full Results](./results/benchmark_results.md)

**Attribution**: Optimization strategy, EAGLE3 sweep, and benchmark execution by **Jimmy Shong** (RadixArk).

## Usage

The manifest below lists the env vars and the two shell invocations (per-server `sglang.launch_server` and the SMG router) verbatim from the run that produced these numbers.
```bash
cat sglang-kimi-26-2node-smg.yaml
```
