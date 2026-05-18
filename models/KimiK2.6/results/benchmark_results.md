# Kimi-K2.6 Benchmark Results

Two variants benched on different workloads. Numbers are not directly comparable across variants — read each section's workload section before drawing comparisons.

---

## Variant A — 1-Node TP=8, EAGLE3 (3, 4) + HiCache (SGLang random workload)

Workload: standard SGLang `bench_serving` random, 1,536 requests, max concurrency 512, single-turn.

```
============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    inf       
Max request concurrency:                 512       
Successful requests:                     1536      
Benchmark duration (s):                  4409.68   
Total input tokens:                      784969    
Total input text tokens:                 784969    
Total generated tokens:                  6434886   
Total generated tokens (retokenized):    6426537   
Request throughput (req/s):              0.35      
Input token throughput (tok/s):          178.01    
Output token throughput (tok/s):         1459.26   
Peak output token throughput (tok/s):    850.00    
Peak concurrent requests:                516       
Total token throughput (tok/s):          1637.28   
Concurrency:                             429.06    
Accept length:                           3.65      
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   1231782.18
Median E2E Latency (ms):                 1274658.11
P90 E2E Latency (ms):                    1559094.16
P99 E2E Latency (ms):                    2504072.49
---------------Time to First Token----------------
Mean TTFT (ms):                          937356.38 
Median TTFT (ms):                        1087881.24
P99 TTFT (ms):                           1284144.10
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          82.43     
Median TPOT (ms):                        36.55     
P99 TPOT (ms):                           819.13    
---------------Inter-Token Latency----------------
Mean ITL (ms):                           70.15     
Median ITL (ms):                         32.76     
P95 ITL (ms):                            65.94     
P99 ITL (ms):                            131.69    
Max ITL (ms):                            1278781.46
==================================================
```

---

## Variant B — 2-Node 2× SMG Replicas, EAGLE3 (4, 6) + HiCache (ClientAgenticBenchmark)

| Field | Value |
|---|---|
| Date | 2026-05-17 (start) → 2026-05-18 (finish) |
| Cluster | 2x `g4-standard-384` (16x RTX PRO 6000 Blackwell, GCP `us-west4-a`) |
| Topology | 2x independent single-node TP=8 SGLang replicas, SMG `power_of_two` router |
| Image | `lmsysorg/sglang:dev-cu13` + `pip install sglang==0.5.10.post1` |
| Workload | **ClientAgenticBenchmark** — 256 multi-turn agentic sequences, 12,542 total requests |
| Prompt tokens | mean 18,108 / p50 13,505 / p90 38,897 / p99 73,234 |
| Completion tokens | mean 458 / p50 133 / p90 1,084 / p99 4,096 (cap) |
| Bench start / finish | 2026-05-17 **18:08:16 UTC** → 2026-05-18 **01:23:08 UTC** |
| **Wall clock** | **26,091.45 s = 7h 14m 51s** |

### Headline numbers (canonical `results-20260518-012308.json`)

| Section | Field | Value |
|---|---|---:|
| **Request counts** | `num_requests_total` | **12,542** |
| | `num_requests_completed` | **12,542 (100.00 %)** |
| | `num_requests_failed` | **0** |
| | `num_sequences` | 256 |
| **Throughput** | `requests_per_second` | **0.4807** |
| | `prompt_tokens_per_second` | **8,704.4** |
| | `completion_tokens_per_second` | **220.1** |
| | **`total_tokens_per_second`** | **8,924.5** |
| **Latency (E2E sec/request)** | mean | 133.9 |
| | p50 | 34.0 |
| | p90 | 407.4 |
| | p95 | 509.8 |
| | p99 | **952.8** |
| **Totals** | prompt tokens | 227,110,098 |
| | completion tokens | 5,742,931 |
| | total tokens | 232,853,029 |

### Ablation lineage on the same ClientAgenticBenchmark workload

| Run | Topology | Speculative | HiCache | Wall | tot tok/s | p99 latency | Completion | Failures |
|---|---|---|:-:|---:|---:|---:|---:|---:|
| Run #2 | 2-node PP=2 + DPA + radix + dyn-chunking | none | no | 20h 36m | 3,069 | — | 12,216 / 12,542 (97.4 %) | 326 (ReadTimeouts) |
| D7 | 2× SMG `power_of_two` | none | no | 17h 37m | 3,669 | 1,238 s | 12,528 / 12,542 (99.89 %) | 14 |
| D8 | 2× SMG + HiCache + EAGLE3 (3, 4) | EAGLE3 (3, 4) | yes | 7h 22m | 8,772 | 1,363 s | 12,542 / 12,542 (100 %) | 0 |
| **Variant B (D9, ship)** | 2× SMG + HiCache + EAGLE3 (4, 6) | **EAGLE3 (4, 6)** | yes | **7h 14m 51s** | **8,924** | **953 s** | **12,542 / 12,542 (100 %)** | **0** |

**Variant B vs D8 (prior anchor)**: −1.7 % wall, +1.7 % tot tok/s, **−30 % p99 latency**. The larger EAGLE3 draft budget materially helps even the deep-tail single-sequence decode phase (which we initially hypothesized would regress — empirically did not).

### EAGLE3 (num_steps × num_draft_tokens) sweep that led to (4, 6)

| Sweep point | num_steps × num_draft_tokens | accept_len (steady state) | accept_rate | gen tok/s/rank | Verdict |
|---|---:|---:|---:|---:|---|
| D8 | 3 × 4 | 2.33 | 0.58 | ~275 | prior ship |
| **Variant B (D9, ship)** | **4 × 6** | **2.83** | **0.57** | **332** | **winner** |
| D10 | 5 × 6 | 2.84 | 0.47 | 295 | lost: lower accept_rate at higher steps |
| D11 | 4 × 6 + `--enable-flashinfer-allreduce-fusion` | 2.78 | 0.56 | 295 | regressed −14 % cluster decode |

Sweep method: ~15-min warmup, 2-min steady-state pull. Workload fixed (`benchmark.py` unmodified).

### Per-rank kernel breakdown at Variant B steady state (torch.profiler, node-1 TP-0)

Total kernel wall: 1,026 ms across 20,814 captured events over 10 captured decode steps.

| Category | Share | Kernel |
|---|---:|---|
| MoE INT4 matmul | 31.3 % | `marlin_moe` (×2 variants) — structural |
| NCCL all-reduce | 18.3 % | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` + AllGather |
| Memory copy (bf16↔fp8 / layout) | 15.8 % | `unrolled_elementwise direct_copy` |
| MLA attention | 12.2 % | `flashinfer::mla::BatchMLAPagedAttentionKernel` |
| HiCache transfer | 7.5 % | `transfer_kernel_impl` (page-first) |
| Other (cutlass + glue) | ~16 % | small GEMMs, RMSNorm, MoE-sum, activations |

GPU util mean 89-92 %, p90 100 %. Cluster is prefill-dominated (4.6 : 1 prefill:decode batch ratio) and GPU-saturated. AR is the largest remediable kernel share but D11 confirmed FI AR fusion regresses on this stack.

### Source bundle (RadixArk gcp-kimi repo, k2.6 branch)

```
_raw/k2.6/d9-full-bench-2026-05-17/
├── REPORT.md                        # detailed report with full comparison
├── results-20260518-012308.json     # canonical bench summary (source of all numbers above)
├── router.log                       # 11 MB SMG router log (full 12,542 request timings)
├── bench_output.log                 # bench client stdout
├── server_info_n1.json              # /get_server_info post-bench
└── replica_launch.sh                # per-replica launch script
```

Sweep bundle: `_raw/k2.6/d9-eagle3-sweep-2026-05-17/`
D11 regression bundle: `_raw/k2.6/d11-fi-ar-fusion-2026-05-17/`
