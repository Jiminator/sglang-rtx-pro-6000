# Kimi-K2.6 D9 — ClientAgenticBenchmark full bench

## Bench stdout

```
=== D9 ClientAgenticBenchmark (2x SMG power_of_two) vs http://localhost:8000/v1 ===
Start: 2026-05-17T18:08:12Z
results-20260518-012308.json
BENCH_EXIT=0
```

## Canonical results JSON (`results-20260518-012308.json`)

```json
{
  "started_at": "2026-05-17T18:08:16.627327+00:00",
  "completed_at": "2026-05-18T01:23:08.083282+00:00",
  "data_path": "/home/jimmy/data.jsonl.zst",
  "model": "moonshotai/Kimi-K2.6",
  "endpoints": [
    "http://localhost:8000/v1"
  ],
  "request_body": {
    "temperature": 0.6,
    "top_p": 0.95,
    "max_tokens": 4096
  },
  "parallelism": 256,
  "request_timeout_s": 1800.0,
  "num_sequences": 256,
  "num_requests_total": 12542,
  "num_requests_completed": 12542,
  "num_requests_failed": 0,
  "wall_clock_seconds": 26091.45061419811,
  "latency_seconds": {
    "mean": 133.91837483321297,
    "p50": 33.96096114302054,
    "p90": 407.3509476860054,
    "p95": 509.84317794512026,
    "p99": 952.7870753179304
  },
  "prompt_tokens": {
    "mean": 18107.965077340137,
    "p50": 13505.0,
    "p90": 38897.0,
    "p95": 49806.0,
    "p99": 73234.0
  },
  "completion_tokens": {
    "mean": 457.8959496093127,
    "p50": 133.0,
    "p90": 1084.0,
    "p95": 2650.0,
    "p99": 4096.0
  },
  "tokens": {
    "prompt": 227110098,
    "completion": 5742931,
    "total": 232853029,
    "cached_prompt": 0,
    "prompt_cache_hit_rate": 0.0
  },
  "throughput": {
    "requests_per_second": 0.48069385583241797,
    "prompt_tokens_per_second": 8704.3875543054,
    "completion_tokens_per_second": 220.10776958774707,
    "total_tokens_per_second": 8924.495323893147
  },
  "endpoint_summaries": [
    {
      "endpoint": "http://localhost:8000/v1",
      "completed": 12542,
      "prompt_tokens": 227110098,
      "completion_tokens": 5742931,
      "cached_prompt_tokens": 0,
      "prompt_cache_hit_rate": 0.0
    }
  ],
  "sample_failures": []
}
```

## Run context

| Field | Value |
|---|---|
| Date | 2026-05-17 (start) → 2026-05-18 (finish) |
| Model | `moonshotai/Kimi-K2.6` (native INT4) |
| Image | `lmsysorg/sglang:dev-cu13` + `pip install sglang==0.5.10.post1` |
| Cluster | 2x `g4-standard-384` (16x RTX PRO 6000 Blackwell, GCP `us-west4-a`) |
| Topology | 2x independent single-node TP=8 SGLang replicas, SMG `power_of_two` router |
| Workload | ClientAgenticBenchmark — 256 multi-turn agentic sequences, 12,542 total requests, mean ISL ~18 K, p99 ISL 73 K tokens |
| Bench start / finish | 2026-05-17 **18:08:16 UTC** → 2026-05-18 **01:23:08 UTC** |
| Wall clock | **26,091.45 s = 7h 14m 51s** |

## Ablation lineage on the same ClientAgenticBenchmark workload

| Run | Topology | Speculative | HiCache | Wall | Output tok/s | Total tok/s | p99 latency | Completion | Failures |
|---|---|---|:-:|---:|---:|---:|---:|---:|---:|
| Run #2 | PP=2 + DPA + radix + dyn-chunking | none | no | 20h 36m | 62.3 | 3,069 | — | 12,216 / 12,542 (97.4 %) | 326 (ReadTimeouts) |
| D7 | 2× SMG `power_of_two` | none | no | 17h 37m | 89.3 | 3,669 | 1,238 s | 12,528 / 12,542 (99.89 %) | 14 |
| D8 | 2× SMG + HiCache + EAGLE3 (3, 4) | EAGLE3 (3, 4) | yes | 7h 22m | ~216 | 8,772 | 1,363 s | 12,542 / 12,542 (100 %) | 0 |
| **D9 (ship)** | 2× SMG + HiCache + EAGLE3 (4, 6) | **EAGLE3 (4, 6)** | yes | **7h 14m 51s** | **220.1** | **8,924** | **953 s** | **12,542 / 12,542 (100 %)** | **0** |

D9 vs D8: −1.7 % wall, +1.9 % output tok/s, +1.7 % total tok/s, **−30 % p99 latency**.
D9 vs Run #2: −65 % wall, +253 % output tok/s, +191 % total tok/s.

## EAGLE3 (num_steps × num_draft_tokens) sweep that led to (4, 6)

| Sweep point | num_steps × num_draft_tokens | accept_len (steady state) | accept_rate | gen tok/s/rank | Verdict |
|---|---:|---:|---:|---:|---|
| D8 | 3 × 4 | 2.33 | 0.58 | ~275 | prior ship |
| **D9 (ship)** | **4 × 6** | **2.83** | **0.57** | **332** | **winner** |
| D10 | 5 × 6 | 2.84 | 0.47 | 295 | lost: lower accept_rate at higher steps |
| D11 | 4 × 6 + `--enable-flashinfer-allreduce-fusion` | 2.78 | 0.56 | 295 | regressed −14 % cluster decode |

Sweep method: ~15-min warmup, 2-min steady-state pull. Workload fixed (`benchmark.py` unmodified).

## Per-rank kernel breakdown at D9 steady state (torch.profiler, node-1 TP-0)

Total kernel wall: 1,026 ms across 20,814 captured events over 10 captured decode steps.

| Category | Share | Kernel |
|---|---:|---|
| MoE INT4 matmul | 31.3 % | `marlin_moe` (×2 variants) — structural |
| NCCL all-reduce | 18.3 % | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` + AllGather |
| Memory copy (bf16↔fp8 / layout) | 15.8 % | `unrolled_elementwise direct_copy` |
| MLA attention | 12.2 % | `flashinfer::mla::BatchMLAPagedAttentionKernel` |
| HiCache transfer | 7.5 % | `transfer_kernel_impl` (page-first) |
| Other (cutlass + glue) | ~16 % | small GEMMs, RMSNorm, MoE-sum, activations |

GPU util mean 89-92 %, p90 100 %. Cluster is prefill-dominated (4.6 : 1 prefill:decode batch ratio) and GPU-saturated.

## Source bundle (RadixArk gcp-kimi repo, k2.6 branch)

```
_raw/k2.6/d9-full-bench-2026-05-17/
├── REPORT.md                        # detailed report with full comparison
├── results-20260518-012308.json     # canonical bench output (reproduced above)
├── router.log                       # 11 MB SMG router log (full 12,542 request timings)
├── bench_output.log                 # bench client stdout (reproduced above)
├── server_info_n1.json              # /get_server_info post-bench
└── replica_launch.sh                # per-replica launch script
```

Sweep bundle: `_raw/k2.6/d9-eagle3-sweep-2026-05-17/`
D11 regression bundle: `_raw/k2.6/d11-fi-ar-fusion-2026-05-17/`
