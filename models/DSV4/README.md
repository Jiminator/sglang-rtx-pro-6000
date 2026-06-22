# DeepSeek-V4 on GCP G4 (RTX PRO 6000, SM120)

Single-node serving configurations and benchmarks for **DeepSeek-V4** on a GCP `g4-standard-384`
(8× RTX PRO 6000 Blackwell, **SM120**, no NVLink, PCIe + gVNIC TCP) using SGLang.

DeepSeek-V4 ships in two variants and two precisions. This directory is organized as:

```
DSV4/
├── flash/        DeepSeek-V4-Flash (DSA — DeepSeek Sparse Attention)
│   ├── fp8/      sgl-project/DeepSeek-V4-Flash-FP8   ← SHIPPED (winning config)
│   └── nvfp4/    deepseek-ai/DeepSeek-V4-Flash       (runs, but slower on SM120 — see notes)
└── pro/          DeepSeek-V4-Pro
    ├── fp8/      (not yet benchmarked on this cluster)
    └── nvfp4/    (not yet benchmarked on this cluster)
```

## Shipped result — Flash / FP8

| Variant / precision | Workload (ISL/OSL) | Output throughput | /GPU | Config |
|---|---|---:|---:|---|
| **Flash / FP8** | 8192 / 65536 | **551.9 tok/s** @ batch 33 | **69.0** | TP=8 + **DP-attention**, triton MoE |

**DP-attention is the lever: 2.45× over pure TP=8** (225.5 tok/s, 28/GPU) at the true 8K/64K
workload. DeepSeek-V4-Flash decode is **DSA-attention-bound** (the sparse-attention decode kernel is
83% of decode GPU time on SM120); DP-attention runs that kernel as 8 independent per-worker streams
instead of awkwardly TP-sharding the single MLA KV head. See
[`flash/fp8/TUNING_REPORT.md`](flash/fp8/TUNING_REPORT.md).

## SM120 constraints (apply to every DSV4 config here)

RTX PRO 6000 is **SM120** and lacks the TMEM/tcgen05 units the datacenter-Blackwell (SM100) kernels
need. This kills the fast FP4 and fused-MoE paths and forces a specific backend set:

- **MoE runner must be `triton`** — auto force-selects marlin, whose `Fp8MoEMethod` doesn't set
  `self.runner` → AttributeError at cuda-graph capture.
- **`--moe-a2a-backend none`** — deepep needs deep_gemm, hard-disabled at `sm_version==120`.
- **KV cache auto-locks to `fp8_e4m3`** (DSA backend hook).
- **NVFP4 runs but is slower than FP8**: the marlin CUDA mxfp4 kernel NaNs on SM120, so MoE falls
  back to a triton GEMV (`_mxfp4_slot_gemv_kernel`) that is ~19× the cost of FP8's MoE — decode
  becomes dual-bound (DSA + MoE) rather than DSA-only. FP8 wins. See
  [`flash/nvfp4/README.md`](flash/nvfp4/README.md).

## Hardware baseline

| Field | Value |
|---|---|
| Node | 1× GCP `g4-standard-384`, 8× RTX PRO 6000 Blackwell SM120 |
| Interconnect | intra-node PCIe (no NVLink); inter-node 2× 200 Gbps gVNIC TCP (single-node here) |
| Image | `lmsysorg/sglang:v0.5.13.post1-cu130` |
| sglang | v0.5.13.post1 |
| Mandatory env | full NCCL/GLOO set + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

## Attribution

Tuning and benchmarks by **Jimmy Shong** (RadixArk), `dsv4-8k64k` campaign, 2026-06.
