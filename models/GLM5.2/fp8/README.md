# GLM-5.2-FP8 (zai-org) — 2-node PP=2 + DP-attention — DEPLOYABLE

`zai-org/GLM-5.2-FP8` is the full-FP8 checkpoint (**704 GB**, ~88 GB/GPU at TP=8 → won't fit one node with a
usable KV pool). It needs **2 nodes**. This directory documents the **working config-only recipe** and the
**measured throughput** on this SM120 cluster, plus every lever that was tried and does not help.

> **Supersedes the prior "BLOCKED" note (July 2026).** The earlier doc concluded the FP8 checkpoint was not
> deployable with config alone. That was the **TP=16** path (a genuine cross-node symm-mem warmup deadlock).
> **PP=2 sidesteps it** — pipeline parallel crosses the node boundary with point-to-point sends, not the
> symm-mem multimem collective — and boots on **stock `lmsysorg/sglang:dev-cu13` (v0.5.15) with NO source
> patch**, using one env var to fix the DSA×PP layer-boundary. The FP8 checkpoint **is** deployable here.

## Throughput (measured, stock v0.5.15, 2026-07)

2-node **PP=2 + DP-attention** (TP=8 × PP=2 = 16 GPUs). `bench_serving`, random, `--random-range-ratio 1.0`,
`--request-rate inf`. gsm8k 0.92, chat coherent.

| Workload (ISL/OSL) | agg output tok/s | **tok/s/GPU (÷16)** | bound by |
|---|---:|---:|---|
| **1K/8K** (1024/8192) | ~2,190 (plateau) | **137** | PP cross-node decode hop |
| **8K/64K** (8192/65536) | ~398 | **24.85** | HBM bandwidth (dense MLA over full 64k) |

1K/8K scales 43 (cc128) → 77 (cc256) → 137 (cc512 plateau); the plateau is the steady-state per-DP-rank
decode rate (273 tok/s, pinned identically on all 8 ranks) × 8 ÷ 16. Past that, more concurrency only
inflates TTFT (the pool is not the binder — the per-token cross-node hop is).

## The recipe (both workloads use the same server)

Launch scripts: [`launch_rank0.sh`](launch_rank0.sh) (node with the dist-init addr) and
[`launch_rank1.sh`](launch_rank1.sh). Core:

```
python3 -m sglang.launch_server --model zai-org/GLM-5.2-FP8 --quantization fp8 \
  --tensor-parallel-size 8 --pipeline-parallel-size 2 --nnodes 2 --node-rank <0|1> \
  --dist-init-addr <rank0-ip>:5000 \
  --dp-size 8 --enable-dp-attention \
  --attention-backend flashinfer \       # DSA decode is SM100-only on this checkpoint → dense MLA
  --moe-runner-backend triton \          # FP8 MoE is triton (flashinfer_cutlass is FP4-only, crashes Fp8MoEMethod)
  --moe-a2a-backend none --ep-size 1 \   # no DeepEP (needs RDMA/NVLink, absent on g4)
  --kv-cache-dtype fp8_e4m3 \
  --disable-shared-experts-fusion \      # loader crashes without it
  --trust-remote-code --disable-radix-cache \
  --mem-fraction-static 0.80 \           # 0.85 OOMs in cuda-graph capture on the heavier PP stage
  --page-size 64 --host 0.0.0.0 --port 8000
```

**The load-bearing env var — this is the unlock:**
```
SGLANG_PP_LAYER_PARTITION=38,40
```
GLM-5.2 has 78 layers; the default 78÷2=39 PP boundary lands on a DSA **skip-topk** layer, and the PP
inter-stage payload doesn't forward the DSA `topk_indices` → both ranks crash
(`AssertionError: PP stage ending at layer 39 must forward DSA topk_indices` / rank-1
`forward_decode() got unexpected kwarg topk_indices`). `38,40` moves the boundary onto a full-topk layer
(GLM-5.2 full-topk layers are ≡ 2 mod 4). This is the maintainer's env-var workaround (sglang issue #28659 /
PR #28785) — **no source patch**. Plus the standard load-bearing NCCL/GLOO block and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (see the launch scripts). On this cluster the NICs are
`ens3`+`enp128s4` → `NCCL_SOCKET_IFNAME=enp128s4,ens3 GLOO_SOCKET_IFNAME=ens3`.

## SM120 constraints (why the config looks the way it does)

- **`--attention-backend flashinfer` (dense MLA), not DSA.** GLM-5.2's DSA sparse decode needs the
  paged-MQA-logits indexer kernel, which is **SM100/ROCm-only** on this checkpoint
  (`DSAPagedMQALogitsBackend.resolve()` → deepgemm "Unsupported architecture" / cutedsl "requires SM100" /
  aiter "requires ROCm"). flashinfer dense MLA bypasses it. **This is why 8K/64K is slow** (24.85/GPU): dense
  MLA reads the full 64k-token KV each decode step instead of the DSA-sparse top-2048.
- **`--moe-runner-backend triton`** — FP8 MoE is a triton path; `flashinfer_cutlass` is FP4/ModelOpt-only and
  crashes `Fp8MoEMethod`.
- **`--kv-cache-dtype fp8_e4m3`**, **`--moe-a2a-backend none`** (DeepEP needs RDMA/NVLink, dead on g4).

## Levers tried — none beat the config above (exhaustive, config-only)

| Lever | Result |
|---|---|
| **EAGLE speculative decoding** | **DEAD — 3 independent walls on SM120.** (1) PP=2 → `server_args.py` forbids spec when `pp_size>1`. (2) TP=16 + flashinfer → EAGLE draft crashes `TypeError: forward_decode() unexpected kwarg 'topk_indices'` (no allowed draft backend accepts DSA topk_indices). (3) TP=16 + native DSA → main decode cuda-graph crashes in the SM100-only DSA logits kernel. **Spec requires SM100 hardware (e.g. GB300); config cannot cross it.** |
| **TP=16** (pp_size=1) | Boots (symm-mem self-disables cross-node), but all-reduces every layer cross-node → comms-bound; loses to PP=2's one-hop-per-token. |
| **Decode context parallelism** `--dcp-size` | **AMD-HIP-only** on CUDA (`ValueError: only supported on the AMD HIP platform`). |
| **Attention context parallelism** `--attn-cp-size` | Boots (composes with PP; no DeepEP wall with `moe-a2a none`), but **−74.5%** at 8K/64K (6.34 vs 24.85/GPU): fuses the 8 GPUs of a stage into one shared schedule, collapsing concurrency 72→8. Only a low-concurrency latency lever. |
| **`SGLANG_OPT_USE_TOPK_V2=1`** (fused MoE topk) | Flat: 1234 vs 1235 agg tok/s (−0.1%). Config is comms-bound, not MoE-topk-bound. |
| **mem-fraction / pool tuning** | No headroom: 1K/8K is decode-rate-bound (pool 94% idle); 8K/64K mfs>0.80 OOMs. |

## The pick for this hardware

For **throughput**, the NVFP4 checkpoint on **one node** (TP=8, fp8 KV) is far better —
see [`../nvfp4/`](../nvfp4/) and the top-level README. The full-FP8 checkpoint costs 2× the nodes, is
forced onto dense attention on SM120 (no DSA sparse decode), and cannot use speculative decoding. **Use
full-FP8 only if a customer specifically requires FP8 weights**; the config above is its config-only maximum
on this SM120 cluster.

Full campaign data: `runs/20260710_glm52fp8_dsv4fp8_sota_humanize/` in the gcp-kimi repo. Tuning by
**Jimmy Shong** (RadixArk).
