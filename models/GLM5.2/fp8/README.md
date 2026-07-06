# GLM-5.2-FP8 (zai-org) — 2-node, TP=16 — BLOCKED on latest main

`zai-org/GLM-5.2-FP8` is the full-FP8 checkpoint (**704 GB**, ~88 GB/GPU at TP=8 → won't fit one node). It
needs **2 nodes at TP=16**. On latest-main `lmsysorg/sglang:dev-cu13` it is **blocked** by a cross-node
warmup deadlock. This directory documents the blocker + the recipe that IS correct for when the substrate is
fixed.

## Status: BLOCKED — cross-node warmup deadlock

Distributed init + weight load **succeed** (~83.7 GB/GPU at TP=16, all 16 ranks load). Then the **first
warmup forward hangs permanently** at:

```
multimem all-gather disabled (Failed to send fd: No such file or directory)
```

Root cause: sglang's own **CUDA symm-mem multimem** collective does a **CUDA-IPC file-descriptor exchange
over a Unix domain socket** — which cannot work across two nodes with no shared filesystem. sglang detects
the fd send failure, **disables multimem**, then **deadlocks on the next collective** (the fallback path
still enters a collective that never completes). The 2400s watchdog **never fires** — the ranks spin rather
than block, so there is no timeout to trip.

### Configs tried (all identical — permanent warmup hang)

| # | Variation |
|---|---|
| 1 | Baseline TP=16, DP-attention on |
| 2 | DP-attention **off** |
| 3 | `--disable-cuda-graph` |
| 4 | `--disable-custom-all-reduce` |
| 5 | `NCCL_SHM_DISABLE=1` + `NCCL_P2P_DISABLE=1` + msg-queue-broadcaster off |

No-code path is **exhausted**.

## Remaining options (not attempted — both need more than a launch flag)

- **PP-across-nodes** — pipeline parallel uses point-to-point sends, avoiding the multimem collective
  entirely. But it needs image **`v0.5.12.post1-cu130`** (latest dev-cu13 has a PP=2 stage-1 forward
  regression), **plus** the GLM DSA topk-boundary patch, which historically loses (~76 tok/s/GPU). Not a
  throughput win even if it boots.
- **Source patch** gating the post-multimem-disable collective (skip the fallback collective, or route it
  through NCCL instead of symm-mem) so warmup can complete cross-node.

## The recipe that IS correct (for when the substrate is fixed)

TP=16 across 2 nodes, and:

- **`--moe-runner-backend triton`** — FP8 MoE is a **triton** path; `flashinfer_cutlass` is **FP4-only** and
  crashes `Fp8MoEMethod`. (This is the opposite of the NVFP4 checkpoint, which uses flashinfer_cutlass.)
- **`--attention-backend flashinfer`**
- **`--disable-shared-experts-fusion`** (loader crashes without it)
- **`--kv-cache-dtype fp8_e4m3`**, `--moe-a2a-backend none`, DP-attention (`--enable-dp-attention` with
  `dp_size == tp_size == 16`), + the load-bearing NCCL/GLOO block + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## The pick for this hardware is NVFP4, not FP8

The NVFP4 checkpoint fits **one node** (TP=8) and hits **335 tok/s/GPU** with fp8 KV — see
[`../nvfp4/`](../nvfp4/). The full-FP8 checkpoint offers no upside on this cluster even if the deadlock is
fixed (2× the nodes, same class of throughput). Documented here for completeness.

Full campaign data: `runs/20260705_glm5.2_sota_humanize/` in the gcp-kimi repo.
