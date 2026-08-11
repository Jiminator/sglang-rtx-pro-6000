# GLM-5.2-NVFP4 — Tuning Report (1K/8K, single node, SM120, latest main)

**Re-verified 2026-08-11** on a freshly pulled `lmsysorg/sglang:dev-cu13` (sglang `0.0.0.dev1+gd59c1ddf7`,
transformers 5.12.1, torch 2.13.0+cu130, flashinfer 0.6.15.post1). Workload **ISL 1024 / OSL 8192**,
no-zipfian (`--random-range-ratio 1.0`), online **`sglang.bench_serving`** as load generator → steady-state
decode plateau read from server logs (at OSL 8192 requests don't complete in-window, so #running-req pins at
the admission ceiling; aggregate = per-rank gen-throughput × dp_size 8). Checkpoint `nvidia/GLM-5.2-NVFP4`
(`GlmMoeDsa`), snapshot **`aec724e8`** (the old `b0b2b68` pin is stale — HF `refs/main` has moved).
Correctness gate: gsm8k on the idle server before load.

## Headline

**fp8 KV + DP-attention = ~3379 output tok/s (~422/GPU) @ 64 running-req/rank, gsm8k 0.900.**
Quote **~420/GPU sustained, ~400 floor** — the plateau climbs to ~444 then decays to ~403 as sequences
lengthen and the pool fills. That is **+26% over the 2026-07-05 anchor of 335/GPU** on the identical recipe,
workload and harness; the gain is the newer build itself, not a config change.

⚠️ **The 07-05 version of [`launch_glm52_nvfp4_1k8k.sh`](launch_glm52_nvfp4_1k8k.sh) does NOT run on latest
main.** Three independent blockers, all config-only. Two kill the server; **the third serves corrupt output
at full speed without erroring.**

## The three blockers (fixed in the launch script)

| # | Bites at | Symptom | Fix |
|---|---|---|---|
| 1 | argument parse | `error: argument --cuda-graph-bs: invalid int value: '16 32 48 64'` | pass the ints **unquoted** |
| 2 | scheduler init | `AttributeError: 'NoneType' object has no attribute 'get_seqlens_expanded'`, all 8 ranks die | `--disable-piecewise-cuda-graph` |
| 3 | inference | boots clean, full speed, **gsm8k 0.000 / Invalid 1.000** | `--dsa-prefill-backend trtllm --dsa-decode-backend trtllm` |

**(1) Quoted bucket list.** The old script passed `--cuda-graph-bs "16 32 48 64"`; bash delivers that as a
single argv element and the flag takes `nargs='+'` of ints. On Kubernetes/GKE each integer must be its own
element of the `args:` list. A container dying on an argparse error reads as a generic CrashLoopBackOff, so
this is the first thing to check on any orchestrated deployment.

**(2) Prefill CUDA-graph capture dereferences a null indexer.** The capture path added since 07-05 calls
`dsa_prefill_cuda_graph.py:122` → `dsa_indexer.py:1225 metadata.get_seqlens_expanded()`. Under
`--attention-backend flashinfer`, `FlashInferMLAAttnBackend.get_indexer_metadata()` returns `None` on this
stack — long-standing and previously harmless, because the indexer simply returned `None` top-k indices.
The new code dereferences it unconditionally.

**(3) `flashinfer_sparse_mla` silently corrupts.** Latest main auto-selects a new backend
(`Set DSA backends for GLM FP8 KV Cache on SM120/SM121: prefill=flashinfer_sparse_mla,
decode=flashinfer_sparse_mla`); 07-05 used `trtllm`. The server boots, reports healthy, and serves at full
throughput — but gsm8k is **0.000 with Invalid 1.000**, and completions emit a correct **first** token then
collapse into repeated `!` (token 0), the signature of NaN logits in decode. Forcing `trtllm` restores
gsm8k 0.900 **and** enlarges the KV pool (205,184 → 228,352 tok/rank). Prefill and decode were overridden
together, so it is **not yet isolated** whether prefill-`flashinfer_sparse_mla` alone is also affected; the
smoke-test signature points at decode.

> **Always gate.** Because blocker 3 is silent, a throughput number means nothing on its own. Run
> `python3 -m sglang.test.few_shot_gsm8k --num-questions 50 --port 8000` first — expect ~0.900 with
> Invalid 0.000. Invalid 1.000 means the DSA override did not take effect.

## Measured (2026-08-11, all three fixes applied)

| Metric | Value |
|---|---|
| KV pool | 228,352 tok/rank, fp8_e4m3, 11.75 GB/GPU |
| DSA backends (`/get_server_info`) | prefill=trtllm, decode=trtllm |
| gsm8k 50q | **0.900 Accuracy / 0.000 Invalid** |
| Saturated samples (#running-req ≥ 60) | 550 |
| Per-rank plateau | p25 408.2 / **median 422.4** / p75 440.2 tok/s |
| **Aggregate** | **~3379 tok/s** |
| **Per GPU** | **~422 tok/s** |

Chronological deciles (per-rank tok/s): 389 → 444 → 443 → 432 → 425 → 421 → 411 → 409 → 403 → 403.

| Config | KV | mfs | pool/rank | agg tok/s | /GPU | gsm8k |
|---|---|---:|---:|---:|---:|---:|
| **latest main `gd59c1ddf7`, 3 fixes** 🥇 | fp8_e4m3 | 0.975 | 228.4K | **~3379** | **422** | 0.900 |
| as-shipped on latest main (boot fixes only) | fp8_e4m3 | 0.975 | 205.2K | — | — | **0.000 ✗** |
| 07-05 anchor `gb28bc1060` | fp8_e4m3 | 0.975 | 229.7K | ~2680 | 335 | 0.900 |
| bf16 baseline (07-05) | bfloat16 | 0.94 | 91.5K | ~1264 | 158 | 0.920 |

## `--cuda-graph-bs` is still required (tested, rejected)

Breakable/piecewise CUDA graph being the default does **not** retire the minimal-bucket trick. Removing
`--cuda-graph-bs` at mfs 0.975 OOMs during capture:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 252.00 MiB.
GPU 5 ... 259.06 MiB is free ... 686.00 MiB allocated in private pools (e.g., CUDA Graphs)
```

The default bucket list captures far more decode graphs than the pool can spare. It is a **memory** lever,
not merely a batch cap.

## Open: the batch ceiling is not actually probed

`#running-req` pinned at exactly **64/rank** with `#queue-req: 0` and token usage 0.68 median / 0.93 peak.
64 is the client's `--max-concurrency 512` ÷ 8 DP ranks — which lands on precisely the same value as the
cuda-graph `max_bs`. **The two constraints are indistinguishable at this concurrency**, so 422/GPU is a
sound sustained figure but not demonstrably the ceiling. Resolving it needs a concurrency sweep, which
changes the workload and breaks comparability with the 07-05 anchor; deliberately not run here.

## Carried forward from 2026-07-05 (still true)

- **fp8-on-stock unblock.** fp8_e4m3 is ~1.8–2.5× denser than bf16 for the DSA KV pool, so at a given
  `--mem-fraction-static` the pool holds far more tokens → more concurrent sequences survive at saturation
  → higher decode plateau. Decode is **KV-pool-bound**; every memory-freeing lever converts into pool.
- **Minimal-env ablation (A1a).** The **only** env var fp8 needs is `SGLANG_DISABLE_DSA_INDEXER_FUSION=1`.
  `SGLANG_SM120_FLASHMLA_BACKEND` and `SGLANG_OPT_FLASHMLA_SPARSE_PREFILL` are **inert** for GLM (different
  DSA dispatch than DeepSeek-V4). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` still required for boot.
- **mfs 0.975 is the ceiling** (0.98 OOMs at boot).
- **MoE runner:** `flashinfer_cutlass` is the only viable NVFP4 runner on SM120 (marlin gsm8k 0.02;
  cutedsl/trtllm have no SM120 build).
- **Spec:** `--speculative-eagle-topk` must be 1 (tree spec blocked on SM120 flashinfer-MLA). ⚠️ **Not
  re-verified on latest main** — EAGLE spec decode on SM120 is reported blocked by upstream regression
  #29787 from v0.5.15. The 07-05 figure (~330/GPU, gsm8k 0.940) stands only for pin `gb28bc1060`.

## Config

TP=8 + DP-attention (dp8), `--moe-a2a-backend none --ep-size 1 --moe-runner-backend flashinfer_cutlass`,
`--attention-backend flashinfer`, `--kv-cache-dtype fp8_e4m3`, **`--dsa-prefill-backend trtllm
--dsa-decode-backend trtllm`**, `--disable-shared-experts-fusion`, **`--disable-piecewise-cuda-graph`**,
`--mem-fraction-static 0.975`, `--chunked-prefill-size 2048`, **`--cuda-graph-bs 16 32 48 64`** (unquoted),
`--context-length 9472`, `--max-running-requests 1024`, + `SGLANG_DISABLE_DSA_INDEXER_FUSION=1` +
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + the NCCL/GLOO block.

Full run data: `_raw/glm5.2/rerun_20260810_latest_main/` (2026-08-11 re-run: boot-failure tracebacks, both
gsm8k gates, `/get_server_info`, server plateau log) and `runs/20260705_glm5.2_sota_humanize/` (original
hill-climb ledger and ablations) in the gcp-kimi repo.
