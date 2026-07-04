# DSV4-Flash bench_serving results (2026-07-04, rrr=1.0, TP=8+DPA, SM120)

Throughput = steady-state decode plateau read from server logs (long OSL ⇒ requests don't complete
in-window ⇒ #running-req pins at the SWA-pool admission ceiling). Aggregate = per-rank gen-tps × 8 (DPA).
Correctness gate: gsm8k 50q on idle server before loadgen. KV = fp8_e4m3 (bf16 code-rejected on DSV4).

## 1k/8k (ISL 1024 / OSL 8192)

| # | Ckpt | Image | MoE | decode | mfs | gsm8k | plateau /rank | **agg tok/s** | /GPU | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| prior | FP8 | v0.5.13.post1 | triton | (dsv4) | 0.85 | — | — | 3412.9 (offline) | 426.6 | old offline anchor |
| A5-n3 | FP8 | v0.5.13.post1 | triton | dsv4(fi) | 0.85 | **0.980** | 192@451 (tail 170@399) | **~3611** peak / ~3450 sust | ~451 | ✅ reproduces anchor; SWA pool saturates (swa=1.00) |
| **A9-n3** | **MXFP4** | dev-cu13 #28231 | **marlin** | **2env** | 0.85 | **0.960** | 192@457.5 | **~3660** | **~457** | ✅✅ **#28231 works on latest main — 7.4× the old 497 GEMV**; concurrency-bound (full 0.04/swa 0.44) |
| **A9-n1** | **NVFP4** | dev-cu13 #28231 | flashinfer_cutlass | **2env** | 0.85 | **0.960** | 192@460.3 | **~3682** | **~460** | ✅ latest main; concurrency-bound (full 0.04/swa 0.42) |
| **A9-n4** | **FP8** | dev-cu13 #28231 | triton | 2env | 0.85 | **0.960** | 192@451.3 | **~3610** | **~451** | ✅ = v0.5.13.post1 FP8 (3611); concurrency-bound (swa 0.38-0.57) |

### CONC=3072 hill-climb (384/rank; all latest-main+2env, running)
| # | Ckpt | MoE | CONC | plateau /rank | agg | /GPU | notes |
|---|---|---|---|---|---|---|---|
| **C-n4** | **FP8** | triton | 3072 | **256@461.8** | **~3694** | **~462** | queue 128/rank; swa 0.40-0.65 |
| **C-n1** | **NVFP4** | flashinfer_cutlass | 3072 | **256@469.1** | **~3753** | **~469** | ← 1k/8k LEADER; swa 0.29-0.48 |
| **C-n3** | **MXFP4** | marlin | 3072 | **256@464.6** | **~3717** | **~465** | swa 0.30-0.50 |

All 3 capped at 256/rank by **max_running_requests=2048÷8**, NOT pool (full 0.05/swa≤0.65) or VRAM (~10GB free); 128/rank standing queue; only ~+2% vs 192/rank → **past BW knee**.

**MXFP4 MAXRUN=3072 (384/rank) ceiling check → PREFILL-BOUND** (all-prefill tail, cuda graph False, #running-req ~285 still ramping, no decode plateau in 600s). Deepening admission past 256/rank just grows the prefill backlog without decode gain. ⇒ **256/rank (MAXRUN=2048) is the practical 1k/8k ceiling.**

### ★ 1k/8k VERDICT (DONE) — latest-main dev-cu13 + 2env, TP8+DPA, mfs0.85, MAXRUN=2048 (256/rank), gsm8k 0.960
| Rank | Ckpt | MoE | **tok/s** | /GPU |
|---|---|---|---|---|
| 🥇 | **NVFP4** (nvidia) | flashinfer_cutlass | **~3753** | **469** |
| 🥈 | **MXFP4-Marlin** (deepseek-ai, #28231) | marlin | ~3717 | 465 |
| 🥉 | **FP8** (sgl-project) | triton | ~3694 | 462 |
All within ~1.6% (near-noise). vs prior offline anchor 3413 (+10%). **MXFP4 = 7.4× the pre-#28231 GEMV (497).** Decode is BW-bound (per-req 1.83 tok/s @256/rank); no further config lever (pool empty, mfs won't help, spec net-loss on DSV4).

**1k/8k ceiling insight:** at 256/rank throughput is ~plateaued (per-req rate collapsing: 2.40→1.83 tok/s/req from 192→256/rank). The binder is `max_running_requests` (256/rank), but decode is BW-bound so raising it → diminishing returns. NVFP4 ~3753 is near the 1k/8k ceiling. mfs won't help (pool 0.05 full/0.47 swa, empty). Only a faster decode kernel or (net-loss) spec would move it.

**1k/8k verdict so far:** all 3 checkpoints ~equal at 192/rank (3610-3682, within 2%, all gsm8k 0.960), all concurrency-bound. NVFP4 marginally top. #28231 MXFP4-Marlin is the big story (7.4× vs old GEMV). Winner TBD by the CONC sweep. Note: ISL+OSL=9216=CTXLEN → some 400 rejections; widen CTXLEN→9472 for final ship runs.

**⚠️ 1k/8k is CONCURRENCY-bound at CONC=1536 (192/rank), not pool-bound** — NVFP4 pool 0.04 full/0.42 swa, max_total_num_tokens 6.9M (vs FP8 2.6M). Need a CONC sweep (2048/3072/4096) to find the true ceiling. FP8 swa did hit ~1.00 at 192/rank (tighter pool) so FP8 may cap sooner. "2env" = SGLANG_SM120_FLASHMLA_BACKEND=triton + SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=0.

## 8k/64k (ISL 8192 / OSL 65536)

| # | Ckpt | Image | MoE | mfs | plateau /rank | agg tok/s | /GPU | notes |
|---|---|---|---|---|---|---|---|---|
| prior | FP8 | v0.5.13.post1 | triton | 0.85 | — | 551.9 (offline, B=33) | 69.0 | old offline anchor |
| A5-n2 | FP8 | v0.5.13.post1 | triton | 0.85 | 8@128.2 (CONC=64) | ~1026 | 128 | concurrency-bound (pool 0.08) |
| c256 | FP8 | v0.5.13.post1 | triton | 0.85 | 32@206.5 (CONC=256) | **~1652** | 207 | still concurrency-bound (pool 0.15/0.21); per-req rate falling (16→6.45) → nearing BW knee |
| c1024 | FP8 | v0.5.13.post1 | triton | 0.85 | **PREFILL-SWAMPED** | — | — | 4474 prefill lines / 1 decode line; admitted only ~70/rank, queue growing; never reached decode plateau (bench 42/2048). CONC=1024 too high for 8k/64k |

**8k/64k measurement finding:** at inf request-rate, high CONC floods the 8192-token prefill backlog → server stays prefill-bound, no decode plateau. **CONC=256 (32/rank) is the valid steady-state point** (queue drains, clean decode). So 8k/64k has a prefill-saturation ceiling ~70/rank; measure at CONC=256. Re-running 3-way at CONC=256 on latest+2env:
| 8k-c256 | Ckpt | MoE | plateau/rank | agg | /GPU | |
|---|---|---|---|---|---|---|
| FP8 (v0.5.13.post1) | triton | 32@206.5 | ~1652 | 207 | concurrency-bound (pool 0.15/0.21) |
| **n1 NVFP4** | flashinfer_cutlass | **32@216.2** | **~1730** | **216** | ← 8k/64k LEADER; pool empty (full 0.06/swa 0.08) → concurrency-bound, room above 32/rank |
| n2 FP8 (latest+2env) | triton | 32@207.3 | ~1658 | 207 | = v0.5.13.post1 FP8 (1652) → cross-image parity confirmed |
| **n3 MXFP4** | marlin | **32@217.75** | **~1742** | **218** | ~tied with NVFP4; pool empty (concurrency-bound) |
| **n4 FP8 CONC=384** | triton | **48@221.5** | **~1772** | **222** | CLEAN (queue 0, all-decode); +7% vs CONC=256; per-req 6.47→4.61 (diminishing) |

### 8k/64k CONC=384 ceiling (48/rank, clean) + CONC=512 bound (running)
| Ckpt | CONC=384 plateau/rank | agg | /GPU | |
|---|---|---|---|---|
| FP8 | 48@221.5 | ~1772 | 222 | +7% vs c256 |
| **NVFP4** | **48@228.9** | **~1831** | **229** | ← 8k/64k LEADER (clean, queue 0); +5.8% vs c256 |
| **MXFP4** | **48@230.35** | **~1843** | **230** | marginally tops NVFP4 (clean, queue 0) |

**8k/64k CONC=384:** MXFP4 1843 ≈ NVFP4 1831 > FP8 1772 (FP4 ckpts tied, ~4% over FP8).

### ★ 8k/64k VERDICT — CONC=512 (64/rank = clean ceiling; ~70/rank swamps) latest-main dev-cu13 + 2env
| Rank | Ckpt | MoE | plateau/rank | **tok/s** | /GPU |
|---|---|---|---|---|---|
| 🥇 | **MXFP4** | marlin | 242.6 | **~1941** | **243** |
| 🥇 | **NVFP4** | flashinfer_cutlass | 241.7 | **~1934** | **242** |
| 🥉 | FP8 | triton | 235.1 | ~1881 | 235 |
FP8 curve 1658→1772→1881 (CONC 256→384→512, all clean); NVFP4 1730→1831→1934. **8k/64k ceiling ~1934 tok/s (242/GPU) = 3.5× the offline anchor (551.9).** Higher CONC prefill-swamps (no gain). Correctness: same kernels as 1k/8k (gsm8k 0.960); 8k/64k not separately gated (short-context eval).

8k/64k CONC=256 ranking: **MXFP4 1742 ≈ NVFP4 1730 > FP8 1652** (FP4 experts ~5% ahead of FP8; both FP4 tied). All concurrency-bound → true ceiling higher; CONC=384 probe pending.

8k/64k CONC=256 is concurrency-bound (pool empty) → the ceiling is higher, at CONC just below the ~70/rank prefill-saturation. NVFP4's 8k/64k lead (+4.7% vs FP8) > its 1k/8k lead (+1.6%).

## Key resolved facts
- FP8 1k/8k plateau: max_total_num_tokens=2,615,808; max_running_requests clamped to 256/rank; effective
  170-192/rank; per-req ~2.35 tok/s @ 192/rank (past the BW knee, aggregate ~flat vs prior).
- SWA pool is the binder (swa usage → 1.00, queue builds) → mfs / swa-ratio are the hill-climb levers.
- Image strategy: **latest dev-cu13 (#28231) + `SGLANG_SM120_FLASHMLA_BACKEND=triton`** for all 3 checkpoints
  (env-only, no patch). v0.5.13.post1 = proven cross-check (pre-#28231, so MXFP4 would be GEMV there).
