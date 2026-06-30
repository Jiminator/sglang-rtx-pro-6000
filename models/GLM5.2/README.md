# GLM-5.2-NVFP4 on GCP G4 (RTX PRO 6000 / SM120)

Optimized SGLang serving configs and benchmarks for [`nvidia/GLM-5.2-NVFP4`](https://huggingface.co/nvidia/GLM-5.2-NVFP4) on a single `g4-standard-384` node (8× RTX PRO 6000 Blackwell, SM120, PCIe, no NVLink).

## Model Overview
GLM-5.2 is a DeepSeek-Sparse-Attention (DSA) MoE (`GlmMoeDsaForCausalLM`): 78 layers, 256 routed experts top-8, `kv_lora_rank` 512, MTP/NEXTN layer 78. The `nvidia/GLM-5.2-NVFP4` release is **expert-only NVFP4** — only the routed experts are quantized; attention, shared experts, dense layers 0–2 and the MTP layer stay bf16. gsm8k 0.920 (base). Pin the complete snapshot `b0b2b68d4be5ee00e95ae013ea0949fe5c0b5a56` (the `refs/main` tag drifted to a weightless partial).

## TL;DR — what to ship
**Max throughput + lowest latency: EAGLE speculative decoding** (3 steps / 4 draft tokens). It is the throughput SOTA *and* it Pareto-dominates the non-speculative config across the entire concurrency range — ~2× the throughput per GPU at matched load and ~3× lower inter-token latency at low load (accept-length ≈ 3.9 on this model's well-trained MTP head). Recipe + launch script: [`nvfp4/launch-eagle-spec.sh`](./nvfp4/launch-eagle-spec.sh).

| Config | Metric | tok/s/GPU | Notes |
|---|---|---|---|
| **EAGLE 3-step (SOTA)** | sustained, steady-state | **~300** | gsm8k 0.94 · accept-len 3.9 · also the latency winner |
| non-spec fp8 (max-batch) | one-shot, b191 | 194.6 | gsm8k 0.97 · simplest high-throughput config |
| stock bf16 dense (baseline) | one-shot, b88 | 147.0 | runs on stock dev-cu13, no special branch |

Headline progression: **stock 147 → fp8-KV + memory levers 194.6 (+32%) → EAGLE spec ~300 (+104% over stock)**, all at ISL 1,024 / OSL 8,192, single node.

## Hardware & Stack
- **Setup:** 1× `g4-standard-384` — 8× RTX PRO 6000 Blackwell (95.6 GiB/GPU), PCIe (no NVLink/RDMA).
- **Image:** `sglang-glmopt:tf512` = `lmsysorg/sglang:nightly-dev-cu13-20260623` + transformers 5.12, with the **`glm-opt` branch** (commit `8a57f86c3`) installed editable. The glm-opt branch is **required** for the two big unlocks below; on stock dev-cu13 the SOTA caps at the 147 bf16-dense config.
- **Boot env (mandatory on glm-opt):** `SGLANG_DISABLE_DSA_INDEXER_FUSION=1` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (without them the #27705 indexer rope buffer OOMs at construction under DP-attention), plus the load-bearing NCCL/GLOO env block.

## The optimization story (what actually moved the number)
1. **bf16 KV + flashinfer attention is the only stock-viable path** on SM120 (fp8 KV / the `dsa` backend crash — no SM120 DSA decode kernel on stock). DP-attention + the `cuda-graph-max-bs = batch` trick → **147 tok/s/GPU** (stock SOTA, no code patch).
2. **fp8_e4m3 KV works on the glm-opt branch** (gsm8k 0.96) and ~1.6×'s the KV pool. Three *stacking memory levers* — fp8 KV + minimal per-worker cuda-graph buckets (`--cuda-graph-bs "8 16 24 32"`) + chunked-prefill shrink — each unlock a higher `mem-fraction-static` notch → **194.6 tok/s/GPU** (best non-spec, +32%).
3. **EAGLE speculative decoding is the real win** → **~300 tok/s/GPU sustained** (+104% over stock). The official MTP head accepts ~3.9 of 4 draft tokens, so each forward emits ~4 tokens; even though the draft+verify pool halves the batch ceiling, the per-request speedup more than pays for it. **3-step beats 4-step** (ties on steady-state throughput but wins whole-run, ITL, and gsm8k). `EAGLE topk>1` (tree spec) is hard-blocked on SM120 (flashinfer-MLA is topk=1 only).

See [`results/sota-summary.md`](./results/sota-summary.md) for the full lever-by-lever sweep and the closed/dead dimensions.

## Pareto curves (concurrency sweeps)
Throughput per GPU vs per-user token rate (y = tok/s/GPU, x = tok/s/user = 1000/ITL), measured with `bench_serving` on the identical workload — up-and-to-the-right is better. Full data + the zipfian/HiCache plot in [`results/concurrency-sweep.md`](./results/concurrency-sweep.md).

![GLM-5.2 throughput/latency Pareto — EAGLE vs non-spec (random 1k/8k)](results/pareto_random_1k8k.png)

- **Random 1k/8k (no cache):** EAGLE Pareto-dominates non-spec at *every* concurrency — higher tok/s/GPU and lower ITL simultaneously (cc 1→128).
- **Zipfian shared-prefix (full 2×2 — {non-spec, EAGLE} × {radix-L1, +L2 HiCache}):** the GPU radix cache (L1) already captures the prefix reuse, so **L2 HiCache barely shifts throughput** (~+3% non-spec; EAGLE ~flat). Its real effect is **hit-rate**: past cc=128 the active decode KV evicts L1 prefixes, and the L2 host tier retains them — **L2 lifts EAGLE's hit-rate from ~36% to ~55% at cc256/512 (+19 pts)** (EAGLE's smaller pool evicts L1 sooner). But the higher hit-rate doesn't convert to throughput at OSL=8192 — the bottleneck is decode-generated KV, which caching can't dedup. **Takeaway: radix L1 is sufficient for 1k/8k; enable L2 HiCache only if your distinct-prefix working set overflows the GPU radix tree.**

## Correctness (gsm8k, the working recipes)
| Config | gsm8k | note |
|---|---|---|
| EAGLE 3-step (fp8 KV) | 0.94 | accept-len 3.9, invalid 0.000 |
| non-spec fp8 (b191) | 0.97 (200q) | |
| stock bf16 dense | 0.92 | base reference |

## Usage
```bash
# Best config — EAGLE speculative decoding (throughput SOTA + latency winner)
bash nvfp4/launch-eagle-spec.sh

# Alternative — non-spec fp8 (simpler, max-batch one-shot)
bash nvfp4/launch-nonspec.sh
```
Both target a single 8-GPU node on the `sglang-glmopt:tf512` image. For continuous serving at concurrency *above* the pool ceiling, drop `--mem-fraction-static` to 0.92 and `--chunked-prefill-size` to 1024 to keep over-subscription queueing gracefully instead of hitting the prefill-activation OOM.

**Attribution:** GLM-5.2-NVFP4 SM120 tuning campaign — RadixArk serving team.
