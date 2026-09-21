# DeepSeek-V4.1-Flash, 8K/1K, 8x RTX PRO 6000

The official [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) checkpoint (MXFP4 routed experts, FP8 dense layers) on one g4-standard-384 node, served with `lmsysorg/sglang:dev-dsv41` (SGLang `da64c5cb`). No source patches.

The starting point is the [upstream single-node recipe](https://github.com/shivajid/sglang-rtx-pro-6000/blob/main/models/DeepSeekv4.1-Flash/sglang-dsv41-flash-1node.yaml): TP=8, EP=8, FP8 KV cache, and Engram tables in pinned host RAM. Four flag changes raise output throughput from 441.7 to 511.6 tok/s.

## Results

256 requests of exactly 8,192 input and 1,024 output tokens at max concurrency 128. The prefix cache is flushed after warmup, and the cache report shows 0% hits. Raw `bench_serving` output is in `results/`.

| Config | Output tok/s | Total tok/s | Mean TTFT | Mean TPOT |
|---|---|---|---|---|
| Upstream recipe | 441.7 | 3,976 | 108 s | 184 ms |
| This recipe | 511.6 | 4,604 | 98 s | 155 ms |

GSM8K (5-shot, all 1,319 questions, two runs each) scores 90.5 and 90.5 on the upstream recipe and 90.3 and 90.0 on this one. The standard error is about 0.8 points.

## Changes

Each row adds one change to the row above.

| Change | Output tok/s | vs upstream |
|---|---|---|
| `--cuda-graph-max-bs-decode 128` | 452.2 | +2.4% |
| `NCCL_MIN_NCHANNELS=16` | 474.6 | +7.4% |
| `--fp8-gemm-backend flashinfer_cutlass` | 507.7 | +14.9% |
| `--chunked-prefill-size 16384 --max-prefill-tokens 32768` | 511.6 | +15.8% |

- Upstream captures decode CUDA graphs only up to batch 64, so batch-128 decode ran eagerly.
- The node has no NVLink, so all-reduce runs over PCIe. With 16 channels, a 1.3 MB all-reduce drops from 0.19 to 0.12 ms and an 84 MB one from 7.0 to 5.6 ms. More channels did not help.
- The dense FP8 weights use 32x32 blocks with ue8m0 scales. FlashInfer runs them as MXFP8 GEMMs that read the same FP8 values and scales as the default Triton kernel.
- 16K prefill chunks added about 1% both times they were tested.

## Limits

In the upstream recipe, prefilling 256 prompts of 8K tokens at about 5,000 tok/s takes roughly 420 of the 593 seconds. About 60% of prefill GPU time goes to SGLang's Triton sparse-MLA kernel. FlashInfer's SM120 sparse-MLA kernel would replace it, but it rejects this model's attention shape at both 8 and 16 heads per GPU. NCCL all-reduce takes another 22%.

These were slower or did not start:

- TP=4 with 2 replicas: -12%. Prefill got 24% faster, but decode ran at half speed.
- Marlin MoE: -4.6%.
- `triton_kernel` MoE and EP=1 fail at load. EP=1 splits the 2,304 expert intermediate size into 288 per GPU, which the FP4 MoE kernel rejects because it needs multiples of 128.
- Removing `--enable-decoder-swa-bounded-replay`: -37%.
- `--enable-mixed-chunk`: -3.4%.
- Shared Engram host layout: -1.8%, because this host has shmem transparent huge pages off.
- DP attention cannot be combined with `--enable-decoder-swa-bounded-replay`. Two-batch overlap and pipeline parallelism are not supported for V4.1 in this image.

32K chunks gained under 1%. Single-batch overlap, fused MoE all-reduce, Engram on GPU, and the DeepGEMM env vars changed nothing.

DSpark speculative decoding reached 521 tok/s with an acceptance length of 5.26 out of 6. The random dataset builds each prompt by repeating ShareGPT text, which inflates acceptance, so that number would not carry over to real traffic and DSpark is left out.

## Upstream numbers

The upstream README tables do not match its own raw result files. From concurrency 32 up, they also claim more output throughput than their own TTFT and TPOT allow. The raw files reproduce within 3% with two settings the recipe does not state: Engram on the GPU (61.5 GB of weights per GPU) and a 260,608-token KV pool. The upstream sweep also resends earlier prompts without flushing the cache, and `--random-range-ratio 0.0` averages about 4.3K input and 0.5K output tokens per request.

## Run

```bash
MODEL_DIR=/path/to/DeepSeek-V4.1-Flash ./launch.sh
./bench.sh
```

The pinned Engram tables need about 175 GB of free host RAM. Startup takes about 4 minutes.
