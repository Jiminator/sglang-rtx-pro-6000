# Gemma 4 26B-A4B — BF16 latency configurations

Measured September 17–18, 2026 on NVIDIA RTX PRO 6000 Blackwell Server Edition
96 GB. Stock SGLang **v0.5.19**, no serving patches or quantization; BF16 weights
and BF16 KV. The two objectives were **mean OR median E2E below 7 seconds at
10K/500** and **below 4 seconds at 10K/300**. Median equals p50.

Tested image:

```text
lmsysorg/sglang@sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9
```

Commands below run inside that image. Model paths refer to read-only local
copies mounted under `/models/google/`. Tested target revision:
`4d7ae4984b7db7de8f8457170b3f1a419ee76d52`; assistant revision:
`6e5aaaf4c42b98394530b8fda2e95cadd65c151c`.

## 7-second target: 10K input / 500 output

**Six independent TP1 replicas on one node, SMG round-robin, natural acceptance.**
Both runs met the median target; neither mean was below 7 seconds.

### Server command (per GPU)

Launch one worker per GPU, each on a distinct port. Set `GPU_ID` and `PORT`
for each worker; for example GPU IDs 0–5 and ports 32200–32205. When Docker
exposes only one physical GPU, its in-container `GPU_ID` is 0.

```bash
env -u SGLANG_SIMULATE_ACC_LEN -u SGLANG_SIMULATE_ACC_METHOD \
  -u SGLANG_SIMULATE_ACC_TOKEN_MODE CUDA_VISIBLE_DEVICES="$GPU_ID" \
  python3 -m sglang.launch_server \
  --model-path /models/google/gemma-4-26B-A4B-it \
  --dtype bfloat16 --kv-cache-dtype bf16 \
  --reasoning-parser gemma4 --tool-call-parser gemma4 \
  --speculative-algorithm NEXTN \
  --speculative-draft-model-path /models/google/gemma-4-26B-A4B-it-assistant \
  --speculative-num-steps 5 --speculative-num-draft-tokens 6 \
  --speculative-eagle-topk 1 --mem-fraction-static 0.9 \
  --max-running-requests 32 --cuda-graph-max-bs-decode 32 \
  --tp-size 1 --dp-size 1 --chunked-prefill-size 8192 \
  --disable-radix-cache --host 0.0.0.0 --port "$PORT"
```

NEXTN resolves to Gemma's Frozen-KV MTP. Radix/prefix reuse is disabled, not
per-request KV. Route to all six workers using the SMG command below.

### Benchmark command

```bash
python3 -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port 32550 \
  --model /models/google/gemma-4-26B-A4B-it \
  --dataset-name random --random-input-len 10000 --random-output-len 500 \
  --random-range-ratio 1 --num-prompts 160 --max-concurrency 32 \
  --request-rate inf --seed 42 --temperature 0 --tokenize-prompt \
  --output-details --output-file result-10k500.jsonl
```

Both runs used seed 42, the stock one-request warmup and EOS ignored (default).
Use a separate output file for the repeat. All 160 requests completed per run:
1,600,000 input tokens and 80,000 output tokens.

### Results

| Run | Mean E2E | Median/p50 | p90 | p95 | p99 | Output tok/s |
|---|---:|---:|---:|---:|---:|---:|
|First|7.371s|**6.719s**|9.946s|10.956s|13.159s|1978.1|
|Repeat|7.010s|**6.651s**|8.676s|9.798s|10.802s|2098.6|

## 4-second target: 10K input / 300 output

**Sixteen independent TP1 replicas, eight per node, SMG round-robin.**
This result uses **simulated acceptance length 4**: three accepted draft tokens
plus one target-derived bonus token. It overrides verification and is **not
lossless generation, accuracy evidence, or a real-traffic SLA guarantee**.
Do not expose these simulation workers to user traffic.

### Server command (per GPU, eight workers on each node)

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" \
SGLANG_SIMULATE_ACC_LEN=4 \
SGLANG_SIMULATE_ACC_METHOD=match-expected \
SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token \
python3 -m sglang.launch_server \
  --model-path /models/google/gemma-4-26B-A4B-it \
  --dtype bfloat16 --kv-cache-dtype bf16 \
  --reasoning-parser gemma4 --tool-call-parser gemma4 \
  --speculative-algorithm NEXTN \
  --speculative-draft-model-path /models/google/gemma-4-26B-A4B-it-assistant \
  --speculative-num-steps 3 --speculative-num-draft-tokens 4 \
  --speculative-eagle-topk 1 --mem-fraction-static 0.9 \
  --max-running-requests 32 --cuda-graph-max-bs-decode 32 \
  --tp-size 1 --dp-size 1 --chunked-prefill-size 8192 \
  --disable-radix-cache --host 0.0.0.0 --port "$PORT"
```

### Benchmark command

First run a full-shape warmup with `--num-prompts 64`. Then run this command for
seeds 42, 123 and 2026, with a distinct output file for each:

```bash
python3 -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port 32550 \
  --model /models/google/gemma-4-26B-A4B-it \
  --dataset-name random --random-input-len 10000 --random-output-len 300 \
  --random-range-ratio 1 --num-prompts 160 --max-concurrency 32 \
  --request-rate inf --seed 42 --temperature 0 --tokenize-prompt \
  --output-details --output-file result-10k300-seed42.jsonl
```

Router and benchmark client ran on the same node; eight workers were local
and eight reached over private IPs. Every run completed 160 requests with
1,600,000 input and 48,000 output tokens. A client-side observer verified peak
concurrency 32; finite runs include ramp-up/drain. No retractions were logged.

### Results

| Seed | Mean E2E | Median/p50 | p90 | p95 | p99 | Output tok/s |
|---|---:|---:|---:|---:|---:|---:|
|42|3.966s|**3.866s**|4.282s|4.934s|5.974s|2248.7|
|123|3.971s|**3.867s**|4.269s|4.887s|5.941s|2241.7|
|2026|4.002s|**3.886s**|4.710s|4.867s|5.873s|2248.2|

Median passed in all three seeds. Mean was borderline: the third run was
**4.001611s**, so this is not a reliable sub-4s mean or tail-latency result.

The same 3/4 configuration on eight GPUs measured 5.275–5.302s mean and
5.084–5.095s median. At that eight-GPU budget, 3/4 reduced mean by 5.6% versus
5/6 (both simulated acceptance 4) and 12.9% versus no-spec. Shorter 1/2 and
2/3 drafts did not improve E2E.

For context, **natural-acceptance 5/6 on 21 GPUs** measured 3.367–3.458s mean
and 3.132–3.157s median at 10K/300. That is a different GPU budget and acceptance
mode, not validation of natural acceptance on the two-node 3/4 configuration.

## SMG routing (both configurations)

Run in the same pinned image, supplying all individual worker URLs in a Bash
array. Use six URLs for the 7s configuration or sixteen for the 4s simulation:

```bash
# Example construction for two nodes; set NODE_A_IP and NODE_B_IP first.
workers=()
for host in "$NODE_A_IP" "$NODE_B_IP"; do
  for port in {32200..32207}; do workers+=("http://$host:$port"); done
done
smg launch --host 127.0.0.1 --port 32550 --prometheus-port 33550 \
  --policy round_robin --worker-urls "${workers[@]}" \
  --worker-startup-check-interval 5 --max-concurrent-requests -1 \
  --history-backend none --disable-retries
```

For the six-worker setup, construct URLs for one host, ports 32200–32205.
Wait until `/workers` reports all workers healthy before benchmarking. Restrict
worker ports to trusted hosts; these commands do not configure authentication.
Do not mix simulated and natural workers in the same router.

Results come from saved `bench_serving` JSONL: `smg-round-robin` and
`smg-round-robin-repeat` (10K/500), `dp16_s3t4_20260918` (16-GPU 10K/300),
`node2_draft_ablation_20260918` (eight-GPU ablation), and
`fixed5_acc4_20260918/fleet-fixed5base` (21-GPU natural baseline).
Full operational bundles are retained separately, not published here.
