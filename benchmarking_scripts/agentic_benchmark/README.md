# Agentic Benchmark for SGLang

Standalone replay benchmark for the Athena RL Exodus Kimi K2.6 trace slice, modified for SGLang.

The runtime interface is intentionally just endpoint URLs:

```bash
python3 agentic_benchmark_sglang.py http://host:8000/v1
python3 agentic_benchmark_sglang.py http://host0:8000/v1 http://host1:8000/v1
```

The script keeps the workload fixed in `agentic_benchmark_sglang.py`: model name (`/models`), sampling
parameters, parallelism, timeout, and the dataset `data.jsonl` path. It groups
rows by `sequence_id`, sorts by `step_index`, runs each sequence in order, and
runs up to `PARALLELISM` sequences concurrently. With multiple URLs, each
sequence is assigned to one URL by a stable hash of `sequence_id`.

The dataset file `data.jsonl` contains the original flat benchmark format,
one request per line:

```json
{"sequence_id":"...","step_index":0,"messages":[...],"tools":[...]}
```

Each run writes a timestamped `results-YYYYMMDD-HHMMSS.json` summary in the
current directory with latency, throughput, token counts, prompt cache hit rate,
and per-endpoint summaries.

