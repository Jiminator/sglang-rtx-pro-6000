# Agentic Kimi K2.6 Replay Benchmark

This repository contains scripts to benchmark SGLang using Kimi K2.6 agentic traces.

## Benchmark Options

There are two main scripts available for running the benchmark:

### 1. Original Benchmark (`agentic_benchmark.py`)
- **Parallelism**: 256
- **Data Source**: Designed to use the original dataset. (Note: Currently expects `data.jsonl` to be unpacked).
- **Usage**:
  ```bash
  python3 agentic_benchmark.py http://<sglang-server-ip>:30000
  ```

### 2. Low Load Benchmark (`agentic_benchmark_sglang_low_load.py`)
- **Parallelism**: 64 (Default, configurable via `--parallelism`)
- **Data Source**: Uses the unpacked `data.jsonl` file.
- **Usage**:
  ```bash
  python3 agentic_benchmark_sglang_low_load.py http://<sglang-server-ip>:30000 --parallelism 64
  ```

## Setup and Prerequisites

1. **Data File**: The benchmark requires `data.jsonl`. If you only have `data.jsonl.zst`, you must decompress it first:
   ```bash
   # Example using zstd command line tool
   zstd -d data.jsonl.zst
   ```
2. **Dependencies**:
   - `httpx`
   - `asyncio`

   ```bash
   pip install httpx
   ```

## Running on Kubernetes
For detailed instructions on building Docker images and running on GKE, see [README.GKE.md](README.GKE.md).
