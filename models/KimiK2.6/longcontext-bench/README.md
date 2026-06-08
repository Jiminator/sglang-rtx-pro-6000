# Long Context Benchmark Results (Kimi K2.6)

Summary of benchmark results for the **64k-8k** and **8k-64k** configurations across various concurrency levels.

**Machine Type:** Single Node (8x NVIDIA GPUs)

### vLLM Server Configuration
The benchmarks were conducted against a vLLM server with the following configuration:
*   **Model:** Kimi-K2.6
*   **Tensor Parallel (TP):** 8
*   **Decode Context Parallel:** 8
*   **Max Model Length:** 262,144 (256k)
*   **KV Cache Dtype:** fp8
*   **Attention Backend:** TRITON_MLA
*   **Speculative Decoding:** Eagle3 (3 speculative tokens)
    *   **Draft Model:** `lightseekorg/kimi-k2.5-eagle3-mla`
*   **Features:** Chunked Prefill, Prefix Caching, Async Scheduling enabled.
*   **Image:** `voipmonitor/vllm:kimi-k26-mtp-upstream-stack-pcie-env-test-20260424`

### 64k-8k Benchmark Results
| Config | inp | out | concurrency | Input (tok/s) | Output (tok/s) | Total (tok/s) | Mean E2E (s) | Configs | Results |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **64k-8k** | 64k | 8k | 80 | **3019.04** | 361.75 | **3380.79** | 817.64 | [Server](./64k-8k/vllm-kimik2.6-1node-g4-reflection.yaml) / [Bench](./64k-8k/vllm-kimi-benchmark-64k-8k-80c.yaml) | [Raw Result](./64k-8k/results/benchmark-results-64k-8k-80c.yaml) |

### 8k-64k Benchmark Results
| Config | inp | out | concurrency | Input (tok/s) | Output (tok/s) | Total (tok/s) | Mean E2E (s) | Configs | Results |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **8k-64k** | 8k | 64k | 64 | 95.66 | 813.76 | 909.42 | 2162.38 | [Server](./8k-64k/vllm-kimik26-server-8k-64k.yaml) / [Bench](./8k-64k/benchmark/vllm-kimik26-benchmark-8k-64k-64c.yaml) | [Raw Result](./8k-64k/results/benchmark-resuls-8k-64k-64c.md) |
| **8k-64k** | 8k | 64k | 80 | 111.79 | 821.06 | 932.85 | 2706.97 | [Server](./8k-64k/vllm-kimik26-server-8k-64k.yaml) / [Bench](./8k-64k/benchmark/vllm-kimik26-benchmark-8k-64k-80c.yaml) | [Raw Result](./8k-64k/results/benchmark-resuls-8k-64k-80c.md) |
| **8k-64k** | 8k | 64k | 128 | 108.68 | **830.50** | 939.19 | 4311.04 | [Server](./8k-64k/vllm-kimik26-server-8k-64k.yaml) / [Bench](./8k-64k/benchmark/vllm-kimik26-benchmark-8k-64k-128c.yaml) | [Raw Result](./8k-64k/results/benchmark-resuls-8k-64k-128c.md) |

---
*   **inp / out**: Targeted context window and generation lengths.
*   **Throughput**: Measured in tokens per second (tok/s).
