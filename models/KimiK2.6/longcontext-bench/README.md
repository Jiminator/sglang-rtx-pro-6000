# Long Context Benchmark Results (Kimi K2.6)

Summary of benchmark results for the **64k-8k** and **8k-64k** configurations across various concurrency levels.

**Machine Type:** Single Node

### 64k-8k Benchmark Results
| Config | inp | out | concurrency | num prompts | Input Throughput (tok/s) | Output Throughput (tok/s) | Total Throughput (tok/s) | Mean E2E Latency (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **64k-8k** | 64k | 8k | 80 | 320 | **3019.04** | 361.75 | **3380.79** | 817.64 |

### 8k-64k Benchmark Results
| Config | inp | out | concurrency | num prompts | Input Throughput (tok/s) | Output Throughput (tok/s) | Total Throughput (tok/s) | Mean E2E Latency (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **8k-64k** | 8k | 64k | 64 | 128 | 95.66 | 813.76 | 909.42 | 2162.38 |
| **8k-64k** | 8k | 64k | 80 | 320 | 111.79 | 821.06 | 932.85 | 2706.97 |
| **8k-64k** | 8k | 64k | 128 | 512 | 108.68 | **830.50** | 939.19 | 4311.04 |

---
*   **inp / out**: Targeted context window and generation lengths.
*   **num prompts**: Total number of successful requests completed in the run.
*   **Throughput**: Measured in tokens per second (tok/s).
