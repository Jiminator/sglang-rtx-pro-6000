============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    9999.0
Max request concurrency:                 512
Successful requests:                     1536
Benchmark duration (s):                  1842.67
Total input tokens:                      783038
Total output tokens:                     6168083
Total output tokens (retokenized):       6164007
Request throughput (req/s):              0.8336
Input token throughput (tok/s):          424.95
Output token throughput (tok/s):         3347.36
Total token throughput (tok/s):          3772.30
Max output tokens per second:            4712.00
Concurrency (mean / cap):                414.07 / 512
Peak concurrency (server-observed):      517
Avg output tokens per request:           4015.68

----------------- E2E Latency (ms) ---------------
Mean:                                    496741.94
Median:                                  503209.68
Std:                                     284243.25
P90:                                     888829.65
P99:                                     1033207.60

--------------- Time to First Token --------------
Mean TTFT (ms):                          5056.69
Median TTFT (ms):                        294.63
Std TTFT (ms):                           8099.44
P99 TTFT (ms):                           26297.83

----- Time per Output Token (excl. 1st) ----------
Mean TPOT (ms):                          125.63
Median TPOT (ms):                        126.55
Std TPOT (ms):                           18.80
P99 TPOT (ms):                           166.40

--------------- Inter-token Latency --------------
Mean ITL (ms):                           122.58
Median ITL (ms):                         116.89
Std ITL (ms):                            138.49
P95 ITL (ms):                            182.36
P99 ITL (ms):                            259.85
==================================================

Workload context:
  dataset:        autobench (random, range_ratio=0.0)
  random_input_len cli flag:  1024   # bench_serving CLI default; not used when --dataset-path is set
  random_output_len cli flag: 1024  # same — actual output_lens come from prepared_dataset.jsonl
  actual avg output_len:      4015.68             # 6168083 total / 1536 requests

Source:
  bundle:    /Users/jimmy/gcp-kimi/_raw/k2.5-nvfp4/2node-userspec-dp8-bf16-flashinfer-2026-05-03
  results:   2node-userspec-dp8-bf16-flashinfer-2026-05-03/auto_benchmark_results/k2.5-nvfp4-2node-userspec-dp8-bf16-flashinfer-2026-05-03/results.jsonl
