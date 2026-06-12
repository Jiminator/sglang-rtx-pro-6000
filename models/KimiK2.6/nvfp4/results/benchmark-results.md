## Serving Benchmark (2560 Prompts, 512 Concurrency)
* **Configuration**:
  * **Dataset**: Random (1024 input / 8192 output max)
  * **Prompts**: 2560
  * **Max Concurrency**: 512
  * **Quantization**: FP4 (`modelopt_fp4`)
  * **Tokenizer**: In-process (zero tokenizer worker processes)

```
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    inf       
Max request concurrency:                 512       
Successful requests:                     2560      
Benchmark duration (s):                  3244.56   
Total input tokens:                      1302740   
Total input text tokens:                 1302740   
Total generated tokens:                  10581410  
Total generated tokens (retokenized):    10583956  
Request throughput (req/s):              0.79      
Input token throughput (tok/s):          401.52    
Output token throughput (tok/s):         3261.28   
Peak output token throughput (tok/s):    4725.00   
Peak concurrent requests:                515       
Total token throughput (tok/s):          3662.79   
Concurrency:                             447.84    
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   567590.24 
Median E2E Latency (ms):                 577204.41 
P90 E2E Latency (ms):                    1012867.20
P99 E2E Latency (ms):                    1117259.64
---------------Time to First Token----------------
Mean TTFT (ms):                          14192.20  
Median TTFT (ms):                        14265.82  
P99 TTFT (ms):                           30499.76  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          138.54    
Median TPOT (ms):                        136.32    
P99 TPOT (ms):                           159.64    
---------------Inter-Token Latency----------------
Mean ITL (ms):                           133.92    
Median ITL (ms):                         0.32      
P95 ITL (ms):                            1278.02   
P99 ITL (ms):                            2577.08   
Max ITL (ms):                            29632.17  
==================================================
```

## Batch serving benchmark (bench_one_batch_server)
* **Configuration**:
  * **Batch size**: 512
  * **Input sequence length**: 1024 tokens
  * **Output sequence length**: 8192 tokens
  * **Quantization**: FP4 (`modelopt_fp4`)
  * **Tokenizer**: In-process (zero tokenizer worker processes)

| Metric | Value |
| --- | --- |
| Total latency | **1,138.99 s** (18.98 mins) |
| Input prefill throughput | **14,015.66 tokens/s** |
| Output decode throughput | **3,807.51 tokens/s** |
| Overall token throughput | **4,142.77 tokens/s** |
| Slowest request TTFT | **37.41 s** |
| Average generation speed per rank | **487.57 tokens/s** |

## Serving Benchmark (1536 Prompts, 512 Concurrency)
* **Configuration**:
  * **Dataset**: Random (1024 input / 8192 output max)
  * **Prompts**: 1536
  * **Max Concurrency**: 512
  * **Quantization**: FP4 (`modelopt_fp4`)
  * **Tokenizer**: Multi-worker (tokenizer-worker-num 8)

```
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    inf       
Max request concurrency:                 512       
Successful requests:                     1536      
Benchmark duration (s):                  2083.20   
Total input tokens:                      784969    
Total input text tokens:                 784969    
Total generated tokens:                  6434886   
Total generated tokens (retokenized):    6437437   
Request throughput (req/s):              0.74      
Input token throughput (tok/s):          376.81    
Output token throughput (tok/s):         3088.95   
Peak output token throughput (tok/s):    4654.00   
Peak concurrent requests:                514       
Total token throughput (tok/s):          3465.76   
Concurrency:                             416.35    
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   564674.76 
Median E2E Latency (ms):                 572117.33 
P90 E2E Latency (ms):                    983879.46 
P99 E2E Latency (ms):                    1122744.02
---------------Time to First Token----------------
Mean TTFT (ms):                          13863.79  
Median TTFT (ms):                        13897.45  
P99 TTFT (ms):                           30154.17  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          138.61    
Median TPOT (ms):                        135.38    
P99 TPOT (ms):                           153.77    
---------------Inter-Token Latency----------------
Mean ITL (ms):                           131.51    
Median ITL (ms):                         0.37      
P95 ITL (ms):                            1161.30   
P99 ITL (ms):                            2206.86   
Max ITL (ms):                            29747.33  
==================================================
```