```
============ Serving Benchmark Result ============
Backend:                                 vllm      
Traffic request rate:                    inf       
Max request concurrency:                 128       
Successful requests:                     512       
Benchmark duration (s):                  19561.99  
Total input tokens:                      2126077   
Total input text tokens:                 2126077   
Total generated tokens:                  16246267  
Total generated tokens (retokenized):    16243349  
Request throughput (req/s):              0.03      
Input token throughput (tok/s):          108.68    
Output token throughput (tok/s):         830.50    
Peak output token throughput (tok/s):    512.00    
Peak concurrent requests:                130       
Total token throughput (tok/s):          939.19    
Concurrency:                             112.83    
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   4311043.43
Median E2E Latency (ms):                 4455394.72
P90 E2E Latency (ms):                    6388585.12
P99 E2E Latency (ms):                    7101641.76
---------------Time to First Token----------------
Mean TTFT (ms):                          1576086.59
Median TTFT (ms):                        1931370.18
P99 TTFT (ms):                           2902807.51
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          116.72    
Median TPOT (ms):                        80.17     
P99 TPOT (ms):                           457.42    
---------------Inter-Token Latency----------------
Mean ITL (ms):                           338.09    
Median ITL (ms):                         242.88    
P95 ITL (ms):                            345.41    
P99 ITL (ms):                            1280.70   
Max ITL (ms):                            2930392.83
==================================================
```

