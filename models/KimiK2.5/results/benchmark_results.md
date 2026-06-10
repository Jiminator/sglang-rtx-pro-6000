```
============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    inf       
Max request concurrency:                 512       
Successful requests:                     1536      
Benchmark duration (s):                  2041.01   
Total input tokens:                      784969    
Total input text tokens:                 784969    
Total generated tokens:                  6434886   
Total generated tokens (retokenized):    6430891   
Request throughput (req/s):              0.75      
Input token throughput (tok/s):          384.60    
Output token throughput (tok/s):         3152.79   
Peak output token throughput (tok/s):    4793.00   
Peak concurrent requests:                516       
Total token throughput (tok/s):          3537.39   
Concurrency:                             408.77    
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   543171.09 
Median E2E Latency (ms):                 540813.49 
P90 E2E Latency (ms):                    955299.04 
P99 E2E Latency (ms):                    1076984.74
---------------Time to First Token----------------
Mean TTFT (ms):                          5105.02   
Median TTFT (ms):                        300.83    
P99 TTFT (ms):                           25162.33  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          136.52    
Median TPOT (ms):                        127.84    
P99 TPOT (ms):                           148.49    
---------------Inter-Token Latency----------------
Mean ITL (ms):                           128.60    
Median ITL (ms):                         121.06    
P95 ITL (ms):                            188.31    
P99 ITL (ms):                            278.54    
Max ITL (ms):                            25355.98  
==================================================
```
