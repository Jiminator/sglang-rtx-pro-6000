```
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    inf       
Max request concurrency:                 512       
Successful requests:                     1536      
Benchmark duration (s):                  2071.06   
Total input tokens:                      784969    
Total input text tokens:                 784969    
Total generated tokens:                  6434886   
Total generated tokens (retokenized):    6436932   
Request throughput (req/s):              0.74      
Input token throughput (tok/s):          379.02    
Output token throughput (tok/s):         3107.05   
Peak output token throughput (tok/s):    4848.00   
Peak concurrent requests:                514       
Total token throughput (tok/s):          3486.07   
Concurrency:                             416.85    
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   562060.47 
Median E2E Latency (ms):                 572022.82 
P90 E2E Latency (ms):                    978105.25 
P99 E2E Latency (ms):                    1111836.31
---------------Time to First Token----------------
Mean TTFT (ms):                          14760.55  
Median TTFT (ms):                        15244.50  
P99 TTFT (ms):                           28926.42  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          140.11    
Median TPOT (ms):                        134.60    
P99 TPOT (ms):                           151.26    
---------------Inter-Token Latency----------------
Mean ITL (ms):                           130.67    
Median ITL (ms):                         0.34      
P95 ITL (ms):                            1192.59   
P99 ITL (ms):                            2191.89   
Max ITL (ms):                            27781.58  
==================================================
```
