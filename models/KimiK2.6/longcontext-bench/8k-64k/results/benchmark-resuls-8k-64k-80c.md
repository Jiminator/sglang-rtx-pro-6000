```
============ Serving Benchmark Result ============
Backend:                                 vllm      
Traffic request rate:                    inf       
Max request concurrency:                 80        
Successful requests:                     320       
Benchmark duration (s):                  12236.25  
Total input tokens:                      1367861   
Total input text tokens:                 1367861   
Total generated tokens:                  10046674  
Total generated tokens (retokenized):    10044126  
Request throughput (req/s):              0.03      
Input token throughput (tok/s):          111.79    
Output token throughput (tok/s):         821.06    
Peak output token throughput (tok/s):    400.00    
Peak concurrent requests:                82        
Total token throughput (tok/s):          932.85    
Concurrency:                             70.79     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   2706971.73
Median E2E Latency (ms):                 2735626.20
P90 E2E Latency (ms):                    4675693.27
P99 E2E Latency (ms):                    5533202.60
---------------Time to First Token----------------
Mean TTFT (ms):                          337207.73 
Median TTFT (ms):                        108761.42 
P99 TTFT (ms):                           1587509.71
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          85.70     
Median TPOT (ms):                        70.81     
P99 TPOT (ms):                           287.59    
---------------Inter-Token Latency----------------
Mean ITL (ms):                           297.03    
Median ITL (ms):                         239.17    
P95 ITL (ms):                            269.95    
P99 ITL (ms):                            1277.61   
Max ITL (ms):                            1627517.96
==================================================
```
