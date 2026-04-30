```
============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    inf       
Max request concurrency:                 512       
Successful requests:                     1536      
Benchmark duration (s):                  2233.75   
Total input tokens:                      784969    
Total input text tokens:                 784969    
Total generated tokens:                  6434886   
Total generated tokens (retokenized):    6422237   
Request throughput (req/s):              0.69      
Input token throughput (tok/s):          351.41    
Output token throughput (tok/s):         2880.76   
Peak output token throughput (tok/s):    4093.00   
Peak concurrent requests:                517       
Total token throughput (tok/s):          3232.17   
Concurrency:                             405.23    
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   589313.75 
Median E2E Latency (ms):                 587101.46 
P90 E2E Latency (ms):                    1031300.15
P99 E2E Latency (ms):                    1174895.74
---------------Time to First Token----------------
Mean TTFT (ms):                          5380.77   
Median TTFT (ms):                        330.27    
P99 TTFT (ms):                           27810.86  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          147.96    
Median TPOT (ms):                        142.60    
P99 TPOT (ms):                           168.43    
---------------Inter-Token Latency----------------
Mean ITL (ms):                           139.82    
Median ITL (ms):                         133.54    
P95 ITL (ms):                            195.57    
P99 ITL (ms):                            283.99    
Max ITL (ms):                            532007.30 
==================================================
```