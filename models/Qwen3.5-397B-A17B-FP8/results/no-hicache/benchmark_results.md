```
============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    inf       
Max request concurrency:                 40        
Successful requests:                     2000      
Benchmark duration (s):                  2286.28   
Total input tokens:                      19715409  
Total input text tokens:                 19715409  
Total generated tokens:                  985972    
Total generated tokens (retokenized):    983891    
Request throughput (req/s):              0.87      
Input token throughput (tok/s):          8623.37   
Output token throughput (tok/s):         431.26    
Peak output token throughput (tok/s):    1160.00   
Peak concurrent requests:                44        
Total token throughput (tok/s):          9054.63   
Concurrency:                             39.83     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   45534.45  
Median E2E Latency (ms):                 45704.63  
P90 E2E Latency (ms):                    81299.53  
P99 E2E Latency (ms):                    96086.49  
---------------Time to First Token----------------
Mean TTFT (ms):                          1371.28   
Median TTFT (ms):                        1128.88   
P99 TTFT (ms):                           14052.80  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          90.45     
Median TPOT (ms):                        90.41     
P99 TPOT (ms):                           140.58    
---------------Inter-Token Latency----------------
Mean ITL (ms):                           89.97     
Median ITL (ms):                         36.25     
P95 ITL (ms):                            299.88    
P99 ITL (ms):                            337.00    
Max ITL (ms):                            1618.68   
==================================================

```
