# Per-concurrency bench summaries

Full canonical `Serving Benchmark Result` block from each of the 9 sweep runs, ordered from highest to lowest concurrency.

### conc=256 (num_prompts=768)

```
============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    9999.0    
Max request concurrency:                 256       
Successful requests:                     768       
Benchmark duration (s):                  1635.60   
Total input tokens:                      6234010   
Total input text tokens:                 6234010   
Total generated tokens:                  383663    
Total generated tokens (retokenized):    381875    
Request throughput (req/s):              0.47      
Input token throughput (tok/s):          3811.45   
Output token throughput (tok/s):         234.57    
Peak output token throughput (tok/s):    2243.00   
Peak concurrent requests:                260       
Total token throughput (tok/s):          4046.02   
Concurrency:                             248.93    
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   530143.82 
Median E2E Latency (ms):                 483150.32 
P90 E2E Latency (ms):                    1006776.42
P99 E2E Latency (ms):                    1348582.45
---------------Time to First Token----------------
Mean TTFT (ms):                          61091.31  
Median TTFT (ms):                        12182.33  
P99 TTFT (ms):                           343637.09 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          1430.35   
Median TPOT (ms):                        1041.74   
P99 TPOT (ms):                           7169.58   
---------------Inter-Token Latency----------------
Mean ITL (ms):                           945.94    
Median ITL (ms):                         620.93    
P95 ITL (ms):                            994.72    
P99 ITL (ms):                            12713.55  
Max ITL (ms):                            414376.64 
==================================================
```

### conc=128 (num_prompts=384)

```
============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    9999.0    
Max request concurrency:                 128       
Successful requests:                     384       
Benchmark duration (s):                  1044.01   
Total input tokens:                      3122115   
Total input text tokens:                 3122115   
Total generated tokens:                  206807    
Total generated tokens (retokenized):    205851    
Request throughput (req/s):              0.37      
Input token throughput (tok/s):          2990.49   
Output token throughput (tok/s):         198.09    
Peak output token throughput (tok/s):    1408.00   
Peak concurrent requests:                132       
Total token throughput (tok/s):          3188.58   
Concurrency:                             123.46    
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   335660.78 
Median E2E Latency (ms):                 329956.50 
P90 E2E Latency (ms):                    585032.54 
P99 E2E Latency (ms):                    741610.40 
---------------Time to First Token----------------
Mean TTFT (ms):                          31620.63  
Median TTFT (ms):                        10236.75  
P99 TTFT (ms):                           176978.67 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          828.32    
Median TPOT (ms):                        622.30    
P99 TPOT (ms):                           1978.16   
---------------Inter-Token Latency----------------
Mean ITL (ms):                           568.31    
Median ITL (ms):                         466.31    
P95 ITL (ms):                            819.38    
P99 ITL (ms):                            6221.37   
Max ITL (ms):                            186965.92 
==================================================
```

### conc=64 (num_prompts=192)

```
============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    9999.0    
Max request concurrency:                 64        
Successful requests:                     192       
Benchmark duration (s):                  626.92    
Total input tokens:                      1533042   
Total input text tokens:                 1533042   
Total generated tokens:                  95057     
Total generated tokens (retokenized):    94830     
Request throughput (req/s):              0.31      
Input token throughput (tok/s):          2445.37   
Output token throughput (tok/s):         151.63    
Peak output token throughput (tok/s):    817.00    
Peak concurrent requests:                67        
Total token throughput (tok/s):          2597.00   
Concurrency:                             61.10     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   199493.78 
Median E2E Latency (ms):                 184026.64 
P90 E2E Latency (ms):                    375805.15 
P99 E2E Latency (ms):                    499267.28 
---------------Time to First Token----------------
Mean TTFT (ms):                          16712.00  
Median TTFT (ms):                        7932.42   
P99 TTFT (ms):                           86265.53  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          478.16    
Median TPOT (ms):                        408.52    
P99 TPOT (ms):                           3459.48   
---------------Inter-Token Latency----------------
Mean ITL (ms):                           370.77    
Median ITL (ms):                         81.65     
P95 ITL (ms):                            706.14    
P99 ITL (ms):                            1010.49   
Max ITL (ms):                            102388.93 
==================================================
```

### conc=32 (num_prompts=96)

```
============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    9999.0    
Max request concurrency:                 32        
Successful requests:                     96        
Benchmark duration (s):                  401.33    
Total input tokens:                      766186    
Total input text tokens:                 766186    
Total generated tokens:                  50056     
Total generated tokens (retokenized):    50039     
Request throughput (req/s):              0.24      
Input token throughput (tok/s):          1909.14   
Output token throughput (tok/s):         124.73    
Peak output token throughput (tok/s):    495.00    
Peak concurrent requests:                34        
Total token throughput (tok/s):          2033.87   
Concurrency:                             30.27     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   126526.79 
Median E2E Latency (ms):                 113131.22 
P90 E2E Latency (ms):                    218373.57 
P99 E2E Latency (ms):                    275483.54 
---------------Time to First Token----------------
Mean TTFT (ms):                          10138.63  
Median TTFT (ms):                        7177.51   
P99 TTFT (ms):                           41028.23  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          236.29    
Median TPOT (ms):                        245.29    
P99 TPOT (ms):                           462.97    
---------------Inter-Token Latency----------------
Mean ITL (ms):                           223.85    
Median ITL (ms):                         67.84     
P95 ITL (ms):                            615.30    
P99 ITL (ms):                            766.49    
Max ITL (ms):                            43446.78  
==================================================
```

### conc=16 (num_prompts=48)

```
============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    9999.0    
Max request concurrency:                 16        
Successful requests:                     48        
Benchmark duration (s):                  248.92    
Total input tokens:                      346056    
Total input text tokens:                 346056    
Total generated tokens:                  23842     
Total generated tokens (retokenized):    23834     
Request throughput (req/s):              0.19      
Input token throughput (tok/s):          1390.25   
Output token throughput (tok/s):         95.78     
Peak output token throughput (tok/s):    288.00    
Peak concurrent requests:                18        
Total token throughput (tok/s):          1486.03   
Concurrency:                             14.21     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   73674.41  
Median E2E Latency (ms):                 70491.01  
P90 E2E Latency (ms):                    128533.74 
P99 E2E Latency (ms):                    175415.54 
---------------Time to First Token----------------
Mean TTFT (ms):                          7010.81   
Median TTFT (ms):                        4670.90   
P99 TTFT (ms):                           27500.77  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          133.30    
Median TPOT (ms):                        147.23    
P99 TPOT (ms):                           223.99    
---------------Inter-Token Latency----------------
Mean ITL (ms):                           134.56    
Median ITL (ms):                         58.31     
P95 ITL (ms):                            530.26    
P99 ITL (ms):                            650.24    
Max ITL (ms):                            19274.70  
==================================================
```

### conc=8 (num_prompts=24)

```
============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    9999.0    
Max request concurrency:                 8         
Successful requests:                     24        
Benchmark duration (s):                  168.42    
Total input tokens:                      186398    
Total input text tokens:                 186398    
Total generated tokens:                  13226     
Total generated tokens (retokenized):    13224     
Request throughput (req/s):              0.14      
Input token throughput (tok/s):          1106.72   
Output token throughput (tok/s):         78.53     
Peak output token throughput (tok/s):    160.00    
Peak concurrent requests:                9         
Total token throughput (tok/s):          1185.25   
Concurrency:                             7.17      
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   50292.47  
Median E2E Latency (ms):                 48259.93  
P90 E2E Latency (ms):                    88774.84  
P99 E2E Latency (ms):                    102982.50 
---------------Time to First Token----------------
Mean TTFT (ms):                          5181.49   
Median TTFT (ms):                        5094.75   
P99 TTFT (ms):                           12816.83  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          83.96     
Median TPOT (ms):                        81.61     
P99 TPOT (ms):                           147.30    
---------------Inter-Token Latency----------------
Mean ITL (ms):                           82.09     
Median ITL (ms):                         51.32     
P95 ITL (ms):                            358.12    
P99 ITL (ms):                            587.22    
Max ITL (ms):                            8286.95   
==================================================
```

### conc=4 (num_prompts=12)

```
============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    9999.0    
Max request concurrency:                 4         
Successful requests:                     12        
Benchmark duration (s):                  115.85    
Total input tokens:                      79080     
Total input text tokens:                 79080     
Total generated tokens:                  5942      
Total generated tokens (retokenized):    5939      
Request throughput (req/s):              0.10      
Input token throughput (tok/s):          682.59    
Output token throughput (tok/s):         51.29     
Peak output token throughput (tok/s):    87.00     
Peak concurrent requests:                5         
Total token throughput (tok/s):          733.88    
Concurrency:                             3.42      
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   32998.40  
Median E2E Latency (ms):                 27372.47  
P90 E2E Latency (ms):                    51005.69  
P99 E2E Latency (ms):                    72015.76  
---------------Time to First Token----------------
Mean TTFT (ms):                          3671.35   
Median TTFT (ms):                        2905.50   
P99 TTFT (ms):                           8608.64   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          60.41     
Median TPOT (ms):                        59.26     
P99 TPOT (ms):                           73.92     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           59.40     
Median ITL (ms):                         47.74     
P95 ITL (ms):                            48.63     
P99 ITL (ms):                            476.89    
Max ITL (ms):                            2111.48   
==================================================
```

### conc=2 (num_prompts=6)

```
============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    9999.0    
Max request concurrency:                 2         
Successful requests:                     6         
Benchmark duration (s):                  102.45    
Total input tokens:                      47986     
Total input text tokens:                 47986     
Total generated tokens:                  3446      
Total generated tokens (retokenized):    3446      
Request throughput (req/s):              0.06      
Input token throughput (tok/s):          468.38    
Output token throughput (tok/s):         33.64     
Peak output token throughput (tok/s):    46.00     
Peak concurrent requests:                3         
Total token throughput (tok/s):          502.02    
Concurrency:                             1.96      
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   33422.83  
Median E2E Latency (ms):                 39946.31  
P90 E2E Latency (ms):                    50195.42  
P99 E2E Latency (ms):                    53005.36  
---------------Time to First Token----------------
Mean TTFT (ms):                          4121.34   
Median TTFT (ms):                        4418.82   
P99 TTFT (ms):                           8360.64   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          48.72     
Median TPOT (ms):                        50.42     
P99 TPOT (ms):                           53.93     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           51.11     
Median ITL (ms):                         45.21     
P95 ITL (ms):                            45.95     
P99 ITL (ms):                            330.24    
Max ITL (ms):                            660.09    
==================================================
```

### conc=1 (num_prompts=3)

```
============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    9999.0    
Max request concurrency:                 1         
Successful requests:                     3         
Benchmark duration (s):                  86.89     
Total input tokens:                      25759     
Total input text tokens:                 25759     
Total generated tokens:                  1747      
Total generated tokens (retokenized):    1747      
Request throughput (req/s):              0.03      
Input token throughput (tok/s):          296.47    
Output token throughput (tok/s):         20.11     
Peak output token throughput (tok/s):    25.00     
Peak concurrent requests:                2         
Total token throughput (tok/s):          316.58    
Concurrency:                             1.00      
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   28958.30  
Median E2E Latency (ms):                 30953.80  
P90 E2E Latency (ms):                    42947.83  
P99 E2E Latency (ms):                    45646.49  
---------------Time to First Token----------------
Mean TTFT (ms):                          4474.96   
Median TTFT (ms):                        6472.85   
P99 TTFT (ms):                           6838.65   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          42.43     
Median TPOT (ms):                        43.45     
P99 TPOT (ms):                           43.61     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           42.12     
Median ITL (ms):                         43.51     
P95 ITL (ms):                            43.69     
P99 ITL (ms):                            43.79     
Max ITL (ms):                            44.39     
==================================================
```
