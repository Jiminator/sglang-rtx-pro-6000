```
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    inf       
Max request concurrency:                 80        
Successful requests:                     320       
Benchmark duration (s):                  28986.73  
Total input tokens:                      1367861   
Total input text tokens:                 1367861   
Total generated tokens:                  10046674  
Total generated tokens (retokenized):    10045893  
Request throughput (req/s):              0.01      
Input token throughput (tok/s):          47.19     
Output token throughput (tok/s):         346.60    
Peak output token throughput (tok/s):    1274.00   
Peak concurrent requests:                81        
Total token throughput (tok/s):          393.78    
Concurrency:                             70.99     
Accept length:                           3.70      
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   6430862.10
Median E2E Latency (ms):                 7239630.58
P90 E2E Latency (ms):                    8284086.86
P99 E2E Latency (ms):                    8830692.47
---------------Time to First Token----------------
Mean TTFT (ms):                          5288588.39
Median TTFT (ms):                        6275737.61
P99 TTFT (ms):                           7300529.41
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          42.76     
Median TPOT (ms):                        31.21     
P99 TPOT (ms):                           231.32    
---------------Inter-Token Latency----------------
Mean ITL (ms):                           36.38     
Median ITL (ms):                         19.42     
P95 ITL (ms):                            30.02     
P99 ITL (ms):                            40.91     
Max ITL (ms):                            1807811.73
==================================================
```
### Benchmark Command
    echo "🚀 Starting sglang Serving Benchmark: 8K Input / 64K Output / 80 Concurrency..."
      
      API_URL="http://10.0.0.16:30000"
      MODEL_NAME="Kimi-K2.6"
      CONCURRENCY=80
      INPUT_LEN=8192
      OUTPUT_LEN=65536
      NUM_PROMPTS=320
      
      python3 -m sglang.bench_serving \
          --backend sglang \
          --base-url $API_URL \
          --model $MODEL_NAME \
          --tokenizer moonshotai/Kimi-K2.6 \
          --dataset-name random \
          --random-input-len $INPUT_LEN \
          --random-output-len $OUTPUT_LEN \
          --num-prompts $NUM_PROMPTS \
          --max-concurrency $CONCURRENCY \
          --output-file /workspace/results_sglang_8k_64k_80c.json \
          --apply-chat-template \
          --seed 42 \
          --request-rate inf

