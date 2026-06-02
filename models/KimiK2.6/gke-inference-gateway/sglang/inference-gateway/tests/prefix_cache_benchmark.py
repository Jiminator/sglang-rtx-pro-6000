import os
import time
from openai import OpenAI

# Gateway IP
VLLM_API_URL = os.getenv("VLLM_API_URL", "http://10.0.0.17/v1")

client = OpenAI(base_url=VLLM_API_URL, api_key="token-not-required")

# A long prefix to populate cache (approx 400 tokens)
LONG_PREFIX = "Once upon a time in a galaxy far, far away... " * 50

def send_request(prompt_suffix):
    full_prompt = LONG_PREFIX + prompt_suffix
    start_time = time.time()
    
    try:
        response = client.chat.completions.create(
            model="/models",
            messages=[{"role": "user", "content": full_prompt}],
            stream=True,
            max_tokens=50,
        )
        
        ttft = None
        for chunk in response:
            delta = chunk.choices[0].delta
            content = getattr(delta, 'content', None)
            reasoning = getattr(delta, 'reasoning_content', None)
            if ttft is None and (content or reasoning):
                ttft = time.time() - start_time
                print(f"Time to First Token (TTFT): {ttft:.4f} seconds")
            # print content if needed, but we just want TTFT
            
        total_time = time.time() - start_time
        print(f"Total Time: {total_time:.4f} seconds")
        return ttft
    except Exception as e:
        print(f"Error: {e}")
        return None

print("=" * 60)
print("Prefix Cache Aware Routing Benchmark")
print("=" * 60)

print("\n[Request 1] Sending request to populate cache...")
send_request("Tell me a short joke.")

print("\nWaiting 5 seconds for cache state to settle in EPP...")
time.sleep(5)

for i in range(2, 21):
    print(f"\n[Request {i}] Sending request with SAME prefix...")
    send_request(f"Tell me a short story part {i}.")

print("\n" + "=" * 60)
print("Benchmark complete. Please check the logs of vllm-kimi-g4-0 and vllm-kimi-g4-1")
print("to verify routing distribution.")
print("=" * 60)
