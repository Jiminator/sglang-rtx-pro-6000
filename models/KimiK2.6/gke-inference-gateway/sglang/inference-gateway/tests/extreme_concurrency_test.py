import asyncio
import os
import time
from openai import AsyncOpenAI

VLLM_API_URL = os.getenv("VLLM_API_URL", "http://10.10.0.24/v1")
client = AsyncOpenAI(base_url=VLLM_API_URL, api_key="token-not-required")

# Generate 10 distinct prefixes
PREFIXES = [
    f"Prefix {i}: Once upon a time in a galaxy far, far away... " * 20
    for i in range(10)
]

async def send_request(i, prefix_idx, prefix, label):
    full_prompt = prefix + f" Request {i} for {label}: Tell me a short joke."
    start_time = time.time()
    try:
        response = await client.chat.completions.create(
            model="/models",
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=20,
        )
        total_time = time.time() - start_time
        print(f"Req {i} ({label}) completed in {total_time:.4f} seconds")
        return total_time
    except Exception as e:
        print(f"Req {i} ({label}) failed: {e}")
        return None

async def main():
    print("=" * 60)
    print("Sending 500 concurrent requests (10 prefixes * 50)...")
    print("=" * 60)
    
    start_time = time.time()
    
    tasks = []
    for p_idx, prefix in enumerate(PREFIXES):
        for i in range(50):
            tasks.append(send_request(i, p_idx, prefix, f"Prefix-{p_idx}"))
        
    await asyncio.gather(*tasks)
    
    print("=" * 60)
    print(f"All 500 requests completed in {time.time() - start_time:.4f} seconds")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
