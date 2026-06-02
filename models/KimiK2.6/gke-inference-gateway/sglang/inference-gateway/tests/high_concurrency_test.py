import asyncio
import os
import time
from openai import AsyncOpenAI

VLLM_API_URL = os.getenv("VLLM_API_URL", "http://10.10.0.24/v1")
client = AsyncOpenAI(base_url=VLLM_API_URL, api_key="token-not-required")

PREFIX_A = "Once upon a time in a galaxy far, far away... " * 50
PREFIX_B = "A long time ago, dinosaurs roamed the Earth... " * 50

async def send_request(i, prefix, label):
    full_prompt = prefix + f" Request {i} for {label}: Tell me a short joke."
    start_time = time.time()
    try:
        response = await client.chat.completions.create(
            model="/models",
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=20,
        )
        total_time = time.time() - start_time
        print(f"Request {i} ({label}) completed in {total_time:.4f} seconds")
        return total_time
    except Exception as e:
        print(f"Request {i} ({label}) failed: {e}")
        return None

async def main():
    print("=" * 60)
    print("Sending 35 concurrent requests for Prefix A and 35 for Prefix B...")
    print("=" * 60)
    
    start_time = time.time()
    
    tasks = []
    for i in range(35):
        tasks.append(send_request(i, PREFIX_A, "Prefix A"))
    for i in range(35):
        tasks.append(send_request(i, PREFIX_B, "Prefix B"))
        
    await asyncio.gather(*tasks)
    
    print("=" * 60)
    print(f"All 70 requests completed in {time.time() - start_time:.4f} seconds")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
