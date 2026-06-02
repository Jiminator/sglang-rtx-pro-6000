import asyncio
import os
import time
from openai import AsyncOpenAI

VLLM_API_URL = os.getenv("VLLM_API_URL", "http://10.0.0.17/v1")
client = AsyncOpenAI(base_url=VLLM_API_URL, api_key="token-not-required")

PREFIX_A = "Once upon a time in a galaxy far, far away... " * 50
PREFIX_B = "A long time ago, dinosaurs roamed the Earth... " * 50

async def send_request(prefix, label, i):
    full_prompt = prefix + f" Request {i} for {label}."
    try:
        response = await client.chat.completions.create(
            model="/models",
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=20,
        )
        return True
    except Exception as e:
        print(f"Failed for {label} {i}: {e}")
        return False

async def main():
    print("=" * 60)
    print("Sending 10 requests for Prefix A and 20 for Prefix B...")
    print("=" * 60)
    
    tasks = []
    for i in range(10):
        tasks.append(send_request(PREFIX_A, "Prefix A", i))
    for i in range(20):
        tasks.append(send_request(PREFIX_B, "Prefix B", i))
        
    start_time = time.time()
    await asyncio.gather(*tasks)
    
    print("=" * 60)
    print(f"All 30 requests completed in {time.time() - start_time:.4f} seconds")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
