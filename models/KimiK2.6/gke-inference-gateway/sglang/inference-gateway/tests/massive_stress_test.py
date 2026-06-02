import asyncio
import os
import time
import uuid
from openai import AsyncOpenAI

VLLM_API_URL = os.getenv("VLLM_API_URL", "http://10.10.0.24/v1")
client = AsyncOpenAI(base_url=VLLM_API_URL, api_key="token-not-required")

# Generate 20 NET NEW distinct prefixes using UUIDs
PREFIXES = [
    f"Prefix {uuid.uuid4()}: Once upon a time in a galaxy far, far away... " * 20
    for _ in range(20)
]

# Use a semaphore to avoid overwhelming the client pod's file descriptors/connections
sem = asyncio.Semaphore(500)

async def send_request(i, prefix_idx, prefix, label):
    async with sem:
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
    print("Sending 1000 requests (20 prefixes * 50) with max concurrency 500...")
    print("=" * 60)
    
    start_time = time.time()
    
    tasks = []
    for p_idx, prefix in enumerate(PREFIXES):
        for i in range(50):
            tasks.append(send_request(i, p_idx, prefix, f"Prefix-{p_idx}"))
        
    await asyncio.gather(*tasks)
    
    print("=" * 60)
    print(f"All 1000 requests completed in {time.time() - start_time:.4f} seconds")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
