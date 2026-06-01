import asyncio
import time
from openai import AsyncOpenAI

VLLM_API_URL = "http://10.0.0.17/v1"
client = AsyncOpenAI(base_url=VLLM_API_URL, api_key="token-not-required")

LONG_PREFIX = "Once upon a time in a galaxy far, far away... " * 50

async def send_request(i):
    full_prompt = LONG_PREFIX + f" Request {i}: Tell me a short joke."
    start_time = time.time()
    try:
        response = await client.chat.completions.create(
            model="/models",
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=20,
        )
        total_time = time.time() - start_time
        print(f"Request {i} completed in {total_time:.4f} seconds")
        return total_time
    except Exception as e:
        print(f"Request {i} failed: {e}")
        return None

async def main():
    print("=" * 60)
    print("Sending 20 concurrent requests to Gateway...")
    print("=" * 60)
    
    start_time = time.time()
    tasks = [send_request(i) for i in range(20)]
    await asyncio.gather(*tasks)
    
    print("=" * 60)
    print(f"All 20 requests completed in {time.time() - start_time:.4f} seconds")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
