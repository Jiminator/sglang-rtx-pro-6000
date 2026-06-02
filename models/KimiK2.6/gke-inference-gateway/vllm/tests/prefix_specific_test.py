import time
from openai import OpenAI

VLLM_API_URL = "http://10.0.0.17/v1"
client = OpenAI(base_url=VLLM_API_URL, api_key="token-not-required")

# Two distinct prefixes (approx 600 tokens each)
PREFIX_A = "Once upon a time in a galaxy far, far away... " * 50
PREFIX_B = "A long time ago, dinosaurs roamed the Earth... " * 50

def send_request(prefix, label):
    print(f"\n[Sending] {label}...")
    full_prompt = prefix + f" Request for {label}."
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model="/models",
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=20,
        )
        print(f"[Completed] in {time.time() - start_time:.4f} seconds")
    except Exception as e:
        print(f"[Failed] {e}")

print("=" * 60)
print("Multi-Prefix Specific Routing Test")
print("=" * 60)

# 1. Target Pod X with Prefix A
send_request(PREFIX_A, "Prefix A (First Run)")
print("Waiting 5 seconds for cache state to propagate...")
time.sleep(5)

# 2. Target Pod Y with Prefix B
send_request(PREFIX_B, "Prefix B (First Run)")
print("Waiting 5 seconds for cache state to propagate...")
time.sleep(5)

# 3. Retarget Pod X with Prefix A
send_request(PREFIX_A, "Prefix A (Second Run - Should hit same pod as first run)")
print("Waiting 5 seconds...")
time.sleep(5)

# 4. Retarget Pod Y with Prefix B
send_request(PREFIX_B, "Prefix B (Second Run - Should hit same pod as first run)")

print("\n" + "=" * 60)
print("Test complete. Please check the logs of both vLLM pods.")
print("Expectation:")
print(" - One pod handles both 'Prefix A' requests.")
print(" - The other pod handles both 'Prefix B' requests.")
print("=" * 60)
