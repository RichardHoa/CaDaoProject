import socket
import requests
import json
import time
import ollama

# Force IPv4 to prevent tunnel timeout
socket.getaddrinfo = lambda *args, **kwargs: [
    (socket.AF_INET, socket.SOCK_STREAM, 6, "", (args[0], args[1]))
]

# Remote API config
APOLLO_URL = "https://apollo.quocanmeomeo.io.vn/v1/chat/completions"
APOLLO_MODEL = "sailor2:8b"
APOLLO_API_KEY = "FUV_danghoa_4lLx6kx1oWMn6H86Px0J"

# Local model config
LOCAL_MODEL = "sailor2:8b"

# Test question
TEST_QUESTION = "What is the meaning of life after all"
NUM_ITERATIONS = 10


def test_local_model():
    """Test local Ollama model and return response time in seconds."""
    start = time.time()
    response = ollama.chat(
        model=LOCAL_MODEL, messages=[{"role": "user", "content": TEST_QUESTION}]
    )
    elapsed = time.time() - start
    # Access response content to ensure it's complete
    _ = response.message.content
    return elapsed


def test_remote_model():
    """Test remote Apollo model and return response time in seconds."""
    start = time.time()

    payload = {
        "model": APOLLO_MODEL,
        "messages": [{"role": "user", "content": TEST_QUESTION}],
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {APOLLO_API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post(APOLLO_URL, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    _ = response.json()["choices"][0]["message"]["content"]

    elapsed = time.time() - start
    return elapsed


def run_benchmark():
    print(f"=== Benchmark: Local vs Remote ===")
    print(f'Test question: "{TEST_QUESTION}"')
    print(f"Iterations: {NUM_ITERATIONS}\n")

    local_times = []
    remote_times = []

    for i in range(NUM_ITERATIONS):
        print(f"Iteration {i + 1}/{NUM_ITERATIONS}...")

        # Test local
        try:
            local_time = test_local_model()
            local_times.append(local_time)
            print(f"  Local: {local_time:.2f}s")
        except Exception as e:
            print(f"  Local: FAILED ({e})")

        # Test remote
        try:
            remote_time = test_remote_model()
            remote_times.append(remote_time)
            print(f"  Remote: {remote_time:.2f}s")
        except Exception as e:
            print(f"  Remote: FAILED ({e})")

        print()

    # Calculate averages
    avg_local = sum(local_times) / len(local_times) if local_times else 0
    avg_remote = sum(remote_times) / len(remote_times) if remote_times else 0

    print("=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(
        f"Local avg response time:  {avg_local:.2f}s ({len(local_times)}/{NUM_ITERATIONS} successful)"
    )
    print(
        f"Remote avg response time: {avg_remote:.2f}s ({len(remote_times)}/{NUM_ITERATIONS} successful)"
    )

    if avg_local > 0 and avg_remote > 0:
        if avg_local < avg_remote:
            print(f"\nLocal is {avg_remote/avg_local:.2f}x faster than Remote")
        else:
            print(f"\nRemote is {avg_local/avg_remote:.2f}x faster than Local")


if __name__ == "__main__":
    run_benchmark()
