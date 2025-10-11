import requests
import random
import time

# ============================================================
# 1. API Configuration
# ============================================================
API_URL = "http://localhost:8000/predict"

# ============================================================
# 2. Random Sample Generator
# ============================================================
def generate_random_flow():
    """Generate a random network flow sample for testing the IDS API."""
    return {
        "features": {
            "Flow Duration": random.randint(1000, 200000),            # microseconds
            "Total Fwd Packets": random.randint(1, 20),
            "Total Backward Packets": random.randint(1, 20),
            "Total Length of Fwd Packets": random.randint(100, 1500),
            "Total Length of Bwd Packets": random.randint(100, 1500),
            "Flow Bytes/s": random.uniform(10, 10000),
            "Flow Packets/s": random.uniform(1, 100),
            "Protocol": random.choice([6, 17, 1])  # TCP, UDP, ICMP
        }
    }

# ============================================================
# 3. Send Random Requests
# ============================================================
if __name__ == "__main__":
    for i in range(5):
        sample = generate_random_flow()
        try:
            response = requests.post(API_URL, json=sample, timeout=5)
            print(f"Request {i+1}")
            print("Status:", response.status_code)
            print("Response:", response.json())
            print("-" * 40)
        except Exception as e:
            print(f"Error sending request {i+1}: {e}")
        time.sleep(1)
