# random_send.py
import requests
import random
import time

URL = "http://localhost:8000/predict"

# Danh sách feature theo sample của bạn với giá trị mặc định/range để random
FEATURE_RANGES = {
    "Flow Duration": (100, 60000),
    "Total Fwd Packets": (1, 200),
    "Total Backward Packets": (0, 200),
    "Total Length of Fwd Packets": (0.0, 20000.0),
    "Total Length of Bwd Packets": (0.0, 20000.0),
    "Fwd Packet Length Max": (1.0, 1500.0),
    "Fwd Packet Length Min": (1.0, 1500.0),
    "Fwd Packet Length Mean": (1.0, 1500.0),
    "Fwd Packet Length Std": (0.0, 500.0),
    "Bwd Packet Length Max": (1.0, 1500.0),
    "Bwd Packet Length Min": (1.0, 1500.0),
    "Bwd Packet Length Mean": (1.0, 1500.0),
    "Bwd Packet Length Std": (0.0, 500.0),
    "Flow Bytes/s": (0.0, 500000.0),
    "Flow Packets/s": (0.0, 10000.0),
    "Flow IAT Mean": (0.0, 20000.0),
    "Flow IAT Std": (0.0, 20000.0),
    "Flow IAT Max": (0.0, 60000.0),
    "Flow IAT Min": (0.0, 1000.0),
    "Fwd IAT Total": (0.0, 1000.0),
    "Fwd IAT Mean": (0.0, 1000.0),
    "Fwd IAT Std": (0.0, 1000.0),
    "Fwd IAT Max": (0.0, 5000.0),
    "Fwd IAT Min": (0.0, 100.0),
    "Bwd IAT Total": (0.0, 1000.0),
    "Bwd IAT Mean": (0.0, 1000.0),
    "Bwd IAT Std": (0.0, 1000.0),
    "Bwd IAT Max": (0.0, 5000.0),
    "Bwd IAT Min": (0.0, 100.0),
    "Fwd PSH Flags": (0, 1),
    "Bwd PSH Flags": (0, 1),
    "Fwd URG Flags": (0, 1),
    "Bwd URG Flags": (0, 1),
    "Fwd Header Length": (20, 200),
    "Bwd Header Length": (20, 200),
    "Fwd Packets/s": (0.0, 10000.0),
    "Bwd Packets/s": (0.0, 10000.0),
    "Min Packet Length": (1.0, 1500.0),
    "Max Packet Length": (1.0, 1500.0),
    "Packet Length Mean": (1.0, 1500.0),
    "Packet Length Std": (0.0, 500.0),
    "Packet Length Variance": (0.0, 250000.0),
    "FIN Flag Count": (0, 5),
    "SYN Flag Count": (0, 5),
    "RST Flag Count": (0, 5),
    "PSH Flag Count": (0, 5),
    "ACK Flag Count": (0, 5),
    "URG Flag Count": (0, 5),
    "CWE Flag Count": (0, 5),
    "ECE Flag Count": (0, 5),
    "Down/Up Ratio": (0.0, 10.0),
    "Average Packet Size": (1.0, 1500.0),
    "Avg Fwd Segment Size": (1.0, 1500.0),
    "Avg Bwd Segment Size": (1.0, 1500.0),
    "Fwd Avg Bytes/Bulk": (0, 10000),
    "Fwd Avg Packets/Bulk": (0, 100),
    "Fwd Avg Bulk Rate": (0, 10000),
    "Bwd Avg Bytes/Bulk": (0, 10000),
    "Bwd Avg Packets/Bulk": (0, 100),
    "Bwd Avg Bulk Rate": (0, 10000),
    "Subflow Fwd Packets": (0, 50),
    "Subflow Fwd Bytes": (0, 20000),
    "Subflow Bwd Packets": (0, 50),
    "Subflow Bwd Bytes": (0, 20000),
    "Init_Win_bytes_forward": (-1, 65535),
    "Init_Win_bytes_backward": (-1, 65535),
    "act_data_pkt_fwd": (0, 10),
    "min_seg_size_forward": (0, 1500),
    "Active Mean": (0.0, 20000.0),
    "Active Std": (0.0, 20000.0),
    "Active Max": (0.0, 60000.0),
    "Active Min": (0.0, 100.0),
    "Idle Mean": (0.0, 20000.0),
    "Idle Std": (0.0, 20000.0),
    "Idle Max": (0.0, 60000.0),
    "Idle Min": (0.0, 0.0),
}

INT_FEATURES = {
    # features that should be int
    "Total Fwd Packets", "Total Backward Packets",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward"
}

FLOAT_FEATURES = set(FEATURE_RANGES.keys()) - INT_FEATURES

def make_random_feature(name):
    lo, hi = FEATURE_RANGES[name]
    if name in INT_FEATURES:
        # special: allow negative init window (-1)
        if lo < 0 and name.startswith("Init_Win"):
            return random.choice([-1, random.randint(0, int(hi))])
        return random.randint(int(lo), int(hi))
    else:
        # float - pick uniform and round sensibly
        v = random.uniform(float(lo), float(hi))
        # choose decimals based on magnitude
        if hi > 1000:
            return round(v, 6)
        return round(v, 4)

def make_random_sample():
    sample = {}
    for feat in FEATURE_RANGES.keys():
        sample[feat] = make_random_feature(feat)
    return sample

def send_sample(sample):
    payload = {"features": sample}
    try:
        r = requests.post(URL, json=payload, timeout=5.0)
        try:
            return r.json()
        except ValueError:
            return {"status_code": r.status_code, "text": r.text}
    except Exception as e:
        return {"error": str(e)}

def main(n=10, sleep=0.1):
    print(f"Sending {n} random samples to {URL}")
    for i in range(n):
        s = make_random_sample()
        resp = send_sample(s)
        print(f"[{i+1}/{n}] response: {resp}")
        time.sleep(sleep)

if __name__ == "__main__":
    # change n to number of random samples you want
    main(n=10, sleep=0.05)
