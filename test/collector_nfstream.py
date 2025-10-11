import psutil
import nfstream
import requests
import json
from datetime import datetime

def detect_active_interface():
    # Lọc các interface có IP đang hoạt động (tránh ảo, loopback)
    candidates = []
    addrs = psutil.net_if_addrs()
    for name, infos in addrs.items():
        for info in infos:
            if info.family.name == "AF_INET" and not info.address.startswith("127."):
                candidates.append(name)
    if not candidates:
        raise RuntimeError("Không tìm thấy interface khả dụng.")
    print("Detected active interfaces:", candidates)
    return candidates[0]

INTERFACE = detect_active_interface()
API_URL = "http://localhost:8000/predict"

def extract_features(flow):
    duration_s = flow.bidirectional_duration_ms / 1000
    features = {
        "flow_duration": duration_s,
        "fwd_pkt_len_mean": flow.src2dst_mean_ps,
        "bwd_pkt_len_mean": flow.dst2src_mean_ps,
        "fwd_iat_std": flow.src2dst_stddev_piat_ms,
        "bwd_iat_std": flow.dst2src_stddev_piat_ms,
        "protocol": flow.protocol,
        "fwd_packets": flow.src2dst_packets,
        "bwd_packets": flow.dst2src_packets,
        "flow_iat_mean": flow.bidirectional_mean_piat_ms,
        "flow_bytes_per_s": (flow.bidirectional_bytes / duration_s) if duration_s > 0 else 0
    }
    return features

class CollectorPlugin(nfstream.NFPlugin):
    def on_update(self, flow):
        features = extract_features(flow)
        payload = {"features": features}

        try:
            response = requests.post(API_URL, json=payload, timeout=3)
            if response.ok:
                result = response.json()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"{flow.src_ip}:{flow.src_port} -> {flow.dst_ip}:{flow.dst_port} | "
                      f"Predicted: {result['prediction']}")
            else:
                print(f"API Error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"Request failed: {e}")

if __name__ == "__main__":
    print(f"Starting NFStream collector on interface: {INTERFACE}")
    streamer = nfstream.NFStreamer(
        source=INTERFACE,
        statistical_analysis=True,
        accounting_mode=1,
        active_timeout=10,
        idle_timeout=5
    )
    plugin = CollectorPlugin()
    for flow in streamer:
        plugin.on_update(flow)
