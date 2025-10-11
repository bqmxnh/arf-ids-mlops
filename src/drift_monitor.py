from river import drift
import os, time

# file flag để GitHub Actions phát hiện
DRIFT_FLAG_FILE = "dataset/drift_trigger.flag"
ADWIN = drift.ADWIN(delta=0.002)  # độ nhạy drift

def monitor(value: float):
    """
    Kiểm tra concept drift bằng ADWIN.
    Nếu phát hiện drift → tạo file flag để retrain tự động.
    """
    ADWIN.update(value)
    if ADWIN.drift_detected:
        print("⚠️ Drift detected!")
        os.makedirs("dataset", exist_ok=True)
        with open(DRIFT_FLAG_FILE, "w") as f:
            f.write(f"Drift at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        return True
    return False
