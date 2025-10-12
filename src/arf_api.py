from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import PlainTextResponse
import joblib, time, csv, threading, os
from pathlib import Path
from river import forest, preprocessing, drift
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# ============================================================
# 1️⃣ FastAPI Setup
# ============================================================
app = FastAPI(title="ARF IDS API", version="3.2")

# ============================================================
# 2️⃣ Paths & Model Loading
# ============================================================
MODELS_DIR = Path("models")
DATASET_DIR = Path("dataset")
MODEL_PATH = MODELS_DIR / "arf_base.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
STREAM_LOG = DATASET_DIR / "stream_data.csv"
DRIFT_FLAG_FILE = DATASET_DIR / "drift_trigger.flag"

# Load artifacts
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)

print(f"✅ Loaded model, scaler, and encoder from {MODELS_DIR}")

# Thread lock để tránh race condition khi nhiều request cùng lúc
model_lock = threading.Lock()
update_counter = 0

# ============================================================
# 3️⃣ Drift Detection Setup
# ============================================================
ADWIN = drift.ADWIN(delta=0.002)

def monitor(value: float) -> bool:
    """Kiểm tra drift; nếu phát hiện, tạo flag để retrain."""
    ADWIN.update(value)
    if ADWIN.drift_detected:
        print("⚠️ Drift detected!")
        os.makedirs(DATASET_DIR, exist_ok=True)
        with open(DRIFT_FLAG_FILE, "w") as f:
            f.write(f"Drift at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        return True
    return False

# ============================================================
# 4️⃣ Prometheus Metrics
# ============================================================
PREDICTION_COUNT = Counter(
    "prediction_requests_total", "Total number of prediction requests"
)
DRIFT_COUNT = Counter(
    "drift_detected_total", "Total number of drift detections"
)
LATENCY = Histogram(
    "prediction_latency_ms", "Prediction latency (milliseconds)"
)
UPDATE_GAUGE = Gauge(
    "model_updates_total", "Number of incremental model updates"
)

@app.get("/metrics")
def metrics():
    """Expose Prometheus metrics."""
    return PlainTextResponse(generate_latest(), media_type="text/plain")

# ============================================================
# 5️⃣ Schema for Incoming Requests
# ============================================================
class Flow(BaseModel):
    features: dict
    label: str | None = None  # optional ground truth

# ============================================================
# 6️⃣ Predict + Online Learn Endpoint
# ============================================================
@app.post("/predict")
def predict(flow: Flow):
    global update_counter
    start_time = time.time()

    try:
        # Chuẩn hoá dữ liệu đầu vào
        x_scaled = scaler.transform_one(flow.features)

        with model_lock:
            # Dự đoán
            y_pred = model.predict_one(x_scaled)
            y_label = encoder.inverse_transform([int(y_pred)])[0]

            # Ghi nhận Prometheus counter
            PREDICTION_COUNT.inc()

            # Kiểm tra drift
            drift_flag = monitor(float(y_pred))
            if drift_flag:
                DRIFT_COUNT.inc()

            # Lấy ground truth nếu có, nếu không thì dùng pseudo-label
            if flow.label:
                y_true = encoder.transform([flow.label])[0]
                used_label = flow.label
                is_pseudo = False
            else:
                y_true = int(y_pred)
                used_label = y_label
                is_pseudo = True

            # Học dần
            model.learn_one(x_scaled, y_true)
            update_counter += 1
            UPDATE_GAUGE.set(update_counter)

            # Lưu model định kỳ
            if update_counter % 100 == 0:
                joblib.dump(model, MODEL_PATH)

            # Ghi log stream
            STREAM_LOG.parent.mkdir(exist_ok=True)
            with open(STREAM_LOG, "a", newline="") as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow(list(flow.features.keys()) + ["Label", "is_pseudo"])
                writer.writerow(list(flow.features.values()) + [used_label, is_pseudo])

        latency_ms = (time.time() - start_time) * 1000
        LATENCY.observe(latency_ms)

        return {
            "prediction": y_label,
            "used_label": used_label,
            "is_pseudo": is_pseudo,
            "drift_detected": drift_flag,
            "latency_ms": round(latency_ms, 3),
            "updates": update_counter
        }

    except Exception as e:
        return {"error": str(e)}

# ============================================================
# 7️⃣ Health Check Endpoint
# ============================================================
@app.get("/")
def root():
    return {
        "status": "running",
        "model": "Adaptive Random Forest IDS",
        "version": "3.2",
        "updates": update_counter,
    }
