from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import PlainTextResponse
import joblib, time, csv, threading, os
from pathlib import Path
from river import forest, preprocessing, drift
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# ============================================================
# FastAPI Setup
# ============================================================
app = FastAPI(title="ARF IDS API", version="3.5")

# ============================================================
# Paths & Model Loading
# ============================================================
MODELS_DIR = Path("models")
DATASET_DIR = Path("dataset")
MODEL_PATH = MODELS_DIR / "arf_base.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
STREAM_LOG = DATASET_DIR / "stream_data.csv"
UNLABELED_LOG = DATASET_DIR / "unlabeled_log.csv"
DRIFT_FLAG_FILE = DATASET_DIR / "drift_trigger.flag"

# Load artifacts
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)
print(f"Loaded model, scaler, and encoder from {MODELS_DIR}")

model_lock = threading.Lock()
update_counter = 0

# ============================================================
# Drift Detection Setup
# ============================================================
ADWIN = drift.ADWIN(delta=0.002)

def monitor(value: float) -> bool:
    """Theo dõi drift dựa trên confidence trung bình."""
    ADWIN.update(value)
    if ADWIN.drift_detected:
        print("Drift detected!")
        os.makedirs(DATASET_DIR, exist_ok=True)
        with open(DRIFT_FLAG_FILE, "w") as f:
            f.write(f"Drift at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        return True
    return False

# ============================================================
# Prometheus Metrics
# ============================================================
PREDICTION_COUNT = Counter("prediction_requests_total", "Total number of prediction requests")
DRIFT_COUNT = Counter("drift_detected_total", "Total number of drift detections")
LATENCY = Histogram("prediction_latency_ms", "Prediction latency (milliseconds)")
UPDATE_GAUGE = Gauge("model_updates_total", "Number of incremental model updates")

@app.get("/metrics")
def metrics():
    """Expose Prometheus metrics."""
    return PlainTextResponse(generate_latest(), media_type="text/plain")

# ============================================================
# Schema for Incoming Requests
# ============================================================
class Flow(BaseModel):
    features: dict

# ============================================================
# Predict + Online Learn (Pseudo-label only)
# ============================================================
@app.post("/predict")
def predict(flow: Flow):
    global update_counter
    start_time = time.time()

    try:
        x_scaled = scaler.transform_one(flow.features)

        with model_lock:
            # --- Dự đoán ---
            proba = model.predict_proba_one(x_scaled)
            if not proba:
                return {"error": "Model returned empty probabilities."}

            y_pred = max(proba, key=proba.get)
            confidence = proba[y_pred]
            y_label = encoder.inverse_transform([int(y_pred)])[0]

            # --- Ghi nhận metrics ---
            PREDICTION_COUNT.inc()
            drift_flag = monitor(confidence)
            if drift_flag:
                DRIFT_COUNT.inc()

            # --- Học nếu confidence đủ cao ---
            if confidence > 0.95:
                y_true = int(y_pred)  # pseudo-label
                model.learn_one(x_scaled, y_true)
                update_counter += 1
                UPDATE_GAUGE.set(update_counter)

                # Log mẫu đã học (chỉ feature + label)
                STREAM_LOG.parent.mkdir(exist_ok=True)
                with open(STREAM_LOG, "a", newline="") as f:
                    writer = csv.writer(f)
                    if f.tell() == 0:
                        writer.writerow(list(flow.features.keys()) + ["Label"])
                    writer.writerow(list(flow.features.values()) + [y_label])
            else:
                # Log mẫu không đủ tin cậy (chỉ feature)
                UNLABELED_LOG.parent.mkdir(exist_ok=True)
                with open(UNLABELED_LOG, "a", newline="") as f:
                    writer = csv.writer(f)
                    if f.tell() == 0:
                        writer.writerow(list(flow.features.keys()))
                    writer.writerow(list(flow.features.values()))

            # --- Định kỳ lưu model ---
            if update_counter > 0 and update_counter % 100 == 0:
                joblib.dump(model, MODEL_PATH)

        # --- Ghi latency ---
        latency_ms = (time.time() - start_time) * 1000
        LATENCY.observe(latency_ms)

        return {
            "prediction": y_label,
            "confidence": round(confidence, 4),
            "drift_detected": drift_flag,
            "learned": confidence > 0.95,
            "latency_ms": round(latency_ms, 3),
            "updates": update_counter
        }

    except Exception as e:
        return {"error": str(e)}

# ============================================================
# Health Check
# ============================================================
@app.get("/")
def root():
    return {
        "status": "running",
        "model": "Adaptive Random Forest IDS",
        "version": "3.5",
        "updates": update_counter,
        "pseudo_label_only": True
    }
