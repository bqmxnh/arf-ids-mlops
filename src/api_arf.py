from fastapi import FastAPI
from pydantic import BaseModel
import joblib, time, csv, threading
from river import forest, preprocessing
from pathlib import Path
from drift_monitor import monitor

# ============================================================
# 1️⃣ FastAPI Setup
# ============================================================
app = FastAPI(title="ARF IDS API", version="3.0")

# ============================================================
# 2️⃣ Load model & preprocessors
# ============================================================
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "arf_base.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
STREAM_LOG = Path("dataset/stream_data.csv")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)

print(f"✅ Loaded model & scaler from {MODELS_DIR}")

# ============================================================
# 3️⃣ Schema
# ============================================================
class Flow(BaseModel):
    features: dict
    label: str | None = None

update_counter = 0
model_lock = threading.Lock()  # tránh race condition khi nhiều node gửi cùng lúc

# ============================================================
# 4️⃣ Predict + Online Learn
# ============================================================
@app.post("/predict")
def predict(flow: Flow):
    global update_counter
    start = time.time()

    try:
        x_scaled = scaler.transform_one(flow.features)

        with model_lock:
            y_pred = model.predict_one(x_scaled)
            y_label = encoder.inverse_transform([int(y_pred)])[0]

            # 🧩 Check drift
            drift_flag = monitor(float(y_pred))

            if flow.label:
                y_true = encoder.transform([flow.label])[0]
                used_label = flow.label
                is_pseudo = False
            else:
                y_true = int(y_pred)
                used_label = y_label
                is_pseudo = True

            model.learn_one(x_scaled, y_true)
            update_counter += 1

            # lưu model mỗi 100 mẫu
            if update_counter % 100 == 0:
                joblib.dump(model, MODEL_PATH)

            # log stream
            STREAM_LOG.parent.mkdir(exist_ok=True)
            with open(STREAM_LOG, "a", newline="") as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow(list(flow.features.keys()) + ["Label", "is_pseudo"])
                writer.writerow(list(flow.features.values()) + [used_label, is_pseudo])

        latency = (time.time() - start) * 1000
        return {
            "prediction": y_label,
            "used_label": used_label,
            "is_pseudo": is_pseudo,
            "drift_detected": drift_flag,
            "latency_ms": round(latency, 3),
            "updates": update_counter
        }

    except Exception as e:
        return {"error": str(e)}

# ============================================================
# 5️⃣ Health Check
# ============================================================
@app.get("/")
def root():
    return {
        "status": "running",
        "model": "ARF IDS",
        "version": "3.0",
        "updates": update_counter
    }
