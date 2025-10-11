from fastapi import FastAPI
from pydantic import BaseModel
import joblib, time, mlflow, csv
from river import forest, preprocessing
from pathlib import Path
from datetime import datetime

# ============================================================
# 1. Setup FastAPI
# ============================================================
app = FastAPI(title="ARF IDS API", version="1.1")

# ============================================================
# 2. Load Model & Preprocessors
# ============================================================
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "arf_base.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
STREAM_LOG = Path("dataset/stream_data.csv")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)

print(f"✅ Model & preprocessing loaded from {MODELS_DIR}")

# ============================================================
# 3. Schema
# ============================================================
class Flow(BaseModel):
    features: dict
    label: str | None = None

update_counter = 0

# ============================================================
# 4. Predict + Learn Endpoint
# ============================================================
@app.post("/predict")
def predict(flow: Flow):
    global update_counter
    start_time = time.time()

    try:
        # Scale input
        x_scaled = scaler.transform_one(flow.features)

        # Predict
        y_pred = model.predict_one(x_scaled)
        y_label = encoder.inverse_transform([int(y_pred)])[0]

        # Learn incrementally if label available
        if flow.label:
            y_true = encoder.transform([flow.label])[0]
            model.learn_one(x_scaled, int(y_true))
            update_counter += 1

            # Save model every 100 updates
            if update_counter % 100 == 0:
                joblib.dump(model, MODEL_PATH)

            # Append to stream dataset
            STREAM_LOG.parent.mkdir(exist_ok=True)
            with open(STREAM_LOG, "a", newline="") as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow(list(flow.features.keys()) + ["Label"])
                writer.writerow(list(flow.features.values()) + [flow.label])

        latency = (time.time() - start_time) * 1000
        return {
            "prediction": y_label,
            "updated": bool(flow.label),
            "latency_ms": round(latency, 3)
        }

    except Exception as e:
        return {"error": str(e)}

# ============================================================
# 5. Health Endpoint
# ============================================================
@app.get("/")
def root():
    return {
        "status": "running",
        "model": "ARF IDS",
        "version": "1.1",
        "updates": update_counter
    }
