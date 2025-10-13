from river import forest, preprocessing, metrics
from sklearn.preprocessing import LabelEncoder
import joblib, pandas as pd, mlflow, time
from pathlib import Path
import sys

# ============================================================
# 1. Paths setup
# ============================================================
csv_path = Path("dataset/merged_for_retrain.csv")
models_dir = Path("models")
models_dir.mkdir(parents=True, exist_ok=True)

print(f"Retraining model from: {csv_path}")

# ============================================================
# 2. Load dataset
# ============================================================
if not csv_path.exists():
    print(f"Error: Dataset not found at {csv_path}")
    sys.exit(1)

data = pd.read_csv(csv_path)
if "Label" not in data.columns:
    print("Error: Dataset missing 'Label' column.")
    sys.exit(1)

X = data.drop(columns=["Label"])
y = data["Label"]

# ============================================================
# 3. Label Encoding
# ============================================================
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# ============================================================
# 4. Load old model & scaler if available
# ============================================================
model_path = models_dir / "arf_base.pkl"
scaler_path = models_dir / "scaler.pkl"
encoder_path = models_dir / "label_encoder.pkl"

if model_path.exists():
    model = joblib.load(model_path)
    print(f"Loaded previous model from {model_path}")
else:
    print("No existing model found. Initializing new ARF model.")
    model = forest.ARFClassifier(n_models=10, seed=42)

if scaler_path.exists():
    scaler = joblib.load(scaler_path)
    print(f"Loaded scaler from {scaler_path}")
else:
    scaler = preprocessing.StandardScaler()

metric_acc = metrics.Accuracy()

# ============================================================
# 5. MLflow setup
# ============================================================
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("IDS_ARF_Retrain")

# ============================================================
# 6. Prequential retraining loop
# ============================================================
with mlflow.start_run(run_name=f"drift_{int(time.time())}"):
    mlflow.log_param("dataset_path", str(csv_path))
    mlflow.log_param("algorithm", "AdaptiveRandomForest")
    mlflow.log_param("n_models", 10)
    mlflow.log_param("retrain_type", "drift_based_incremental")

    print("Starting retraining process...")

    for i, (xi, yi) in enumerate(zip(X.to_dict(orient="records"), y_encoded), start=1):
        # Scale and predict
        scaler.learn_one(xi)
        xi_scaled = scaler.transform_one(xi)

        if i > 50:
            y_pred = model.predict_one(xi_scaled)
            metric_acc.update(yi, y_pred)

        model.learn_one(xi_scaled, yi)

        # Log progress every 10k samples
        if i % 10000 == 0:
            acc = metric_acc.get()
            mlflow.log_metric("progress_acc", acc, step=i)
            print(f"{i:,} samples processed | Accuracy: {acc:.4f}")

    # Final accuracy
    final_acc = metric_acc.get()
    print(f"Retraining completed | Final Accuracy: {final_acc:.4f}")
    mlflow.log_metric("final_acc", final_acc)

# ============================================================
# 7. Save updated artifacts
# ============================================================
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(encoder, encoder_path)

mlflow.log_artifact(str(model_path))
mlflow.log_artifact(str(scaler_path))
mlflow.log_artifact(str(encoder_path))

print("Retrained model and artifacts saved successfully.")
