from river import preprocessing, metrics, forest
from sklearn.preprocessing import LabelEncoder
import joblib, mlflow, pandas as pd
from pathlib import Path
import time, sys

# ============================================================
# 1. Path setup
# ============================================================
csv_path = Path("dataset/balanced_syn_cleaned.csv")
models_dir = Path("models")
mlruns_dir = Path("mlruns")

models_dir.mkdir(parents=True, exist_ok=True)
mlruns_dir.mkdir(parents=True, exist_ok=True)

print(f"Dataset: {csv_path}")
print(f"Saving model to: {models_dir}")
print(f"MLflow logs will be saved to: {mlruns_dir.resolve()}")

# ============================================================
# 2. Load dataset
# ============================================================
try:
    data = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Error: Dataset not found at {csv_path}")
    sys.exit(1)

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
# 4. MLflow setup
# ============================================================
mlflow.set_tracking_uri("http://mlflow:5000")  # hoặc file:///mlruns nếu chạy local
mlflow.set_experiment("IDS_ARF_Base")

# ============================================================
# 5. Model & Metrics initialization
# ============================================================
scaler = preprocessing.StandardScaler()
model = forest.ARFClassifier(n_models=10, seed=42)

metric_acc = metrics.Accuracy()
metric_prec = metrics.Precision()
metric_rec = metrics.Recall()
metric_f1 = metrics.F1()

# ============================================================
# 6. Training loop (Prequential Test-Then-Train)
# ============================================================
with mlflow.start_run(run_name=f"base_{int(time.time())}"):
    mlflow.log_param("dataset_path", str(csv_path))
    mlflow.log_param("algorithm", "AdaptiveRandomForest")
    mlflow.log_param("num_samples", len(data))
    mlflow.log_param("n_models", 10)
    mlflow.log_param("seed", 42)

    print("Starting base training...")

    for i, (xi, yi) in enumerate(zip(X.to_dict(orient="records"), y_encoded), start=1):
        # Scale features
        scaler.learn_one(xi)
        xi_scaled = scaler.transform_one(xi)

        # Predict after warm-up
        if i > 50:
            y_pred = model.predict_one(xi_scaled)
            if y_pred is not None:
                metric_acc.update(yi, y_pred)
                metric_prec.update(yi, y_pred)
                metric_rec.update(yi, y_pred)
                metric_f1.update(yi, y_pred)

        # Incremental learning
        model.learn_one(xi_scaled, yi)

        # Log intermediate progress
        if i % 10000 == 0:
            acc = metric_acc.get()
            f1 = metric_f1.get()
            mlflow.log_metric("progress_acc", acc, step=i)
            mlflow.log_metric("progress_f1", f1, step=i)
            print(f"Processed {i:,} samples | Accuracy: {acc:.4f} | F1: {f1:.4f}")

    # Final metrics
    final_acc = metric_acc.get()
    final_prec = metric_prec.get()
    final_rec = metric_rec.get()
    final_f1 = metric_f1.get()

    mlflow.log_metric("final_acc", final_acc)
    mlflow.log_metric("final_precision", final_prec)
    mlflow.log_metric("final_recall", final_rec)
    mlflow.log_metric("final_f1", final_f1)

    print(f"Final Accuracy: {final_acc:.4f}")
    print(f"Final Precision: {final_prec:.4f}")
    print(f"Final Recall: {final_rec:.4f}")
    print(f"Final F1-score: {final_f1:.4f}")

# ============================================================
# 7. Save artifacts (model, scaler, encoder)
# ============================================================
model_path = models_dir / "arf_base.pkl"
scaler_path = models_dir / "scaler.pkl"
encoder_path = models_dir / "label_encoder.pkl"

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(encoder, encoder_path)

mlflow.log_artifact(str(model_path))
mlflow.log_artifact(str(scaler_path))
mlflow.log_artifact(str(encoder_path))

print("Training completed. Artifacts saved successfully.")
