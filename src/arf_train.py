from river import preprocessing, metrics, forest
from sklearn.preprocessing import LabelEncoder
import joblib, mlflow, pandas as pd
from pathlib import Path
import time, sys

# Paths setup
csv_path = Path("dataset/balanced_syn_cleaned.csv")
models_dir = Path("models")
mlruns_dir = Path("mlruns")

# Ensure directories exist
models_dir.mkdir(parents=True, exist_ok=True)
mlruns_dir.mkdir(parents=True, exist_ok=True)

print(f"Dataset: {csv_path}")
print(f"Saving model to: {models_dir}")
print(f"MLflow logs will be saved to: {mlruns_dir.resolve()}")

# Load dataset
try:
    data = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Dataset not found: {csv_path}")
    sys.exit(1)

if "Label" not in data.columns:
    print("Dataset missing 'Label' column!")
    sys.exit(1)

X = data.drop(columns=["Label"])
y = data["Label"]

# Label Encoding
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# MLflow setup (safe for GitHub Actions / Docker)
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("IDS_ARF_Base")

# Model initialization
scaler = preprocessing.StandardScaler()
model = forest.ARFClassifier(n_models=10, seed=42)
metric = metrics.Accuracy()

# Training loop
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

        # Predict (skip first few warm-up samples)
        if i > 50:
            y_pred = model.predict_one(xi_scaled)
            metric.update(yi, y_pred)

        # Learn incrementally
        model.learn_one(xi_scaled, yi)

        # Log intermediate metrics
        if i % 10000 == 0:
            acc = metric.get()
            mlflow.log_metric("progress_acc", acc, step=i)
            print(f"Processed {i:,} samples | Accuracy: {acc:.4f}")

    # Final metric
    final_acc = metric.get()
    mlflow.log_metric("final_acc", final_acc)
    print(f"Final Accuracy: {final_acc:.4f}")

# Save artifacts (model, scaler, encoder)
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
