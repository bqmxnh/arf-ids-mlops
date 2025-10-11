from river import preprocessing, metrics, forest
from sklearn.preprocessing import LabelEncoder
import joblib, mlflow, pandas as pd
from pathlib import Path
import time

# ============================================================
# 1. Dataset & Model Paths
# ============================================================
csv_path = Path("dataset/balanced_syn_cleaned.csv")
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

print(f"📘 Dataset: {csv_path}")
print(f"📦 Saving model to: {models_dir}")

data = pd.read_csv(csv_path)
X = data.drop(columns=["Label"])
y = data["Label"]

# ============================================================
# 2. Label Encoding
# ============================================================
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# ============================================================
# 3. MLflow Setup (SAFE for GitHub Actions)
# ============================================================
mlruns_dir = Path("mlruns")
mlruns_dir.mkdir(exist_ok=True)
mlflow.set_tracking_uri(f"file://{mlruns_dir.resolve()}")   # ✅ relative path (no permission issue)
mlflow.set_experiment("IDS_ARF_Base")

# ============================================================
# 4. Train Adaptive Random Forest
# ============================================================
scaler = preprocessing.StandardScaler()
model = forest.ARFClassifier(n_models=10, seed=42)
metric = metrics.Accuracy()

with mlflow.start_run(run_name=f"base_{int(time.time())}"):
    mlflow.log_param("dataset_path", str(csv_path))
    mlflow.log_param("algorithm", "AdaptiveRandomForest")
    mlflow.log_param("num_samples", len(data))
    mlflow.log_param("n_models", 10)
    mlflow.log_param("seed", 42)

    print("🚀 Starting base training...")
    for i, (xi, yi) in enumerate(zip(X.to_dict(orient="records"), y_encoded)):
        scaler.learn_one(xi)
        xi_scaled = scaler.transform_one(xi)

        if i > 50:
            y_pred = model.predict_one(xi_scaled)
            metric.update(yi, y_pred)

        model.learn_one(xi_scaled, yi)

        if (i + 1) % 10000 == 0:
            acc = metric.get()
            mlflow.log_metric("progress_acc", acc, step=i+1)
            print(f"Processed {i+1:,} samples | Accuracy: {acc:.4f}")

    final_acc = metric.get()
    mlflow.log_metric("final_acc", final_acc)
    print(f"✅ Final Accuracy: {final_acc:.4f}")

# ============================================================
# 5. Save artifacts (model + preprocessing)
# ============================================================
joblib.dump(model, models_dir / "arf_base.pkl")
joblib.dump(scaler, models_dir / "scaler.pkl")
joblib.dump(encoder, models_dir / "label_encoder.pkl")

mlflow.log_artifact(str(models_dir / "arf_base.pkl"))
mlflow.log_artifact(str(models_dir / "scaler.pkl"))
mlflow.log_artifact(str(models_dir / "label_encoder.pkl"))

print("🎯 Training completed. Artifacts saved successfully.")
