from river import forest, preprocessing, metrics
from sklearn.preprocessing import LabelEncoder
import joblib, pandas as pd, mlflow, time, sys, os
from pathlib import Path

# ============================================================
# 1. Paths
# ============================================================
csv_path = Path("dataset/merged_for_retrain.csv")
models_dir = Path("models")
models_dir.mkdir(parents=True, exist_ok=True)

print(f"[Retrain] Loading dataset from {csv_path}")

if not csv_path.exists():
    print("No merged dataset found. Exiting retrain.")
    sys.exit(0)

# ============================================================
# 2. Load dataset
# ============================================================
data = pd.read_csv(csv_path)
if "Label" not in data.columns:
    print("Missing 'Label' column.")
    sys.exit(1)

if len(data) < 500:
    print(f"Too few samples ({len(data)}). Skipping retrain to avoid overfit.")
    sys.exit(0)

X = data.drop(columns=["Label"])
y = data["Label"]

# ============================================================
# 3. Encode labels
# ============================================================
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# ============================================================
# 4. Load or init model/scaler
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

scaler = joblib.load(scaler_path) if scaler_path.exists() else preprocessing.StandardScaler()

# ============================================================
# 5. Setup metrics
# ============================================================
metric_acc = metrics.Accuracy()
metric_prec = metrics.Precision()
metric_rec = metrics.Recall()
metric_f1 = metrics.F1()

# ============================================================
# 6. Setup MLflow
# ============================================================
try:
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("IDS_ARF_Retrain")
    mlflow_enabled = True
except Exception:
    print("MLflow unavailable — running offline.")
    mlflow_enabled = False

# ============================================================
# 7. Prequential retraining loop
# ============================================================
print("Starting retraining (test-then-train)...")

if mlflow_enabled:
    run_ctx = mlflow.start_run(run_name=f"retrain_{int(time.time())}")
else:
    from contextlib import nullcontext
    run_ctx = nullcontext()

with run_ctx:
    if mlflow_enabled:
        mlflow.log_params({
            "algorithm": "AdaptiveRandomForest",
            "n_models": 10,
            "dataset_size": len(data),
            "retrain_type": "drift_based_incremental"
        })

    for i, (xi, yi) in enumerate(zip(X.to_dict(orient="records"), y_encoded), start=1):
        scaler.learn_one(xi)
        xi_scaled = scaler.transform_one(xi)

        if i > 50:
            y_pred = model.predict_one(xi_scaled)
            metric_acc.update(yi, y_pred)
            metric_prec.update(yi, y_pred)
            metric_rec.update(yi, y_pred)
            metric_f1.update(yi, y_pred)

        model.learn_one(xi_scaled, yi)

        if i % 10000 == 0:
            acc, f1 = metric_acc.get(), metric_f1.get()
            print(f"📈 {i:,} samples processed | Acc={acc:.4f} | F1={f1:.4f}")
            if mlflow_enabled:
                mlflow.log_metric("progress_acc", acc, step=i)
                mlflow.log_metric("progress_f1", f1, step=i)

    final_metrics = {
        "acc": metric_acc.get(),
        "prec": metric_prec.get(),
        "rec": metric_rec.get(),
        "f1": metric_f1.get()
    }

    print(f"Retrain complete | Acc={final_metrics['acc']:.4f} | F1={final_metrics['f1']:.4f}")
    if mlflow_enabled:
        mlflow.log_metrics({f"final_{k}": v for k, v in final_metrics.items()})

# ============================================================
# 8. Save artifacts
# ============================================================
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(encoder, encoder_path)

if mlflow_enabled:
    mlflow.log_artifact(str(model_path))
    mlflow.log_artifact(str(scaler_path))
    mlflow.log_artifact(str(encoder_path))

print("Saved updated ARF model and artifacts successfully.")
