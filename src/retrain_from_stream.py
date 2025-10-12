from river import forest, preprocessing, metrics
from sklearn.preprocessing import LabelEncoder
import joblib, pandas as pd, mlflow, time
from pathlib import Path

csv_path = Path("dataset/merged_for_retrain.csv")
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

print(f"Retraining model from {csv_path}")
data = pd.read_csv(csv_path)
X, y = data.drop(columns=["Label"]), data["Label"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
scaler = preprocessing.StandardScaler()
model = forest.ARFClassifier(n_models=10, seed=42)
metric = metrics.Accuracy()

mlflow.set_tracking_uri("https://mlflow-service-n8ab.onrender.com")
mlflow.set_experiment("IDS_ARF_Retrain")

with mlflow.start_run(run_name=f"drift_{int(time.time())}"):
    for i, (xi, yi) in enumerate(zip(X.to_dict(orient="records"), y_encoded)):
        scaler.learn_one(xi)
        xi_scaled = scaler.transform_one(xi)
        if i > 50:
            y_pred = model.predict_one(xi_scaled)
            metric.update(yi, y_pred)
        model.learn_one(xi_scaled, yi)
        if (i + 1) % 10000 == 0:
            mlflow.log_metric("progress_acc", metric.get(), step=i+1)
            print(f"{i+1:,} | Acc: {metric.get():.4f}")

    acc = metric.get()
    print(f"Retrain done | Acc={acc:.4f}")
    mlflow.log_metric("final_acc", acc)

joblib.dump(model, models_dir / "arf_base.pkl")
joblib.dump(scaler, models_dir / "scaler.pkl")
joblib.dump(encoder, models_dir / "label_encoder.pkl")

print("Saved retrained model to models/")
