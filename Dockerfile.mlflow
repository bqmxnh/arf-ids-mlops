# Dockerfile.mlflow — standalone MLflow service
FROM python:3.11-slim

WORKDIR /app

# Cài MLflow
RUN pip install --no-cache-dir mlflow

# Mở cổng 5000
EXPOSE 5000

# Chạy MLflow UI server
CMD ["mlflow", "ui", "--backend-store-uri", "sqlite:///mlflow.db", "--host", "0.0.0.0", "--port", "5000"]
