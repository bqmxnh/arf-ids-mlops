# ARF-IDS-MLOps

Hệ thống Intrusion Detection System (IDS) sử dụng **Adaptive Random Forest (ARF)** kết hợp **MLOps pipeline**: học online, phát hiện drift, retrain tự động và deploy.

---

## 🧩 Mục tiêu dự án

- Phát triển mô hình ARF để phân loại traffic mạng (normal / attack) theo từng flow.
- Học dần (online learning) khi nhận dữ liệu mới, vừa dự đoán vừa cập nhật mô hình.
- Phát hiện concept drift (khi phân phối dữ liệu đổi) để tự động retrain.
- Logging và tracking qua MLflow hoặc hệ thống monitoring (Prometheus / Grafana).
- Triển khai API để các node client gửi dữ liệu và nhận kết quả realtime.

---

## 📂 Cấu trúc thư mục
<img width="500" height="359" alt="image" src="https://github.com/user-attachments/assets/56458f68-252f-4b44-b31a-80835f16a4d3" />

---

## 🛠️ Cài đặt & chạy local

### 1. Clone repo & chuyển vào thư mục
```bash
git clone https://github.com/bqmxnh/arf-ids-mlops.git
cd arf-ids-mlops
```

### 2. Cài dependencies
```bash
python src/arf_train.py
```

### 3. Huấn luyện mô hình base lần đầu
```bash
pip install -r requirements.txt
```

### 4. Chạy API FastAPI
```bash
uvicorn src.api_arf:app --host 0.0.0.0 --port 8000
```

### 5. Gửi thử một request
```bash
curl http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": { ... } }'
```
