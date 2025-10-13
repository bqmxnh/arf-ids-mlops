# ARF-IDS-MLOps

Hệ thống Intrusion Detection System (IDS) sử dụng **Adaptive Random Forest (ARF)** kết hợp **MLOps pipeline**: học online, phát hiện drift, retrain tự động và deploy.

## Mục tiêu dự án

- Phát triển mô hình ARF để phân loại traffic mạng (normal / attack) theo từng flow.
- Học dần (online learning) khi nhận dữ liệu mới, vừa dự đoán vừa cập nhật mô hình.
- Phát hiện concept drift (khi phân phối dữ liệu đổi) để tự động retrain.
- Logging và tracking qua MLflow hoặc hệ thống monitoring (Prometheus / Grafana).
- Triển khai API để các node client gửi dữ liệu và nhận kết quả realtime.


## Cấu trúc thư mục
<img width="500" height="359" alt="image" src="https://github.com/user-attachments/assets/56458f68-252f-4b44-b31a-80835f16a4d3" />


## Chuẩn bị môi trường
### AWS EC2

- **Hệ điều hành:** Ubuntu 22.04 LTS  
- **Mở các port sau trên Security Group:**

| **Cổng** | **Dịch vụ**   | **Mục đích**                     |
|:--------:|:--------------|:---------------------------------|
| `22`     | SSH           | Kết nối điều khiển từ xa         |
| `80`     | FastAPI       | Cung cấp API dự đoán IDS         |
| `5000`   | MLflow        | Giao diện theo dõi thí nghiệm ML |
| `9090`   | Prometheus    | Thu thập và giám sát metrics     |
| `3000`   | Grafana       | Dashboard trực quan hóa giám sát |

### Yêu cầu phần mềm
- **Cài đặt Docker, Docker Compose và Git:**
```bash
sudo apt update && sudo apt install -y docker.io docker-compose git
sudo systemctl enable docker
sudo usermod -aG docker ubuntu
exit
# đăng nhập lại SSH
```

## Clone dự án & cấu trúc chính
```bash
git clone https://github.com/bqmxnh/arf-ids-mlops.git
cd arf-ids-mlops
```

## Build & chạy hệ thống
```bash
docker compose build
docker compose up -d
```
- **Kiểm tra container:**
```bash
docker ps
```

| **Dịch vụ**   | **Cổng** | **Mô tả**                               |
|:--------------|:--------:|:----------------------------------------|
| `arf-api`     | `80`     | API phục vụ dự đoán & học trực tuyến    |
| `mlflow`      | `5000`   | Theo dõi mô hình, thí nghiệm            |
| `prometheus`  | `9090`   | Thu thập số liệu hệ thống               |
| `grafana`     | `3000`   | Giám sát trực quan                      |

## Truy cập dịch vụ

| **Thành phần** | **URL** | **Chức năng** |
|:----------------|:--------|:--------------|
| **FastAPI**     | `http://<EC2-IP>/` | Cung cấp endpoint `/predict`, `/metrics`, và health check cho hệ thống IDS |
| **MLflow**      | `http://<EC2-IP>:5000` | Giao diện quản lý, theo dõi thí nghiệm và version của mô hình học máy |
| **Prometheus**  | `http://<EC2-IP>:9090` | Lưu trữ và truy vấn các raw metrics thu thập từ API & hệ thống |
| **Grafana**     | `http://<EC2-IP>:3000` | Hiển thị dashboard trực quan hóa các chỉ số drift, latency và tình trạng mô hình |

## Cấu hình Grafana
### Truy cập `http://<EC2-IP>:3000`  
**Tài khoản mặc định:**  
- Username: `admin`  
- Password: `admin`  
### Thêm data source:
- **Type:** Prometheus
- **URL:** `http://prometheus:9090`
### Import dashboard:
```bash
+ → Import → Upload → dashboards/drift_monitor.json
```
Sau khi import dashboard, bạn sẽ thấy các biểu đồ hiển thị:

| **Chỉ số** | **Ý nghĩa** |
|:------------|:-------------|
| **Prediction Rate** | Tốc độ xử lý và dự đoán luồng dữ liệu |
| **Drift Ratio** | Tỷ lệ sai lệch dữ liệu so với phân phối gốc |
| **Retrain Count** | Số lần mô hình được huấn luyện lại |
| **Average Latency** | Độ trễ trung bình của hệ thống phản hồi |

## Tự động Huấn luyện & Triển khai qua GitHub Actions

Hệ thống sử dụng 2 workflow GitHub Actions để tự động hóa toàn bộ chu trình phát hiện drift → retrain → deploy lại mô hình.

### Workflow: Drift-based Retrain & EC2 Deploy

Tệp: `.github/workflows/drift_retrain.yml`

Mục tiêu:  
Tự động kiểm tra drift trong dữ liệu streaming mỗi 12 giờ, huấn luyện lại mô hình nếu phát hiện thay đổi, và triển khai mô hình mới lên EC2.

#### Quy trình chi tiết:

| Bước | Hành động | Mô tả |
|:----:|:-----------|:------|
| 1 | Kiểm tra lịch | Workflow chạy định kỳ mỗi 12 giờ (`cron: 0 */12 * * *`) hoặc thủ công qua GitHub UI. |
| 2 | Thiết lập môi trường Python | Dùng `actions/setup-python@v5` để cài Python 3.11 và các thư viện cần thiết từ `requirements.txt`. |
| 3 | Kiểm tra file drift flag | Nếu tồn tại `dataset/drift_trigger.flag` → phát hiện drift (`drift_found=true`). |
| 4 | Gộp & Huấn luyện lại mô hình | Chạy `merge_streaming.py` để hợp nhất dữ liệu mới, sau đó `retrain_from_stream.py` để cập nhật mô hình. |
| 5 | Commit & Push mô hình mới | GitHub Actions tự động commit model mới trong thư mục `models/` lên branch `main`. |
| 6 | Triển khai lại trên EC2 | Sử dụng `appleboy/ssh-action` để SSH vào EC2, pull code mới và rebuild toàn bộ Docker stack. |

Kết quả: Mỗi khi drift được phát hiện, pipeline sẽ tự động huấn luyện lại mô hình, cập nhật trên GitHub, và triển khai ngay lên EC2.

---

### Workflow: Auto Deploy to EC2

Tệp: `.github/workflows/auto_deploy.yml`

Mục tiêu:  
Tự động triển khai lại ứng dụng mỗi khi có commit mới lên nhánh `main`, hoặc khi bạn chạy thủ công từ tab Actions.

#### Quy trình chi tiết:

| Bước | Hành động | Mô tả |
|:----:|:-----------|:------|
| 1 | Kích hoạt workflow | Khi có push lên `main` hoặc chạy thủ công. |
| 2 | SSH vào EC2 | Dùng `appleboy/ssh-action` để đăng nhập vào instance qua khóa SSH bí mật (`EC2_SSH_KEY`). |
| 3 | Cập nhật mã nguồn | Pull code mới từ GitHub về thư mục `~/arf-ids-mlops`. Nếu chưa có thì tự clone. |
| 4 | Tái khởi động dịch vụ Docker | Dừng toàn bộ container (`docker-compose down`), sau đó rebuild và khởi động lại (`docker-compose up -d --build`). |

Kết quả: Mỗi khi bạn cập nhật mã nguồn (ví dụ FastAPI, Prometheus config, model mới, v.v.), GitHub sẽ tự động đưa thay đổi lên EC2 và khởi động lại dịch vụ.

---

### Secrets cần cấu hình trong GitHub

Trước khi sử dụng, cần thêm các secrets trong repository:

| Tên Secret | Ý nghĩa |
|:------------|:---------|
| `EC2_HOST` | Public IP hoặc hostname của instance EC2 |
| `EC2_SSH_KEY` | Private key để GitHub Actions SSH vào EC2 |
| (tùy chọn) `EC2_USERNAME` | Mặc định là `ubuntu` |
## Giám sát & Cảnh báo (Monitoring & Alerting)

Hệ thống sử dụng **Prometheus** và **Grafana** để theo dõi hiệu năng mô hình cũng như phát hiện sớm các bất thường trong quá trình dự đoán.

### 1. Prometheus
- Thu thập các chỉ số từ API thông qua endpoint `/metrics`.  
- Các metric được ghi nhận gồm tốc độ dự đoán, độ trễ, số lượng mẫu xử lý và trạng thái học dần.  
- Dữ liệu được lưu trữ định kỳ và cung cấp cho Grafana để hiển thị trực quan.

### 2. Grafana
- Trực quan hóa các chỉ số được thu thập bởi Prometheus, bao gồm:

| Chỉ số | Ý nghĩa |
|:-------|:--------|
| **Prediction Rate** | Tốc độ xử lý và dự đoán các luồng dữ liệu đến |
| **Drift Ratio** | Mức độ sai lệch giữa dữ liệu mới và phân phối ban đầu |
| **Retrain Count** | Số lần mô hình được huấn luyện lại do phát hiện drift |
| **Average Latency** | Độ trễ trung bình của API trong quá trình xử lý yêu cầu |

### 3. Cảnh báo (Alerting)
Có thể thiết lập các rule cảnh báo trong Grafana để phát hiện bất thường:

- **Drift Ratio > 0.3:** Dữ liệu đầu vào thay đổi đáng kể → kích hoạt retrain.  
- **Latency cao bất thường:** Thời gian phản hồi tăng → cần kiểm tra hiệu năng mô hình hoặc hệ thống.  
- **API không phản hồi trong >60 giây:** Cảnh báo downtime hoặc lỗi kết nối đến FastAPI service.

## Tác giả

**MinhBQ**, **QuanTC**  
Khoa Mạng máy tính & Truyền thông – Trường Đại học Công nghệ Thông tin, ĐHQG-HCM  
(22520855, 22520938)@gm.uit.edu.vn  
