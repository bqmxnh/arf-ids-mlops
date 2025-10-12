FROM python:3.11-slim

WORKDIR /app

# Copy toàn bộ mã nguồn để chắc chắn không thiếu module nào
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "src.arf_api:app", "--host", "0.0.0.0", "--port", "8000"]

