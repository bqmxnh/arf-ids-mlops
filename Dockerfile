FROM python:3.11-slim

WORKDIR /app
COPY ./src ./src
COPY ./models ./models
COPY ./dataset ./dataset
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "src.arf_api:app", "--host", "0.0.0.0", "--port", "8000"]
