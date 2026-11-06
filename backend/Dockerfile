FROM python:3.11-slim

WORKDIR /app

# System dependencies for numpy/scipy builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train models at build time
# (alternatively mount pre-trained saved_models/ as a volume)
RUN python ml/train.py

EXPOSE 5000
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:5000", "app:create_app()"]

