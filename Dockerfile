FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements-lock-py312.txt ./
RUN pip install --no-cache-dir -r requirements-lock-py312.txt
COPY . .
CMD ["python", "run.py"]
