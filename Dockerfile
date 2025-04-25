FROM python:3.10-slim

WORKDIR /app
# ffmpeg を入れる
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
CMD gunicorn --bind "0.0.0.0:${PORT:-8080}" main:app
