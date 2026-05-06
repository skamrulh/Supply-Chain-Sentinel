FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt


RUN pip install --upgrade pip \
    && pip install --no-cache-dir \
       --index-url https://download.pytorch.org/whl/cpu \
       torch==2.2.2 \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 5000


HEALTHCHECK --interval=10s --timeout=5s --start-period=90s --retries=12 \
  CMD curl -f http://localhost:5000/health || exit 1

CMD ["python", "predictor.py"]
