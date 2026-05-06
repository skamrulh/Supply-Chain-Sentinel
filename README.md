# Supply Chain Sentinel

![CI](https://github.com/<your-username>/Supply-Chain-Sentinel/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red)
![Docker](https://img.shields.io/badge/deploy-Docker-2496ED)
![License](https://img.shields.io/badge/license-MIT-green)

AI-Driven Anomaly Detection & Observability Platform for Logistics Systems.

Detects operational anomalies in supply chain telemetry in real time using an **LSTM Autoencoder**, Redis Streams for event ingestion, and a full Prometheus + Grafana observability stack.

---

## Architecture

```
[Ingestion Service]
  Generates synthetic logistics events (temperature, vibration, sensor ID)
  → writes to Redis Stream "supply_chain_stream" every 1 second
        │
        │  Redis Stream
        ▼
[ML Engine]                               port 5000 (API)  port 8001 (metrics)
  ├── FastAPI /predict /health /model/info
  ├── LSTM Autoencoder (auto-trains on first run, persists to disk)
  ├── Redis Stream consumer (background thread — reads & runs inference)
  └── Prometheus metrics: sentinel_predictions_total,
                          sentinel_anomalies_total,
                          sentinel_reconstruction_error,
                          sentinel_model_threshold
        │
        ▼
[API Gateway]                             port 8001 (external)
  ├── POST /predict   — proxies to ML Engine
  ├── GET  /health    — gateway liveness
  └── WS   /ws/alerts — streams anomaly events from Redis pub/sub

        Observability Plane
─────────────────────────────────────────────────
[Redis Exporter :9121] → [Prometheus :9090] → [Grafana :3000]
```

---

## Quick Start

```bash
git clone https://github.com/<your-username>/Supply-Chain-Sentinel.git
cd Supply-Chain-Sentinel

docker compose build --no-cache
docker compose up
```

**First run:** the ML Engine auto-trains the LSTM Autoencoder on 2000 synthetic series (~1–2 min on CPU), then saves the model to `ml_engine/models/`. Subsequent restarts load the saved model in seconds.

| Service | URL |
|---|---|
| ML Engine API + Swagger | http://localhost:5000/docs |
| API Gateway | http://localhost:8001 |
| Prometheus | http://localhost:9090 |
| Grafana (admin/admin) | http://localhost:3000 |
| Redis Exporter | http://localhost:9121/metrics |

---

## API Usage

### Check health

```bash
curl http://localhost:5000/health
# {"status": "ok", "ready": true}
```

### Run anomaly detection

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "series": [
      [3.5, 0.2],
      [4.1, 0.3],
      [3.8, 0.1],
      [3.9, 0.2]
    ]
  }'
```

```json
{
  "reconstruction_error": 0.0032,
  "threshold": 0.0187,
  "anomaly": false
}
```

### Simulate an anomaly

```bash
# High temperature + vibration spike
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"series": [[22.0, 4.5], [23.5, 5.1], [25.0, 6.2]]}'
```

```json
{
  "reconstruction_error": 0.1842,
  "threshold": 0.0187,
  "anomaly": true
}
```

### Get model metadata

```bash
curl http://localhost:5000/model/info
# {"n_features": 3, "seq_len": 30, "threshold": 0.0187, "device": "cpu"}
```

---

## Prometheus Metrics

Exposed at `http://localhost:8001/metrics`:

| Metric | Type | Description |
|---|---|---|
| `sentinel_predictions_total` | Counter | Total inference calls |
| `sentinel_anomalies_total` | Counter | Total anomalies flagged |
| `sentinel_reconstruction_error` | Histogram | MSE per prediction |
| `sentinel_model_threshold` | Gauge | Current anomaly threshold |

---

## Running Tests

```bash
pip install fastapi uvicorn httpx pytest pydantic numpy redis websockets
pytest tests/ -v
# 30+ tests across predictor, API endpoints, consumer, and ingestion service
```

---

## How the LSTM Autoencoder Works

1. **Training** (first run only): 2000 synthetic multivariate time-series (temperature, vibration, delivery delay — 3 features × 30 timesteps) are generated, scaled with `StandardScaler`, and used to train the autoencoder for 25 epochs.

2. **Threshold calibration**: After training, reconstruction errors on the validation set are computed. The anomaly threshold is set to `mean + 3 × std` of validation errors.

3. **Inference**: Each incoming event series is scaled, encoded, decoded, and the MSE between input and reconstruction is compared to the threshold.

4. **Model persistence**: Weights, scaler parameters (all sklearn fields), and threshold are saved to `ml_engine/models/`. Warm restarts skip training entirely.

---

## Tech Stack

`Python 3.10` · `PyTorch 2.2` · `FastAPI` · `Redis Streams` · `prometheus-client` · `Grafana` · `Prometheus` · `Docker Compose` · `GitHub Actions`
