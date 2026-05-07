"""
Supply Chain Sentinel — ML Engine
==================================

FastAPI service that:
- Trains an LSTM Autoencoder on first run (synthetic logistics data)
- Persists the trained model and scaler to disk for warm restarts
- Exposes /predict and /health on port 5000
- Exposes Prometheus metrics on port 8001 (background thread)

Prometheus metrics:
  sentinel_predictions_total     — counter
  sentinel_anomalies_total       — counter
  sentinel_reconstruction_error  — histogram
  sentinel_model_threshold       — gauge
"""

import os
import json
import time
import logging
import threading
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from prometheus_client import Counter, Histogram, Gauge, start_http_server

# ── Logging ───────────────────────────────────────────────────────────────────
LOG = logging.getLogger("predictor")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ── Paths / Device ────────────────────────────────────────────────────────────

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).parent / "models"))
MODEL_DIR.mkdir(exist_ok=True, parents=True)

MODEL_PATH  = MODEL_DIR / "lstm_autoencoder.pth"
SCALER_PATH = MODEL_DIR / "lstm_autoencoder.scaler.json"
THRESH_PATH = MODEL_DIR / "recon_threshold.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Prometheus Metrics ────────────────────────────────────────────────────────
PREDICTIONS_TOTAL = Counter(
    "sentinel_predictions_total",
    "Total inference requests processed",
)
ANOMALIES_TOTAL = Counter(
    "sentinel_anomalies_total",
    "Total anomalies detected",
)
RECON_ERROR = Histogram(
    "sentinel_reconstruction_error",
    "LSTM Autoencoder reconstruction MSE per prediction",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
)
MODEL_THRESHOLD = Gauge(
    "sentinel_model_threshold",
    "Current anomaly threshold (mean + 3*std of validation reconstruction errors)",
)

# ── Model ─────────────────────────────────────────────────────────────────────
class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hid_dim: int = 64, n_layers: int = 1):
        super().__init__()
        self.n_features = n_features
        self.hid_dim    = hid_dim
        self.n_layers   = n_layers
        self.encoder = nn.LSTM(input_size=n_features, hidden_size=hid_dim,
                               num_layers=n_layers, batch_first=True)
        self.decoder = nn.LSTM(input_size=hid_dim, hidden_size=hid_dim,
                               num_layers=n_layers, batch_first=True)
        self.output_layer = nn.Linear(hid_dim, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, c) = self.encoder(x)
        batch_size, seq_len, _ = x.size()
        dec_in = torch.zeros((batch_size, seq_len, self.hid_dim), device=x.device)
        dec_out, _ = self.decoder(dec_in, (h, c))
        return self.output_layer(dec_out)


# ── Synthetic Data ────────────────────────────────────────────────────────────
def generate_synthetic_series(n_series=1000, seq_len=30, n_features=3,
                               anomaly_frac=0.01, seed=42):
    rng, X = np.random.RandomState(seed), []
    for _ in range(n_series):
        t = np.linspace(0, 50, seq_len)
        s = np.stack([
            np.sin(t * (0.1 + 0.05 * f)) + 0.1 * rng.randn(seq_len) +
            (0.01 * np.arange(seq_len) if rng.rand() < 0.3 else 0)
            for f in range(n_features)
        ], axis=1)
        if rng.rand() < anomaly_frac:
            s[rng.randint(0, seq_len):, rng.randint(0, n_features)] += rng.uniform(5, 10)
        X.append(s)
    return np.stack(X)


# ── Training ──────────────────────────────────────────────────────────────────
def train_autoencoder(save_path, n_features, seq_len=30, epochs=25,
                      batch_size=64, lr=1e-3):
    LOG.info("Auto-training LSTM Autoencoder...")
    X = generate_synthetic_series(n_series=2000, seq_len=seq_len,
                                  n_features=n_features, anomaly_frac=0.0)
    nsamples, _, nfeat = X.shape
    scaler = StandardScaler().fit(X.reshape(-1, nfeat))
    Xs     = scaler.transform(X.reshape(-1, nfeat)).reshape(nsamples, seq_len, nfeat)

    split   = int(0.8 * nsamples)
    X_train = torch.tensor(Xs[:split], dtype=torch.float32, device=DEVICE)
    X_val   = torch.tensor(Xs[split:], dtype=torch.float32, device=DEVICE)

    model     = LSTMAutoencoder(n_features=n_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(X_train.size(0))
        train_loss = 0.0
        for i in range(0, X_train.size(0), batch_size):
            batch = X_train[perm[i: i + batch_size]]
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, X_train.size(0) / batch_size)
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), X_val).item()
        LOG.info(f"Epoch {epoch}/{epochs} train={train_loss:.6f} val={val_loss:.6f}")

    with torch.no_grad():
        errors = ((model(X_val) - X_val) ** 2).mean(dim=(1, 2)).cpu().numpy()

    mean_err = float(errors.mean())
    std_err  = float(errors.std())
    threshold = mean_err + 3.0 * std_err

    torch.save({"model_state": model.state_dict(), "n_features": n_features,
                "seq_len": seq_len, "hid_dim": model.hid_dim, "n_layers": model.n_layers},
               save_path)

    # Persist ALL scaler fields needed to fully reconstruct sklearn StandardScaler
    with open(SCALER_PATH, "w") as f:
        json.dump({
            "scale_":           scaler.scale_.tolist(),
            "mean_":            scaler.mean_.tolist(),
            "var_":             scaler.var_.tolist(),
            "n_features_in_":   int(scaler.n_features_in_),
            "n_samples_seen_":  int(scaler.n_samples_seen_),
        }, f)

    with open(THRESH_PATH, "w") as f:
        json.dump({"threshold": threshold, "mean_err": mean_err, "std_err": std_err}, f)

    LOG.info(f"Training complete. threshold={threshold:.6f}")
    return model, scaler, threshold


# ── Predictor ─────────────────────────────────────────────────────────────────
class SupplyChainPredictor:
    def __init__(self):
        self.model:     LSTMAutoencoder = None
        self.scaler:    StandardScaler  = None
        self.threshold: float           = None
        self.seq_len    = 30
        self.n_features = 3
        self._load_or_train()

    def _load_or_train(self):
        if MODEL_PATH.exists() and SCALER_PATH.exists() and THRESH_PATH.exists():
            LOG.info("Loading saved model...")
            data = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
            self.n_features = data["n_features"]
            self.seq_len    = data["seq_len"]
            self.model = LSTMAutoencoder(n_features=self.n_features,
                                         hid_dim=data["hid_dim"],
                                         n_layers=data["n_layers"]).to(DEVICE)
            self.model.load_state_dict(data["model_state"])
            self.model.eval()

            # Reconstruct fully-qualified StandardScaler (all sklearn attrs)
            with open(SCALER_PATH) as f:
                meta = json.load(f)
            self.scaler = StandardScaler()
            self.scaler.scale_          = np.array(meta["scale_"])
            self.scaler.mean_           = np.array(meta["mean_"])
            self.scaler.var_            = np.array(meta["var_"])
            self.scaler.n_features_in_  = int(meta["n_features_in_"])
            self.scaler.n_samples_seen_ = int(meta["n_samples_seen_"])

            with open(THRESH_PATH) as f:
                self.threshold = float(json.load(f)["threshold"])
            MODEL_THRESHOLD.set(self.threshold)
            LOG.info("Model loaded.")
        else:
            LOG.warning("No model found — starting auto-training...")
            self.model, self.scaler, self.threshold = train_autoencoder(
                save_path=MODEL_PATH, n_features=self.n_features, seq_len=self.seq_len)
            MODEL_THRESHOLD.set(self.threshold)

    def predict(self, series) -> dict:
        arr = np.array(series, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[0] > self.seq_len:
            arr = arr[-self.seq_len:]
        elif arr.shape[0] < self.seq_len:
            arr = np.vstack([arr, np.repeat(arr[-1:], self.seq_len - arr.shape[0], axis=0)])

        arr = (arr - self.scaler.mean_) / self.scaler.scale_
        x   = torch.tensor(arr, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            mse = float(((self.model(x) - x) ** 2).mean().item())

        is_anomaly = mse > self.threshold
        PREDICTIONS_TOTAL.inc()
        RECON_ERROR.observe(mse)
        if is_anomaly:
            ANOMALIES_TOTAL.inc()

        return {"reconstruction_error": mse, "threshold": self.threshold,
                "anomaly": is_anomaly}


# ── Background Prometheus metrics server ──────────────────────────────────────
def _start_metrics_server(port: int = 8001):
    try:
        start_http_server(port)
        LOG.info(f"Prometheus metrics server on :{port}")
    except OSError as e:
        LOG.warning(f"Metrics server port {port} unavailable: {e}")


# ── FastAPI App ───────────────────────────────────────────────────────────────
predictor: SupplyChainPredictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    threading.Thread(target=_start_metrics_server, args=(8001,), daemon=True).start()
    predictor = SupplyChainPredictor()
    from consumer import consume_loop   # local import (same /app directory)
    threading.Thread(target=consume_loop, args=(predictor,), daemon=True,
                     name="redis-consumer").start()
    LOG.info("ML Engine ready — API on :5000, metrics on :8001, consumer running.")
    yield


app = FastAPI(title="Supply Chain Sentinel — ML Engine", version="1.0.0",
              lifespan=lifespan)


class PredictRequest(BaseModel):
    series: List[List[float]]


@app.get("/health")
def health():
    ready = predictor is not None and predictor.model is not None
    return {"status": "ok" if ready else "training", "ready": ready}


@app.post("/predict")
def predict(req: PredictRequest):
    if predictor is None:
        raise HTTPException(503, detail="Model not ready")
    try:
        return predictor.predict(req.series)
    except Exception as e:
        LOG.exception("Prediction failed")
        raise HTTPException(500, detail=str(e))


@app.get("/model/info")
def model_info():
    if predictor is None:
        raise HTTPException(503, detail="Model not ready")
    return {"n_features": predictor.n_features, "seq_len": predictor.seq_len,
            "threshold": predictor.threshold, "device": str(DEVICE)}


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")
