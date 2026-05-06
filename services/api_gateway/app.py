"""
Supply Chain Sentinel — API Gateway

Provides:
  GET  /health       — liveness check
  GET  /predict      — proxy to ML Engine /predict
  WS   /ws/alerts    — real-time anomaly push over WebSocket (Redis pub/sub)
"""
import os
import json
import logging
import httpx

import redis.asyncio as aioredis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn

LOG = logging.getLogger("api_gateway")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
ML_ENGINE_URL = os.getenv("ML_ENGINE_URL", "http://ml_engine:5000")

app = FastAPI(title="Supply Chain Sentinel API Gateway", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "healthy", "service": "api-gateway"}


# ── Proxy to ML Engine /predict ───────────────────────────────────────────────
class PredictRequest(BaseModel):
    series: List[List[float]]


@app.post("/predict")
async def predict(req: PredictRequest):
    """Forward prediction requests to the ML Engine."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(
                f"{ML_ENGINE_URL}/predict",
                json={"series": req.series},
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code,
                                detail=e.response.text)
        except httpx.RequestError as e:
            raise HTTPException(status_code=503,
                                detail=f"ML Engine unreachable: {e}")


# ── WebSocket — real-time anomaly alerts ──────────────────────────────────────
@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """
    Stream anomaly events to connected clients.

    The ML Engine consumer publishes anomaly events to the Redis channel
    'alerts_channel'; this endpoint relays them to WebSocket clients.
    """
    await websocket.accept()
    r = aioredis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    pubsub = r.pubsub()
    await pubsub.subscribe("alerts_channel")
    LOG.info("WebSocket client connected to alerts channel.")
    try:
        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    data = json.loads(message["data"])
                    await websocket.send_json(data)
                except json.JSONDecodeError:
                    await websocket.send_text(message["data"])
    except WebSocketDisconnect:
        LOG.info("WebSocket client disconnected.")
    except Exception as e:
        LOG.warning(f"WebSocket error: {e}")
    finally:
        await pubsub.unsubscribe("alerts_channel")
        await r.aclose()


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
