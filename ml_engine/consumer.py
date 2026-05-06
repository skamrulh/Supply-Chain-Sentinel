"""
Supply Chain Sentinel — Redis Stream Consumer

Reads logistics events from the Redis Stream 'supply_chain_stream'
via a consumer group and runs inference through SupplyChainPredictor.

Started as a background thread by predictor.py's @app.on_event("startup"),
so both the API and the consumer run inside the same container process.
"""
import time
import json
import logging
import os

import redis

LOG = logging.getLogger("consumer")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

REDIS_URL     = os.getenv("REDIS_URL", "redis://redis:6379/0")
STREAM_KEY    = "supply_chain_stream"
GROUP_NAME    = "ml_group"
CONSUMER_NAME = "ml_engine_1"


def ensure_consumer_group(r: redis.Redis) -> None:
    try:
        r.xgroup_create(STREAM_KEY, GROUP_NAME, id="0", mkstream=True)
        LOG.info("Consumer group created.")
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" in str(e) or "already exists" in str(e):
            LOG.info("Consumer group already exists.")
        else:
            raise


def consume_loop(predictor) -> None:
    """
    Main consumer loop. Receives a SupplyChainPredictor instance so it can
    call predictor.predict() directly — no extra imports needed.

    Args:
        predictor: an initialised SupplyChainPredictor from predictor.py
    """
    r = _connect(REDIS_URL)
    ensure_consumer_group(r)

    LOG.info("Consumer loop started.")
    while True:
        try:
            entries = r.xreadgroup(
                GROUP_NAME, CONSUMER_NAME,
                {STREAM_KEY: ">"}, count=10, block=5000,
            )
            if not entries:
                continue
            for _stream, msgs in entries:
                for msg_id, data in msgs:
                    try:
                        _process(data, predictor)
                        r.xack(STREAM_KEY, GROUP_NAME, msg_id)
                    except Exception:
                        LOG.exception("Processing failed — message kept for retry.")
        except redis.exceptions.ConnectionError:
            LOG.warning("Redis connection lost — retrying in 5 s…")
            time.sleep(5)
            r = _connect(REDIS_URL)
            ensure_consumer_group(r)
        except Exception:
            LOG.exception("Unexpected consumer error — retrying in 5 s…")
            time.sleep(5)


def _connect(url: str) -> redis.Redis:
    while True:
        try:
            client = redis.Redis.from_url(url, decode_responses=True)
            client.ping()
            LOG.info("Consumer connected to Redis.")
            return client
        except Exception as e:
            LOG.warning(f"Waiting for Redis… ({e})")
            time.sleep(5)


def _process(data: dict, predictor) -> None:
    """Parse a stream message and run inference."""
    payload = data.get("payload") or data.get("data")

    # The ingestion service writes fields directly into the stream entry
    # so 'payload' may be absent; fall back to building series from the fields
    if payload is None:
        try:
            series = [[float(data["temperature"]), float(data["vibration"])]]
        except (KeyError, ValueError):
            LOG.warning(f"Unrecognised message format: {data}")
            return
    else:
        series = json.loads(payload) if isinstance(payload, str) else payload

    result = predictor.predict(series)
    if result["anomaly"]:
        LOG.warning(
            f"ANOMALY DETECTED  error={result['reconstruction_error']:.4f}"
            f"  threshold={result['threshold']:.4f}"
        )
    else:
        LOG.debug(f"Normal  error={result['reconstruction_error']:.4f}")
