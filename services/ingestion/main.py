import time
import random
import redis
import os
import logging
from datetime import datetime, timezone
from pydantic import BaseModel

# ------------------------
# Configuration
# ------------------------
REDIS_HOST = os.getenv("REDIS_HOST", "redis")   # FIXED
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
STREAM_KEY = "supply_chain_stream"

# ------------------------
# Logging
# ------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Ingestion")

# ------------------------
# Redis connection
# ------------------------
def connect_redis():
    while True:
        try:
            client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=True
            )
            client.ping()
            logger.info("Connected to Redis successfully")
            return client
        except Exception as e:
            logger.warning(f"Waiting for Redis... ({e})")
            time.sleep(5)

r = connect_redis()

# ------------------------
# Data model
# ------------------------
class LogisticsEvent(BaseModel):
    timestamp: str
    sensor_id: str
    temperature: float
    vibration: float
    log_text: str

# ------------------------
# Generator
# ------------------------
def generate_mock_data():
    while True:
        is_anomaly = random.random() < 0.10

        temp = random.uniform(2.0, 8.0) if not is_anomaly else random.uniform(15.0, 25.0)
        vibration = random.uniform(0.1, 0.5) if not is_anomaly else random.uniform(2.0, 5.0)

        log_msg = (
            "Shipment stable. Route clear."
            if not is_anomaly
            else "ALERT: Cooling system failure. Route congestion detected."
        )

        event = LogisticsEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            sensor_id=f"TRUCK-{random.randint(100, 105)}",
            temperature=round(temp, 2),
            vibration=round(vibration, 2),
            log_text=log_msg
        )

        try:
            r.xadd(STREAM_KEY, event.model_dump())  # FIXED
            logger.info(f"Ingested {event.sensor_id} | anomaly={is_anomaly}")
        except Exception as e:
            logger.error(f"Redis write failed: {e}")

        time.sleep(1)

# ------------------------
# Entry
# ------------------------
if __name__ == "__main__":
    logger.info("Starting Ingestion Service...")
    generate_mock_data()

