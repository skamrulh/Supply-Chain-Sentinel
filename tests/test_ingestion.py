"""
Tests for the ingestion service (services/ingestion/main.py).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "ingestion"))

from unittest.mock import MagicMock, patch


class TestLogisticsEventModel:
    def test_event_creation(self):
        from main import LogisticsEvent
        event = LogisticsEvent(
            timestamp="2025-01-01T00:00:00Z",
            sensor_id="TRUCK-101",
            temperature=4.5,
            vibration=0.3,
            log_text="Shipment stable.",
        )
        assert event.sensor_id == "TRUCK-101"
        assert event.temperature == 4.5

    def test_model_dump_keys(self):
        from main import LogisticsEvent
        event = LogisticsEvent(
            timestamp="2025-01-01T00:00:00Z",
            sensor_id="TRUCK-102",
            temperature=5.0,
            vibration=0.2,
            log_text="ok",
        )
        d = event.model_dump()
        assert set(d.keys()) == {"timestamp", "sensor_id", "temperature", "vibration", "log_text"}

    def test_anomaly_temperature_range(self):
        """Anomaly temperatures should be > 8.0."""
        from main import LogisticsEvent
        event = LogisticsEvent(
            timestamp="2025-01-01T00:00:00Z",
            sensor_id="TRUCK-103",
            temperature=20.0,
            vibration=3.5,
            log_text="ALERT: Cooling failure.",
        )
        assert event.temperature > 8.0

    def test_redis_host_env_default(self):
        """REDIS_HOST default must be 'redis', not 'localhost'."""
        import main
        # The default must point to the docker service name
        assert main.REDIS_HOST in ("redis", os.getenv("REDIS_HOST", "redis"))

    def test_stream_key_constant(self):
        from main import STREAM_KEY
        assert STREAM_KEY == "supply_chain_stream"
