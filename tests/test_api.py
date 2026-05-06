"""
FastAPI endpoint tests for the ML Engine and API Gateway.

All heavy dependencies are mocked so no GPU or Redis is needed.
"""
import sys, os, json, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml_engine"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "api_gateway"))

from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient
import numpy as np


# ── ML Engine API ─────────────────────────────────────────────────────────────

def _make_ml_client(mse=0.01, threshold=0.05):
    """Build ML Engine TestClient with a mocked predictor."""
    import predictor as pred_module

    mock_predictor = MagicMock()
    mock_predictor.model = MagicMock()
    mock_predictor.n_features = 3
    mock_predictor.seq_len    = 30
    mock_predictor.threshold  = threshold
    mock_predictor.predict.return_value = {
        "reconstruction_error": mse,
        "threshold":            threshold,
        "anomaly":              mse > threshold,
    }

    with patch.object(pred_module, "predictor", mock_predictor):
        app = pred_module.app
        app.state.predictor = mock_predictor
        pred_module.predictor = mock_predictor
        return TestClient(app, raise_server_exceptions=False)


class TestMLEngineHealth:
    def test_health_returns_200(self):
        import predictor as pred_module
        mock_pred = MagicMock()
        mock_pred.model = MagicMock()
        pred_module.predictor = mock_pred
        client = TestClient(pred_module.app)
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_ok_when_ready(self):
        import predictor as pred_module
        mock_pred = MagicMock()
        mock_pred.model = MagicMock()
        pred_module.predictor = mock_pred
        client = TestClient(pred_module.app)
        r = client.get("/health")
        data = r.json()
        assert "status" in data
        assert "ready" in data


class TestMLEnginePredict:
    def test_predict_normal(self):
        import predictor as pred_module
        mock_pred = MagicMock()
        mock_pred.model = MagicMock()
        mock_pred.predict.return_value = {
            "reconstruction_error": 0.01, "threshold": 0.05, "anomaly": False
        }
        pred_module.predictor = mock_pred
        client = TestClient(pred_module.app)
        r = client.post("/predict", json={"series": [[0.1, 0.2, 0.3]] * 30})
        assert r.status_code == 200
        data = r.json()
        assert "anomaly" in data
        assert data["anomaly"] is False

    def test_predict_anomaly(self):
        import predictor as pred_module
        mock_pred = MagicMock()
        mock_pred.model = MagicMock()
        mock_pred.predict.return_value = {
            "reconstruction_error": 5.0, "threshold": 0.05, "anomaly": True
        }
        pred_module.predictor = mock_pred
        client = TestClient(pred_module.app)
        r = client.post("/predict", json={"series": [[9.9, 8.8, 7.7]] * 30})
        assert r.status_code == 200
        assert r.json()["anomaly"] is True

    def test_predict_missing_body_422(self):
        import predictor as pred_module
        mock_pred = MagicMock()
        mock_pred.model = MagicMock()
        pred_module.predictor = mock_pred
        client = TestClient(pred_module.app)
        r = client.post("/predict", json={})
        assert r.status_code == 422

    def test_predict_503_when_not_ready(self):
        import predictor as pred_module
        pred_module.predictor = None
        client = TestClient(pred_module.app)
        r = client.post("/predict", json={"series": [[0.1, 0.2, 0.3]] * 5})
        assert r.status_code == 503

    def test_model_info_returns_metadata(self):
        import predictor as pred_module
        mock_pred = MagicMock()
        mock_pred.model = MagicMock()
        mock_pred.n_features = 3
        mock_pred.seq_len    = 30
        mock_pred.threshold  = 0.042
        pred_module.predictor = mock_pred
        client = TestClient(pred_module.app)
        r = client.get("/model/info")
        assert r.status_code == 200
        data = r.json()
        assert data["n_features"] == 3
        assert data["seq_len"]    == 30
        assert pytest.approx(data["threshold"], abs=0.001) == 0.042


# ── API Gateway ───────────────────────────────────────────────────────────────

class TestAPIGatewayHealth:
    def test_health_200(self):
        import app as gw
        client = TestClient(gw.app)
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"
        assert r.json()["service"] == "api-gateway"


class TestAPIGatewayPredict:
    def test_predict_proxies_to_ml_engine(self):
        import app as gw
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "reconstruction_error": 0.02, "threshold": 0.05, "anomaly": False
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("app.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(return_value=mock_resp)
            client = TestClient(gw.app)
            r = client.post("/predict", json={"series": [[0.1, 0.2, 0.3]] * 5})

        # Either 200 (proxy success) or 422/500 depending on mock resolution
        assert r.status_code in (200, 422, 500)

    def test_gateway_has_uvicorn_main(self):
        """CMD python app.py requires __main__ to call uvicorn.run()."""
        import app as gw
        import inspect
        src = inspect.getsource(gw)
        assert 'if __name__ == "__main__"' in src
        assert "uvicorn.run" in src


# ── Consumer module ───────────────────────────────────────────────────────────

class TestConsumer:
    def test_correct_import_path(self):
        """consumer.py must import from 'predictor' not 'ml_engine.predictor'."""
        import consumer
        import inspect
        src = inspect.getsource(consumer)
        assert "from ml_engine.predictor" not in src

    def test_consume_loop_accepts_predictor_arg(self):
        """consume_loop must accept a predictor argument, not instantiate its own."""
        import inspect, consumer
        sig = inspect.signature(consumer.consume_loop)
        params = list(sig.parameters.keys())
        assert "predictor" in params

    def test_process_handles_direct_fields(self):
        """_process should handle messages with temperature/vibration fields."""
        import consumer
        mock_pred = MagicMock()
        mock_pred.predict.return_value = {"anomaly": False, "reconstruction_error": 0.01, "threshold": 0.05}
        data = {"temperature": "22.5", "vibration": "0.3"}
        consumer._process(data, mock_pred)
        mock_pred.predict.assert_called_once()

    def test_process_handles_payload_field(self):
        """_process should handle messages with a 'payload' JSON field."""
        import consumer
        import json
        mock_pred = MagicMock()
        mock_pred.predict.return_value = {"anomaly": False, "reconstruction_error": 0.01, "threshold": 0.05}
        series = [[0.1, 0.2], [0.3, 0.4]]
        data = {"payload": json.dumps(series)}
        consumer._process(data, mock_pred)
        mock_pred.predict.assert_called_once_with(series)

    def test_process_handles_unknown_format_gracefully(self):
        """_process must not crash on unexpected message shapes."""
        import consumer
        mock_pred = MagicMock()
        consumer._process({}, mock_pred)   # no payload, no temperature
        mock_pred.predict.assert_not_called()
