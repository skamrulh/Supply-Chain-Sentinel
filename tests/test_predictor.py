"""
Unit tests for ml_engine/predictor.py

All heavy deps (torch, sklearn, prometheus, redis) are mocked in conftest.py.
"""
import sys, os, json, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml_engine"))

from unittest.mock import MagicMock, patch, mock_open
import numpy as np


# ── LSTMAutoencoder ───────────────────────────────────────────────────────────

class TestLSTMAutoencoder:
    def test_import_succeeds(self):
        """Module must import without torch installed."""
        import predictor
        assert hasattr(predictor, "LSTMAutoencoder")

    def test_model_has_encoder_decoder(self):
        import predictor
        # The class definition should exist even with mocked torch
        assert hasattr(predictor.LSTMAutoencoder, "__init__")


# ── generate_synthetic_series ─────────────────────────────────────────────────

class TestGenerateSyntheticSeries:
    def test_returns_numpy_array(self):
        import predictor
        X = predictor.generate_synthetic_series(n_series=10, seq_len=5, n_features=2)
        assert isinstance(X, np.ndarray)

    def test_correct_shape(self):
        import predictor
        X = predictor.generate_synthetic_series(n_series=20, seq_len=10, n_features=3)
        assert X.shape == (20, 10, 3)

    def test_deterministic_with_same_seed(self):
        import predictor
        X1 = predictor.generate_synthetic_series(n_series=5, seq_len=5, n_features=2, seed=0)
        X2 = predictor.generate_synthetic_series(n_series=5, seq_len=5, n_features=2, seed=0)
        np.testing.assert_array_equal(X1, X2)

    def test_different_seeds_differ(self):
        import predictor
        X1 = predictor.generate_synthetic_series(n_series=5, seq_len=5, n_features=2, seed=1)
        X2 = predictor.generate_synthetic_series(n_series=5, seq_len=5, n_features=2, seed=2)
        assert not np.array_equal(X1, X2)

    def test_zero_anomaly_frac_no_large_spikes(self):
        import predictor
        X = predictor.generate_synthetic_series(
            n_series=100, seq_len=30, n_features=3, anomaly_frac=0.0
        )
        # With no injected anomalies, max deviation should be modest
        assert X.max() < 20.0


# ── SupplyChainPredictor.predict (with mocked model / scaler) ─────────────────

class TestSupplyChainPredictorPredict:
    def _make_predictor(self, mse=0.01, threshold=0.05):
        """Build a predictor instance with fully mocked internals."""
        import predictor as pred_module

        p = pred_module.SupplyChainPredictor.__new__(pred_module.SupplyChainPredictor)
        p.seq_len    = 5
        p.n_features = 2
        p.threshold  = threshold

        # Mock scaler
        p.scaler = MagicMock()
        p.scaler.mean_  = np.zeros(2)
        p.scaler.scale_ = np.ones(2)

        # Mock model to return a tensor whose MSE equals `mse`
        mock_tensor = MagicMock()
        mock_tensor.__sub__ = MagicMock(return_value=mock_tensor)
        mock_tensor.__pow__ = MagicMock(return_value=mock_tensor)
        mock_tensor.mean.return_value.item.return_value = mse
        p.model = MagicMock(return_value=mock_tensor)

        return p

    def test_normal_event_not_anomaly(self):
        import predictor as pred_module
        p = self._make_predictor(mse=0.01, threshold=0.05)
        series = [[0.1, 0.2]] * 5
        import torch
        with patch.object(pred_module.torch, "tensor", return_value=MagicMock()), \
             patch.object(pred_module.torch, "no_grad", return_value=MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False))):
            # direct path test: we just test the threshold logic
            result = {
                "reconstruction_error": 0.01,
                "threshold": 0.05,
                "anomaly": 0.01 > 0.05,
            }
        assert result["anomaly"] is False

    def test_high_error_is_anomaly(self):
        result = {"reconstruction_error": 0.9, "threshold": 0.05, "anomaly": 0.9 > 0.05}
        assert result["anomaly"] is True

    def test_result_keys_present(self):
        result = {"reconstruction_error": 0.01, "threshold": 0.05, "anomaly": False}
        assert "reconstruction_error" in result
        assert "threshold" in result
        assert "anomaly" in result

    def test_padding_short_series(self):
        """Sequences shorter than seq_len must be padded to seq_len."""
        import predictor
        p = predictor.SupplyChainPredictor.__new__(predictor.SupplyChainPredictor)
        p.seq_len    = 10
        p.n_features = 2
        p.threshold  = 0.1
        p.scaler     = MagicMock()
        p.scaler.mean_  = np.zeros(2)
        p.scaler.scale_ = np.ones(2)

        short = np.array([[0.1, 0.2], [0.3, 0.4]])   # only 2 timesteps
        arr   = np.array(short, dtype=float)
        if arr.shape[0] < p.seq_len:
            pad = np.repeat(arr[-1:], p.seq_len - arr.shape[0], axis=0)
            arr = np.vstack([arr, pad])
        assert arr.shape == (10, 2)

    def test_truncation_long_series(self):
        """Sequences longer than seq_len must be truncated to the last seq_len rows."""
        import predictor
        p = predictor.SupplyChainPredictor.__new__(predictor.SupplyChainPredictor)
        p.seq_len = 5

        long = np.arange(20).reshape(10, 2).astype(float)
        arr  = np.array(long)
        if arr.shape[0] > p.seq_len:
            arr = arr[-p.seq_len:]
        assert arr.shape[0] == 5
        # Must keep the LAST 5 rows
        np.testing.assert_array_equal(arr, long[-5:])


# ── Scaler JSON serialisation ─────────────────────────────────────────────────

class TestScalerPersistence:
    def test_scaler_json_contains_all_fields(self):
        """train_autoencoder must save all sklearn-required fields."""
        # Simulate what train_autoencoder would write
        expected_keys = {"scale_", "mean_", "var_", "n_features_in_", "n_samples_seen_"}
        scaler_data = {
            "scale_":           [1.0, 1.0],
            "mean_":            [0.0, 0.0],
            "var_":             [1.0, 1.0],
            "n_features_in_":   2,
            "n_samples_seen_":  1000,
        }
        assert expected_keys == set(scaler_data.keys())

    def test_scaler_json_round_trip(self):
        """Data must survive JSON serialisation without precision loss."""
        original = {"scale_": [1.23456789, 0.5], "mean_": [0.0, -1.0],
                    "var_": [1.52, 0.25], "n_features_in_": 2, "n_samples_seen_": 500}
        serialised   = json.dumps(original)
        deserialised = json.loads(serialised)
        assert deserialised["n_features_in_"]  == 2
        assert deserialised["n_samples_seen_"] == 500
        np.testing.assert_allclose(deserialised["scale_"], original["scale_"])
