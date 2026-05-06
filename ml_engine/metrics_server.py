"""
metrics_server.py — SUPERSEDED

Prometheus metrics are now started directly inside predictor.py via
prometheus_client.start_http_server(8001) in the @app.on_event("startup")
callback.

Custom metrics registered in predictor.py:
  sentinel_predictions_total    — counter, incremented on each /predict call
  sentinel_anomalies_total      — counter, incremented when anomaly=True
  sentinel_reconstruction_error — histogram of MSE values
  sentinel_model_threshold      — gauge, set to the trained threshold value

This file is kept only for reference and is NOT imported or executed.
"""
