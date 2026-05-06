"""
Stub heavy dependencies (torch, sklearn) so tests run in ~1s without GPU.
"""
import sys
from unittest.mock import MagicMock, patch
import numpy as np

# ── Stub torch ────────────────────────────────────────────────────────────────
torch_mock = MagicMock()
# make torch.device return something sensible
torch_mock.device.return_value = "cpu"
torch_mock.cuda.is_available.return_value = False
torch_mock.no_grad.return_value.__enter__ = lambda s: s
torch_mock.no_grad.return_value.__exit__ = MagicMock(return_value=False)
sys.modules.setdefault("torch", torch_mock)
sys.modules.setdefault("torch.nn", MagicMock())

# ── Stub sklearn ──────────────────────────────────────────────────────────────
sklearn_mock = MagicMock()
sys.modules.setdefault("sklearn", sklearn_mock)
sys.modules.setdefault("sklearn.preprocessing", MagicMock())

# ── Stub prometheus_client ────────────────────────────────────────────────────
prom = MagicMock()
# make Counter/Histogram/Gauge return objects that support .inc()/.observe()/.set()
prom.Counter.return_value  = MagicMock()
prom.Histogram.return_value = MagicMock()
prom.Gauge.return_value    = MagicMock()
sys.modules.setdefault("prometheus_client", prom)

# ── Stub redis ────────────────────────────────────────────────────────────────
sys.modules.setdefault("redis", MagicMock())
sys.modules.setdefault("redis.asyncio", MagicMock())
sys.modules.setdefault("redis.exceptions", MagicMock())
