┌─────────────────────────┐
│   Ingestion Service     │
│  (Schema-Validated)     │
│                         │
│  - Synthetic / Real     │
│    Logistics Events     │
└───────────┬─────────────┘
            │ Redis Streams
            ▼
┌─────────────────────────┐
│         Redis           │
│   (Streaming Backbone)  │
└───────────┬─────────────┘
            │ Consumer Group
            ▼
┌─────────────────────────┐
│       ML Engine         │
│  - Auto-Training        │
│  - LSTM Autoencoder     │
│  - Threshold Learning   │
│  - Model Persistence    │
│  - Prometheus Metrics   │
└───────────┬─────────────┘
            │ HTTP / Internal Calls
            ▼
┌─────────────────────────┐
│      API Gateway        │
│  - Inference Access     │
│  - Integration Layer    │
└─────────────────────────┘

        Observability Plane
────────────────────────────────
┌─────────────┐   ┌─────────────┐
│ Redis       │→→ │ Prometheus  │
│ Exporter    │   │             │
└─────────────┘   └──────┬──────┘
                          ▼
                   ┌─────────────┐
                   │  Grafana    │
                   │ Dashboards  │
                   └─────────────┘


Design Principles Highlighted

Loose coupling via streams

ML lifecycle automation

Metrics-first engineering

Failure-aware service startup

Production observability