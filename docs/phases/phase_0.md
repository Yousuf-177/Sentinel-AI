# Phase 0 — Project Scaffolding & Environment Foundation

**Status:** ✅ Complete  
**Duration:** Phase 0 (Pre-development setup)  
**Date Completed:** October 2026  

---

## 📋 Objective

Set up the complete project scaffold, repository structure, configuration files, and code architecture for SENTINEL-AI — ready for Phase 1 (data & EDA) to begin immediately without any setup friction.

---

## ✅ Deliverables Completed

### 1. Repository Structure (Mono-repo)

Created a well-organized mono-repo with clear separation of concerns:

```
Sentinel-AI/
├── ml/                          # Machine Learning pipeline
│   ├── config/                  # Feature schemas, model configs, alert schema
│   │   ├── feature_schema.json  # 42 CICFlowMeter-compatible features (single source of truth)
│   │   ├── model_config.yaml    # CNN-LSTM hyperparameters & training config
│   │   └── alert_schema.json    # Frozen alert JSON contract for all downstream consumers
│   ├── data/                    # Data loading & preprocessing
│   │   ├── data_loader.py       # CIC-IDS2017 & UNSW-NB15 dataset loaders
│   │   └── preprocessing.py     # Cleaning, label mapping, scaling, splitting
│   ├── models/                  # Model architecture & training
│   │   ├── cnn_lstm.py          # 1D-CNN + LSTM hybrid architecture
│   │   ├── trainer.py           # Training loop, evaluation, artifact saving
│   │   └── artifacts/           # Saved model files (gitignored)
│   ├── explainability/          # XAI module
│   │   └── shap_explainer.py    # SHAP DeepExplainer + human-readable output
│   ├── ingestion/               # Packet capture & flow aggregation
│   │   ├── packet_sniffer.py    # Scapy-based capture with 5-tuple extraction
│   │   └── flow_aggregator.py   # Hash-based bidirectional flow table
│   ├── inference/               # Real-time inference bridge
│   │   └── pipeline.py          # End-to-end: flow → preprocess → predict → explain → route
│   ├── notebooks/               # Jupyter notebooks
│   │   └── 01_eda_plan.md       # EDA notebook plan (placeholder for .ipynb)
│   ├── tests/                   # Unit tests
│   │   └── test_smoke.py        # Import verification & config validation tests
│   └── requirements.txt         # Python dependencies (pinned minimum versions)
│
├── backend/                     # Node.js real-time backend
│   ├── src/server.js            # Express + Socket.io server with health check
│   └── package.json             # Node.js dependencies
│
├── frontend/                    # React dashboard
│   ├── src/App.js               # Placeholder (built in Phase 5)
│   └── package.json             # React + Chart.js + Socket.io-client dependencies
│
├── infra/                       # Infrastructure & deployment
│   ├── docker/
│   │   ├── Dockerfile.inference # Python inference engine container
│   │   └── Dockerfile.backend   # Node.js backend container
│   └── scripts/
│       └── download_datasets.py # Dataset download instructions & directory setup
│
├── data/                        # Dataset storage (gitignored)
│   └── README.md                # Setup instructions
│
├── docs/                        # Documentation
│   └── architecture/README.md   # Architecture docs placeholder
│
├── presentation/                # Presentation materials
├── reports/                     # Generated PDF reports (Phase 7)
│
├── .gitignore                   # Python, Node.js, ML, datasets, IDE, OS
├── .env.example                 # All configurable environment variables
├── docker-compose.yml           # Full-stack orchestration (scaffold)
├── LICENSE                      # MIT License with academic use notice
├── README.md                    # Comprehensive project README
└── SENTINEL-AI_Project_Plan.md  # Detailed project plan
```

---

### 2. Git Repository

| Item | Status |
|------|--------|
| Git initialized | ✅ `git init` completed |
| `.gitignore` created | ✅ Covers Python, Node.js, ML artifacts, datasets, IDE, OS files |
| Branch: `main` | ✅ Ready for first commit |

---

### 3. Configuration Files

#### 3.1 Feature Schema (`ml/config/feature_schema.json`)
- **42 CICFlowMeter-compatible features** defined with name, dtype, description, and category
- Categories: temporal, packet_count, packet_length, flow_rate, inter_arrival, flags, header
- **6-class label schema** with mapping notes
- This is the **single source of truth** used by both training and live flow aggregation

#### 3.2 Model Configuration (`ml/config/model_config.yaml`)
- CNN architecture: 2 Conv1D layers (64, 128 filters), kernel size 3, MaxPooling, BatchNorm
- LSTM architecture: 2 LSTM layers (128, 64 units), dropout 0.3
- Dense head: 2 Dense layers (128, 64), dropout 0.4, softmax output
- Training: Adam optimizer, lr=0.001, batch size 256, 50 epochs
- EarlyStopping (patience=7), ReduceLROnPlateau (factor=0.5, patience=3)
- Class weights: "balanced" (auto-computed)
- Data split: 70/15/15 stratified

#### 3.3 Alert Schema (`ml/config/alert_schema.json`)
- JSON Schema (draft-07) defining the frozen contract for alert objects
- Fields: alert_id, timestamp, flow_id, 5-tuple, label, confidence, explanation, action, mitigation_status, inference_latency_ms
- Actions: ALLOW, ALERT, MITIGATE
- This contract is consumed by: backend, dashboard, database, report generator

#### 3.4 Environment Template (`.env.example`)
- All configurable parameters externalized: model paths, confidence threshold, ports, DB credentials, mitigation settings
- Placeholder values for safe out-of-box usage

---

### 4. ML Pipeline Code Scaffolding

All modules are fully implemented (not just stubs) with docstrings, logging, and error handling:

| Module | File | Key Classes/Functions | Lines |
|--------|------|----------------------|-------|
| Data Loader | `ml/data/data_loader.py` | `DataLoader` — loads CIC-IDS2017 & UNSW-NB15, validates datasets | ~115 |
| Preprocessing | `ml/data/preprocessing.py` | `Preprocessor` — clean, map labels, scale, split, save/load | ~195 |
| CNN-LSTM Model | `ml/models/cnn_lstm.py` | `build_cnn_lstm_model()` — config-driven architecture builder | ~155 |
| Trainer | `ml/models/trainer.py` | `Trainer` — train with class weights, evaluate, save artifacts | ~135 |
| SHAP Explainer | `ml/explainability/shap_explainer.py` | `SHAPExplainer` — DeepExplainer, top-k features, human-readable | ~100 |
| Packet Sniffer | `ml/ingestion/packet_sniffer.py` | `PacketSniffer` — Scapy capture, 5-tuple extraction, threading | ~115 |
| Flow Aggregator | `ml/ingestion/flow_aggregator.py` | `FlowAggregator` — hash table, timeout emission, 42 features | ~260 |
| Inference Pipeline | `ml/inference/pipeline.py` | `InferencePipeline` — end-to-end: flow → predict → explain → route | ~125 |

#### Key Design Decisions:
- **Label Mapping**: Both CIC-IDS2017 and UNSW-NB15 raw labels are mapped to a unified 6-class schema (`preprocessing.py`)
- **Bidirectional Flows**: Flow aggregator sorts 5-tuple to group (A→B) and (B→A) packets into the same flow
- **Config-driven Architecture**: CNN-LSTM model reads hyperparameters from YAML, no magic numbers in code
- **SHAP Optimization**: DeepExplainer used over KernelExplainer; background set capped at 100 samples for speed
- **Decision Routing**: 98% confidence threshold for auto-mitigation (configurable via .env)
- **Explain selectively**: SHAP explanations generated only for non-benign or low-confidence predictions (saves latency)

---

### 5. Backend & Frontend Scaffolding

#### Backend (`backend/`)
- `server.js`: Express + Socket.io with CORS, health check (`/health`), alert broadcasting
- Placeholder endpoints for historical queries (`/api/alerts`) and mitigation control (`/api/mitigation/toggle`)
- `package.json`: Express 4.x, Socket.io 4.x, dotenv, morgan, influx client

#### Frontend (`frontend/`)
- `App.js`: Placeholder with documented planned components (LiveAlertFeed, AlertDetail, LiveCharts, etc.)
- `package.json`: React 18, Socket.io-client, Chart.js, React Router

---

### 6. Infrastructure

| Item | File | Description |
|------|------|-------------|
| Python Dockerfile | `infra/docker/Dockerfile.inference` | Python 3.11-slim, libpcap, iptables, Scapy deps |
| Node Dockerfile | `infra/docker/Dockerfile.backend` | Node 18-slim, production deps |
| Docker Compose | `docker-compose.yml` | Full stack: inference + backend + InfluxDB 2.7 |
| Dataset Script | `infra/scripts/download_datasets.py` | Download instructions + directory creation |

#### Docker Compose Design:
- Inference engine uses `network_mode: host` + `NET_ADMIN` + `NET_RAW` capabilities (required for Scapy capture and iptables)
- InfluxDB 2.7 with auto-initialization (org, bucket, admin token)
- Frontend service commented out (Phase 5)
- All services use `.env` file for configuration

---

### 7. Testing

- `ml/tests/test_smoke.py`: Comprehensive smoke tests covering:
  - All module imports (8 classes/functions)
  - Config file validation (feature schema, model config, alert schema)
  - DataLoader instantiation and validation
  - Preprocessor label mapping integrity
  - FlowAggregator packet processing and flow emission

---

### 8. Documentation

| Document | Location | Contents |
|----------|----------|----------|
| Project README | `README.md` | Architecture diagram, project structure, tech stack, quick start, datasets, classification schema, security notice |
| Data README | `data/README.md` | Directory structure, download instructions |
| Artifacts README | `ml/models/artifacts/README.md` | Expected model files, regeneration instructions |
| Reports README | `reports/README.md` | Expected PDF output format |
| Architecture README | `docs/architecture/README.md` | Links to key schema files |
| EDA Plan | `ml/notebooks/01_eda_plan.md` | Planned EDA analysis sections |
| License | `LICENSE` | MIT with academic use notice |

---

## 📊 Summary Statistics

| Metric | Value |
|--------|-------|
| Total files created | **35** |
| Python source files | 10 (8 modules + 2 init) |
| Configuration files | 4 (feature schema, model config, alert schema, .env.example) |
| Infrastructure files | 4 (2 Dockerfiles, docker-compose, download script) |
| Documentation files | 8 (READMEs, plans, license) |
| Test files | 1 (smoke tests with 13 test cases) |
| Backend/Frontend scaffolds | 4 files |
| Total ML features defined | 42 |
| Total classification classes | 6 |
| Python dependencies | 25 packages |

---

## 🔗 Key File References

- Feature Schema: [`ml/config/feature_schema.json`](file:///d:/My%20Desktop/Materials/SEMESTERS/Project/Sentinel-AI/ml/config/feature_schema.json)
- Model Config: [`ml/config/model_config.yaml`](file:///d:/My%20Desktop/Materials/SEMESTERS/Project/Sentinel-AI/ml/config/model_config.yaml)
- Alert Schema: [`ml/config/alert_schema.json`](file:///d:/My%20Desktop/Materials/SEMESTERS/Project/Sentinel-AI/ml/config/alert_schema.json)
- CNN-LSTM Model: [`ml/models/cnn_lstm.py`](file:///d:/My%20Desktop/Materials/SEMESTERS/Project/Sentinel-AI/ml/models/cnn_lstm.py)
- Preprocessor: [`ml/data/preprocessing.py`](file:///d:/My%20Desktop/Materials/SEMESTERS/Project/Sentinel-AI/ml/data/preprocessing.py)
- Flow Aggregator: [`ml/ingestion/flow_aggregator.py`](file:///d:/My%20Desktop/Materials/SEMESTERS/Project/Sentinel-AI/ml/ingestion/flow_aggregator.py)
- Inference Pipeline: [`ml/inference/pipeline.py`](file:///d:/My%20Desktop/Materials/SEMESTERS/Project/Sentinel-AI/ml/inference/pipeline.py)

---

## ➡️ What's Next: Phase 1

Phase 1 will pick up with:
1. Setting up the Python virtual environment and installing dependencies (`pip install -r ml/requirements.txt`)
2. Downloading the CIC-IDS2017 and UNSW-NB15 datasets
3. Running Exploratory Data Analysis (EDA notebook)
4. Data cleaning and preprocessing with the scaffolded pipeline
5. Setting up the isolated VirtualBox lab environment

The scaffolding created in Phase 0 means Phase 1 can focus purely on **data and analysis** — all code structure, configurations, and architecture are already in place.
