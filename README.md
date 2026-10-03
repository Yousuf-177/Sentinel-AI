# SENTINEL-AI

### Explainable Deep Learning Framework for Real-Time Network Intrusion Detection & Autonomous Mitigation

---

## 📋 Overview

SENTINEL-AI is an end-to-end network intrusion detection and prevention system that combines deep learning classification with explainable AI (XAI) to provide real-time, human-interpretable threat detection and autonomous mitigation.

**What makes it different:** Most IDS-with-ML projects stop at classification. SENTINEL-AI goes from raw packet capture → CNN-LSTM classification → SHAP explanation → confidence-gated auto-mitigation → real-time dashboard → time-series persistence → automated PDF reporting — fully containerized and deployable.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SENTINEL-AI Pipeline                         │
├──────────────┬──────────────┬──────────────┬────────────────────────┤
│  Layer 1:    │  Layer 2:    │  Layer 3:    │  Layer 4:              │
│  INGESTION   │  INFERENCE   │  XAI         │  RESPONSE & DISPLAY   │
│              │              │              │                        │
│  Scapy       │  Preprocess  │  SHAP Deep   │  Auto-Mitigation      │
│  Sniffer     │  → CNN-LSTM  │  Explainer   │  (iptables)           │
│  → Flow      │  Classifier  │  → Top-3     │  Dashboard (React)    │
│  Aggregator  │  → Label +   │  Feature     │  InfluxDB Persistence │
│  → Features  │  Confidence  │  Explanation │  PDF Reporting        │
└──────────────┴──────────────┴──────────────┴────────────────────────┘
```

---

## 📁 Project Structure

```
Sentinel-AI/
├── ml/                          # Machine Learning pipeline
│   ├── config/                  # Feature schemas, model configs, hyperparameters
│   ├── data/                    # Data loading, preprocessing, augmentation
│   ├── models/                  # Model architectures, training, evaluation
│   ├── explainability/          # SHAP integration, explanation generation
│   ├── ingestion/               # Packet capture (Scapy), flow aggregation
│   ├── inference/               # Real-time inference bridge, decision router
│   ├── notebooks/               # Jupyter notebooks (EDA, experiments)
│   └── tests/                   # ML pipeline unit tests
│
├── backend/                     # Node.js real-time backend
│   ├── src/                     # Express + Socket.io server source
│   ├── routes/                  # API route definitions
│   ├── services/                # Business logic services
│   └── tests/                   # Backend unit tests
│
├── frontend/                    # React dashboard
│   ├── src/                     # React components, pages, hooks
│   ├── public/                  # Static assets
│   └── tests/                   # Frontend tests
│
├── infra/                       # Infrastructure & deployment
│   ├── docker/                  # Dockerfiles for each service
│   ├── scripts/                 # Utility scripts (setup, data download)
│   └── config/                  # Environment configs, docker-compose
│
├── data/                        # Dataset storage (gitignored)
│   ├── raw/                     # Original downloaded datasets
│   ├── processed/               # Cleaned, preprocessed data
│   └── interim/                 # Intermediate processing artifacts
│
├── docs/                        # Project documentation
│   ├── phases/                  # Phase completion documentation
│   └── architecture/            # Architecture diagrams and decisions
│
├── presentation/                # Presentation materials
├── reports/                     # Generated reports (PDF output)
│
├── .gitignore
├── .env.example                 # Environment variable template
├── README.md                    # This file
├── LICENSE
└── SENTINEL-AI_Project_Plan.md  # Detailed project plan
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Packet Capture | **Scapy** (Python) | Raw packet sniffing + flow aggregation |
| ML Framework | **TensorFlow / Keras** | CNN-LSTM model training & inference |
| Explainability | **SHAP** (DeepExplainer) | Per-alert feature importance explanations |
| Backend | **Node.js + Express + Socket.io** | Real-time alert streaming via WebSocket |
| Frontend | **React + Chart.js** | Live dashboard, historical analysis UI |
| Persistence | **InfluxDB** | Time-series flow & alert storage |
| Reporting | **ReportLab** (Python) | Automated PDF security audit reports |
| Mitigation | **iptables** (Linux) | Confidence-gated firewall rule insertion |
| Deployment | **Docker + Docker Compose** | Full-stack containerized deployment |
| Lab Simulation | **VirtualBox + Nmap + hping3** | Isolated attack simulation environment |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+ & npm
- Docker & Docker Compose
- VirtualBox (for lab simulation)

### Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd Sentinel-AI

# 2. Set up Python environment
python -m venv venv
source venv/bin/activate    # Linux/macOS
# or: .\venv\Scripts\activate  # Windows
pip install -r ml/requirements.txt

# 3. Set up Node.js backend
cd backend && npm install && cd ..

# 4. Set up React frontend
cd frontend && npm install && cd ..

# 5. Copy environment variables
cp .env.example .env
# Edit .env with your configuration

# 6. (Production) Run with Docker Compose
docker compose up
```

---

## 📊 Datasets

| Dataset | Source | Description |
|---------|--------|-------------|
| **CIC-IDS2017** | [UNB](https://www.unb.ca/cic/datasets/ids-2017.html) | Realistic labeled network flows, 5 days of traffic, ~2.8M flows |
| **UNSW-NB15** | [UNSW](https://research.unsw.edu.au/projects/unsw-nb15-dataset) | Hybrid real/synthetic traffic, 49 features, ~2.5M records |

> **Note:** Datasets are not included in the repository (too large). Download links and instructions are in `infra/scripts/download_datasets.sh`.

---

## 📈 Classification Schema

| Class | Description | Example Attacks |
|-------|-------------|-----------------|
| Benign | Normal/legitimate traffic | — |
| DoS / DDoS | Denial of Service | SYN flood, Slowloris, GoldenEye |
| Port Scan | Reconnaissance probing | Nmap SYN scan, FIN scan |
| Brute Force | Credential attacks | SSH brute force, FTP brute force |
| Web Attack | Application-layer attacks | SQL injection, XSS, CSRF |
| Infiltration | Advanced persistent threats | Backdoor, droppers |

---

## 🔐 Security Notice

> **⚠️ IMPORTANT:** This system is designed for **isolated lab/academic environments only**. The auto-mitigation (IPS) component modifies firewall rules. **Never** point this at production or shared networks. Always test in an isolated VirtualBox/VMware lab setup.

---

## 📝 License

This project is developed as an academic/university project. See [LICENSE](LICENSE) for details.

---

## 📞 Contact

- **Developer:** [Your Name]
- **Guide:** [Guide Name]
- **Institution:** [University / Department]
