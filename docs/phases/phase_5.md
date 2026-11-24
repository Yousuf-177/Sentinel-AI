# Phase 5 — Real-time dashboard

**Status:** Complete  
**Date:** 24 November 2026

Phase 5 shows live classified alerts in the browser. The Node server was already receiving `new_alert` and rebroadcasting `alert`. This phase mounts the React shell on that stream. Firewall rules, the time-series store, and PDF reports stay in later phases.

## Run it

Three processes, from the project root.

Backend:

```powershell
cd backend
npm start
```

Dashboard (another terminal):

```powershell
cd frontend
npm start
```

The UI is [http://localhost:3000](http://localhost:3000). The API and Socket.io server are `http://localhost:3001`. Override the API origin with `REACT_APP_API_URL` if the backend is not on that host.

Inference, with emit turned on (another terminal, from the project root):

```powershell
.\venv\Scripts\Activate.ps1
$env:OPENBLAS_NUM_THREADS = "1"
$env:TF_CPP_MIN_LOG_LEVEL = "2"
python -m ml.inference.service --pcap data/interim/pcaps/syn_scan.pcap --emit
```

`--emit` sends each alert to `http://127.0.0.1:3001` as `new_alert`. Without it, the service only prints and the dashboard stays empty.

## What the dashboard does

- **Live Feed** lists alerts as they arrive (label, confidence, source, destination, action). A row opens the SHAP detail. `explanation: null` shows the existing “not available” state.
- **Charts** update from the same list: attack-type doughnut (`by_label` on the stats payload, otherwise the alerts in memory), alerts over the last 60 minutes, and a confidence histogram in 10% bins.
- The 60-minute line is rebuilt in the browser from the alerts the client holds, so it moves when a new `alert` event arrives. The client keeps at most 500 alerts.
- Stat cards read `by_label`, `by_action.MITIGATE`, and `avg_latency_ms` from `stats_update`.
- **Historical** and **Settings** are placeholders. The Settings screen shows the server mitigation flag and does not call `POST /api/mitigation/toggle`. Auto-block stays Phase 6, because the Python service does not read that flag yet.

## Contract

The payload is the object in [docs/architecture/alert_schema.md](../architecture/alert_schema.md). Direction is unchanged: Python `new_alert` → Node log, in-memory store, and `alert` / `stats_update` to the browser.