# Project Plan: SENTINEL-AI
### Explainable Deep Learning Framework for Real-Time Network Intrusion Detection & Autonomous Mitigation

**Goal**: A working, demo-ready, end-to-end system — live packet capture → CNN-LSTM classification → SHAP explanation → confidence-gated auto-mitigation → real-time dashboard → time-series persistence → automated PDF reporting — fully containerized and deployable, not just a Jupyter notebook with a model.

**Timeline**: 8 weeks (2 months), solo developer, full-time effort assumed
**Team**: You (Developer / ML Engineer / DevOps — one person, all roles)
**Constraints assumed** (stated explicitly since none were given): no budget for paid infra (open-source/free-tier stack only, as already specified in the synopsis); development happens on a personal machine or a single cloud VM; a GPU (local or free-tier Colab/Kaggle) is available for model training; live capture is tested in an isolated VM lab, not a production network.

---

## 0. Assumptions & Scope Decisions (read this first)

Before the WBS: this synopsis describes an ambitious 4-layer system (ingestion, DL inference, XAI, full-stack real-time web + IPS + persistence + reporting). Solo, in 8 weeks, **all of it is buildable at demo/academic fidelity, but not at production-grade robustness simultaneously.** The plan below is sequenced so that:

- A **fully working, demonstrable end-to-end pipeline exists by end of Week 6** (this is your safety net — everything after is hardening, not new capability).
- Weeks 7–8 are **integration, production packaging, and buffer** — deliberately not packed with new features, because solo projects that leave integration to the last week almost always slip.
- If you fall behind, the **cut list** (Section 8) tells you exactly what to drop first without losing the "end-to-end" story for your evaluation/demo.

**Definition of Done** for the whole project:
1. Live or lab-simulated traffic is captured and converted to flow features in real time.
2. The trained CNN-LSTM classifies flows into the 6 categories with reported accuracy/precision/recall/F1/FPR on a held-out test set.
3. Every alert on the dashboard shows a SHAP-based, human-readable explanation.
4. High-confidence malicious flows trigger an automatic (togglable) firewall block, visible on the dashboard.
5. All classified flows are queryable historically from a time-series store.
6. A weekly PDF security audit report is auto-generated.
7. The whole stack runs via `docker compose up` on a clean machine, with a README that lets a stranger reproduce it.

---

## Milestones

| # | Milestone | Target | Success Criteria |
|---|-----------|--------|-------------------|
| M1 | Data & environment ready | End Wk 1 | Datasets downloaded, cleaned, EDA done; dev environment + repo scaffolded and reproducible |
| M2 | Model v1 trained & evaluated | End Wk 2 | CNN-LSTM trained on CIC-IDS2017/UNSW-NB15; accuracy/precision/recall/F1/FPR reported; model artifact saved |
| M3 | XAI + live ingestion working | End Wk 3 | SHAP explanations generated for sample predictions; Scapy sniffer + flow aggregator emitting real feature vectors |
| M4 | Inference pipeline live end-to-end | End Wk 4 | Captured packet → flow → model prediction → SHAP explanation, running as one pipeline, printed/logged (no UI yet) |
| M5 | Dashboard streaming live alerts | End Wk 5 | Node/Socket.io backend + React/Next.js dashboard show live classified alerts with charts, sub-second latency |
| M6 | Full defensive loop + persistence | End Wk 6 | Auto-mitigation (iptables) with manual override toggle; all flows persisted to InfluxDB/MongoDB; historical query tab works |
| M7 | Reporting + integrated system test | End Wk 7 | ReportLab weekly PDF report auto-generates; full system tested against simulated attacks (hping3/Nmap) in isolated VM lab |
| M8 | Production package & submission-ready | End Wk 8 | Dockerized, deployed, documented, demo video/script ready, final report/synopsis-to-thesis writeup complete |

---

## Dependencies Map

```
Dataset Prep ──> Model Training ──> Model Evaluation ──> SHAP Integration ──┐
                                                                              │
Scapy Sniffer ──> Flow Aggregator ──> Feature Vector Schema ─────────────────┼──> Inference Bridge (Python)
                                                                              │
                                                                              v
                                              Node.js/Socket.io Backend ──> React/Next.js Dashboard
                                                        │
                                                        ├──> Auto-Mitigation (iptables)
                                                        ├──> Time-Series Persistence (InfluxDB/Mongo) ──> Historical Query Tab
                                                        └──> ReportLab PDF Reporting
                                                                              │
                                                                              v
                                              Integration Testing (simulated attacks, VM lab)
                                                                              │
                                                                              v
                                              Dockerization ──> Deployment ──> Docs & Demo Prep
```

**Critical path**: Dataset Prep → Model Training/Eval → SHAP → Inference Bridge → Backend → Dashboard → Integration Test → Deployment. Everything else (mitigation, persistence, reporting) can be built in parallel *once the Inference Bridge (M4) exists*, since they all just consume its output stream.

---

## Phase 1: Foundations — Environment, Data, EDA (Week 1)

| Task | Effort | Depends On | Done Criteria |
|------|--------|------------|----------------|
| Set up repo structure (mono-repo: `/ml`, `/backend`, `/frontend`, `/infra`), Git, `.gitignore`, README skeleton | 3h | – | Repo pushed, structure documented |
| Set up Python env (3.11, venv/conda), install Scapy/PyShark, TensorFlow/Keras, SHAP, Pandas, NumPy, Scikit-learn | 2h | Repo | `pip freeze` reproducible; smoke-test imports pass |
| Set up isolated lab environment: VirtualBox/VMware VM(s) for safe attack simulation, install Nmap/hping3 on attacker VM | 6h | – | Attacker VM can ping/scan a target VM; capture works via Scapy on host/VM |
| Download & inspect CIC-IDS2017 and UNSW-NB15 datasets | 3h | Python env | Both datasets on disk, loadable in Pandas |
| Exploratory Data Analysis: class distribution, feature ranges, missing/inf values, correlation check | 6h | Datasets | EDA notebook with class-imbalance stats, feature summary documented |
| Data cleaning & preprocessing pipeline: handle inf/NaN, encode labels to the 6-class schema, train/val/test split | 6h | EDA | Clean, versioned dataset artifact (e.g. parquet) with reproducible split |
| Decide & document final feature set (40+ statistical features matching CICFlowMeter schema) | 3h | EDA | `feature_schema.json` — single source of truth used by both training and the live flow aggregator |

**Week 1 total**: ~29h — leaves buffer for setup friction (VM networking is the usual time-sink here).

---

## Phase 2: Model Development (Week 2)

| Task | Effort | Depends On | Done Criteria |
|------|--------|------------|----------------|
| Build data loader / class-imbalance handling (SMOTE or class-weighted loss) | 4h | Phase 1 clean data | Balanced batches confirmed via sanity check |
| Implement 1D-CNN + LSTM hybrid architecture in TensorFlow/Keras | 6h | Data loader | Model compiles, trains one epoch without error |
| Train baseline model, tune hyperparameters (batch size, learning rate, layers) | 10h | Architecture | Training curves logged (TensorBoard/plots), no divergence |
| Evaluate on held-out test set: Accuracy, Precision, Recall, F1, False Positive Rate per class | 4h | Trained model | Metrics table produced; Recall on attack classes specifically reported |
| Error analysis: confusion matrix, worst-performing class investigation | 3h | Evaluation | Written notes on failure modes (informs XAI narrative later) |
| Save model artifact + versioned preprocessing pipeline (scaler, encoders) | 2h | Final model | `model.h5`/SavedModel + `preprocessor.pkl` saved together, loadable standalone |
| Write model card / methodology section for report (architecture, dataset, metrics) | 3h | Evaluation | Draft section ready to paste into final report |

**Week 2 total**: ~32h. **This is the highest-risk week** — if the model underperforms, you need slack here, not in Week 7. Timebox tuning; a "good enough" model (e.g., >90% macro F1) that ships beats a perfect model that eats three weeks.

---

## Phase 3: XAI Integration + Live Ingestion Pipeline (Week 3)

| Task | Effort | Depends On | Done Criteria |
|------|--------|------------|----------------|
| Integrate SHAP (DeepExplainer for the NN, or KernelExplainer as fallback) against the saved model | 6h | Model artifact | `shap_values` computed for a batch of test flows without error |
| Build `ExplainAlert` function: rank features by \|SHAP value\|, top-3, human-readable string templating | 5h | SHAP working | For 10 sample alerts, output reads like "SYN Flag Count: +400% vs baseline" |
| Benchmark SHAP latency per flow; if too slow for real-time, cache a background dataset / reduce explainer scope | 3h | ExplainAlert | Per-flow explanation time measured and documented (this determines dashboard latency budget) |
| Build Scapy-based packet sniffer module | 5h | Lab VM ready | Sniffer captures packets on target interface, prints 5-tuple + flags |
| Implement Flow Aggregation algorithm (from synopsis §2.2): hash-based flow table, timeout-based emission | 8h | Sniffer | Given a pcap or live capture, correctly emits feature vectors matching `feature_schema.json` |
| Unit-test flow aggregator against a known pcap with expected flow counts | 3h | Aggregator | Test passes on at least one benign + one attack pcap (e.g. from CIC-IDS2017 raw pcaps if available, else self-captured) |

**Week 3 total**: ~30h.

---

## Phase 4: Inference Bridge — Connecting Capture to Classification (Week 4)

| Task | Effort | Depends On | Done Criteria |
|------|--------|------------|----------------|
| Wire flow aggregator output → preprocessing pipeline → model.predict() → SHAP explain, as one Python service | 8h | Phase 2 + 3 | Live captured traffic produces `{label, confidence, explanation}` printed to console in real time |
| Implement Decision Router (from synopsis §2.5 logic, without firewall action yet — just routing/logging) | 3h | Inference service running | Flows correctly routed to "Alert Only" vs "Would-Mitigate" branch based on 98% threshold |
| Load-test the pipeline with simulated traffic (hping3 flood) to check for backlog/drops | 4h | Router | Pipeline sustains simulated attack traffic without unbounded queue growth; latency measured end-to-end |
| Set up Python → Node bridge (`socket.io-client` in Python or Flask-SocketIO), define event schema (JSON payload for alerts) | 6h | Inference pipeline stable | Python process emits a well-formed alert JSON object over a socket, ready for a Node listener |
| Document the alert JSON schema (this is the contract the whole rest of the stack depends on) | 2h | Bridge | `alert_schema.md`/JSON Schema file committed — freezes the interface so backend/frontend work can start in parallel next week |

**Week 4 total**: ~23h — lighter week deliberately, since this is the riskiest integration point (multi-process, multi-language); build in slack.

> **This is your true midpoint checkpoint.** By end of Week 4 you should be able to run a simulated attack in the lab VM and see correctly-labeled, explained alerts in a terminal. If this isn't working, everything downstream (dashboard, mitigation, reporting) has nothing to consume — treat slippage here as the top priority to fix before moving on.

---

## Phase 5: Real-Time Backend & Dashboard (Week 5)

| Task | Effort | Depends On | Done Criteria |
|------|--------|------------|----------------|
| Scaffold Node.js + Express + Socket.io backend; receive alert events from the Python bridge | 5h | Alert schema frozen | Node server logs incoming alerts from Python in real time |
| Define WebSocket API from Node backend → frontend clients (broadcast alert events) | 3h | Node backend | A minimal HTML test page receives live events via `socket.io-client` |
| Scaffold React/Next.js app, TailwindCSS setup, base layout (Live Feed / Historical Analysis / Settings tabs) | 5h | – (can start in parallel earlier) | App shell runs, navigable, styled |
| Build Live Alert Feed component: streaming table/list of alerts with label, confidence, source IP, explanation | 6h | WebSocket API | New alerts appear in the UI within ~1s of emission, no manual refresh |
| Integrate Chart.js: live-updating charts (attack-type distribution, alerts-over-time, confidence histogram) | 6h | Live feed working | At least 2 charts update live as alerts stream in |
| Alert detail view: click an alert → see full SHAP explanation, raw feature values | 4h | Live feed | Detail panel/modal shows top-3 SHAP features per alert |
| Basic responsive/UI polish pass (layout, color coding by severity, loading states) | 4h | All components above | Dashboard usable on a laptop screen without visual breakage |

**Week 5 total**: ~33h.

---

## Phase 6: Active Defense, Persistence & Historical Analysis (Week 6)

| Task | Effort | Depends On | Done Criteria |
|------|--------|------------|----------------|
| Implement `AutoMitigate` (synopsis §2.5): confidence-gated iptables rule insertion via Python `subprocess` | 6h | Inference bridge | Confirmed: a simulated malicious flow from attacker VM gets its IP dropped, verified via `iptables -L` |
| Add manual override toggle (auto-block on/off) exposed via API + dashboard UI control | 4h | AutoMitigate + backend | Toggling the dashboard switch changes live mitigation behavior |
| Safety mechanism: block-list review + one-click "unblock" (critical — you *will* accidentally block yourself/your own VM during testing) | 4h | AutoMitigate | Unblock action removes the iptables rule and confirms in UI |
| Set up InfluxDB or MongoDB (time-series collection) via Docker container | 3h | – | DB reachable locally, schema/bucket created |
| Persist every classified flow (+ explanation + mitigation status) asynchronously to the DB | 5h | DB running, Node backend | Flows visible in DB after a test run; write does not block the real-time alert path |
| Build Historical Analysis tab: query DB by time range / attack type / source IP, render as table + chart | 8h | Persistence working | Historical queries return correct filtered results in the UI |

**Week 6 total**: ~30h. **This is your M6 checkpoint — the full IDS+IPS loop with persistence is now complete.** Everything after this point is reporting, integration hardening, and productionization, not new core functionality.

---

## Phase 7: Reporting & Full Integration Testing (Week 7)

| Task | Effort | Depends On | Done Criteria |
|------|--------|------------|----------------|
| Implement ReportLab weekly PDF report generator: pulls from time-series DB, summarizes alert counts, top attackers, mitigation actions | 8h | Persistence working | Manually-triggered run produces a readable PDF with real data |
| Schedule the report job (cron / node-schedule / simple Python scheduler) | 2h | Report generator | Report regenerates automatically on schedule without manual trigger |
| End-to-end integration test #1: run full stack, launch simulated Nmap port scan from attacker VM, verify full path (capture → classify → explain → alert on dashboard → persisted → appears in report) | 4h | All prior phases | Full path verified with screenshots/logs for the report |
| End-to-end integration test #2: simulated DoS/DDoS flood (hping3), verify auto-mitigation triggers correctly and doesn't crash the pipeline under load | 4h | Test #1 | Auto-block fires at the right confidence threshold; system remains responsive |
| End-to-end integration test #3: benign traffic baseline — confirm false-positive rate is reasonable, no legitimate traffic gets auto-blocked in test run | 3h | Test #1 | FPR on live/simulated benign traffic documented |
| Bug-fixing pass from integration testing | 8h | Tests #1–3 | All blocking bugs found during integration resolved |
| Performance/latency measurement end-to-end (capture → dashboard render), documented for the report's evaluation section | 3h | Stable system | Latency numbers recorded (this satisfies your stated evaluation criteria) |

**Week 7 total**: ~32h.

---

## Phase 8: Production Packaging, Deployment & Documentation (Week 8)

| Task | Effort | Depends On | Done Criteria |
|------|--------|------------|----------------|
| Write Dockerfiles for each service (Python inference, Node backend, React frontend, DB) | 6h | Stable integrated system | Each service builds and runs individually in a container |
| Write `docker-compose.yml` wiring all services + networking + volumes + env vars | 5h | Individual Dockerfiles | `docker compose up` brings up the entire stack from a clean checkout |
| Externalize configuration (thresholds, DB creds, ports) into `.env` / config files, remove hardcoded values | 3h | Compose file | No secrets or magic numbers in source; `.env.example` provided |
| Add basic health checks / restart policies to compose services | 2h | Compose working | A killed container auto-recovers; `docker compose ps` shows healthy services |
| Security pass: least-privilege for the mitigation subprocess (sudoers scoping if Linux), input validation on API endpoints, rate-limit the WebSocket if needed | 5h | Full stack running | Documented in report as "security considerations"; no obvious open holes for a demo environment |
| Set up basic logging across services (structured logs, log rotation or at least log files) | 3h | Full stack | Logs from all 3 layers accessible in one place (e.g. `docker compose logs`) |
| Write full README: architecture diagram, setup instructions, how to run the lab simulation, how to reproduce model training | 5h | Everything above | A fresh clone + README alone gets someone to a running demo |
| Write/finalize project report sections not yet done (results, limitations, future work) using Week 2/7 draft notes | 6h | Evaluation + integration results | Report complete and consistent with what was actually built |
| Prepare demo script + record a backup demo video (in case live demo fails during evaluation) | 4h | Working deployed system | 5–8 min video showing the full attack → detect → explain → mitigate → report loop |
| Final buffer / contingency for slipped tasks from Weeks 1–7 | 8h | – | Reserved, not pre-assigned |

**Week 8 total**: ~47h (includes deliberate buffer).

---

## 8-Week Effort Summary

| Week | Phase | Hours |
|------|-------|-------|
| 1 | Foundations, data, EDA | ~29h |
| 2 | Model development | ~32h |
| 3 | XAI + live ingestion | ~30h |
| 4 | Inference bridge | ~23h |
| 5 | Backend + dashboard | ~33h |
| 6 | Mitigation, persistence, historical UI | ~30h |
| 7 | Reporting + integration testing | ~32h |
| 8 | Production packaging, docs, demo | ~47h |
| **Total** | | **~256h** (~32h/week average — realistic for a solo academic project with buffer) |

This assumes roughly 4.5–6 hours of focused work per day, 6 days/week. If your actual availability is lower (e.g. alongside coursework/job), stretch the same phase structure across 10–12 weeks rather than compressing scope in Weeks 1–6, since those weeks build the model and core pipeline your entire evaluation depends on.

---

## Risks & Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| CNN-LSTM underperforms (low recall on attack classes) | High | Medium | Timebox tuning to Week 2; fall back to a simpler, well-understood baseline (e.g. Random Forest) as a documented comparison point if DL underperforms — still satisfies the evaluation objective |
| SHAP is too slow for real-time (KernelExplainer especially) | Medium | High | Benchmark early (Week 3); switch to DeepExplainer or precompute/cache a small background set; if still slow, explain only high-confidence/attack alerts, not every benign flow |
| Live packet capture requires root/admin and is blocked or unreliable in your dev environment (esp. cloud VMs, WSL) | High | Medium | Confirmed lab VM setup in Week 1 specifically to de-risk this before it blocks later phases; always have offline pcap replay as a fallback demo path |
| Auto-mitigation accidentally blocks your own dev machine / VM host | Medium | High (it will happen at least once) | Manual override + one-click unblock built in Week 6; always test mitigation against the isolated attacker VM's IP, never your host, during development |
| Multi-language, multi-process integration (Python ↔ Node ↔ React ↔ DB) breaks in ways hard to debug solo | High | Medium | Alert JSON schema frozen at end of Week 4 as a hard contract; each layer tested in isolation before wiring together; integration testing gets its own dedicated week (7), not squeezed at the end |
| Scope creep — synopsis lists many "nice to have" extensions (SDN flow tables, GNNs, LIME as secondary) | Medium | High | Explicitly out of scope for the 8-week build (see Section 8, Cut List); note them as "future work" in the report — this is normal and expected for a minor project |
| Solo developer illness/unavailability for several days | Medium | Medium | Week 8 buffer (8h) plus the generally conservative per-task estimates absorb ~1 lost week cumulative; if more is lost, cut from Section 8's list, don't cut integration testing |
| Dataset class imbalance causes misleading accuracy | Medium | Medium | SMOTE/class-weighted loss planned explicitly in Week 2; macro F1 and per-class recall reported, not just overall accuracy |
| Running a real IPS (auto firewall blocking) is inherently sensitive/risky if pointed at anything beyond the lab VM | High | Low (if scope respected) | Scope is explicitly the isolated lab network per the synopsis (§1.4) — do not run auto-mitigation against any shared/production network |

---

## Weekly Checkpoints (solo cadence)

- **Daily**: 10-minute self-standup — what got done yesterday, what's today's single most important task, any blocker.
- **End of each week**: Compare actual progress against that week's Milestone (M1–M8). If behind by more than ~1 day, consult the Cut List (Section 8) before letting slippage cascade into the next week.
- **End of Week 4**: Hard checkpoint — full pipeline must produce a labeled, explained alert from simulated traffic in a terminal. This is the point to escalate (seek help, cut scope) if not achieved.
- **End of Week 6**: Hard checkpoint — full IDS+IPS+persistence loop working. From here, no new core features, only integration/hardening/docs.

---

## 8. Scope Cut List (use only if behind schedule)

If time runs short, cut in this order — each item can be dropped or simplified without breaking the "end-to-end working system" story:

1. **LIME as secondary XAI method** — synopsis lists SHAP as primary; LIME was always optional. Drop first.
2. **Windows Firewall API support** — implement iptables only (Linux lab VM); mention Windows as "future work."
3. **Automated report scheduling** — keep the PDF generator but trigger it manually/on-demand instead of via cron.
4. **UI polish pass** (Week 5's last task) — a functional-but-plain dashboard still satisfies "complete user interface"; visual refinement is the first UI item to cut.
5. **MongoDB *and* InfluxDB** — pick one (InfluxDB is purpose-built for time-series and less setup overhead) rather than supporting both as the synopsis's "or" implies.
6. **CNN+LSTM hybrid → CNN-only or LSTM-only** — if training instability eats too much of Week 2, a single-architecture model with the comparison documented as a limitation is an acceptable fallback; the hybrid is what the synopsis proposes, but a working simpler model beats a broken complex one.
7. **Full multi-VM lab network** — a single attacker + single target VM pair is sufficient; don't build a larger simulated topology.

**Never cut**: the SHAP explanation requirement, the confidence-gated auto-mitigation with override, the live dashboard, the Week 7 integration testing week, or the Dockerized deployment — these are what make this an "end-to-end" system rather than a model in a notebook, which is explicitly your stated goal.

---

## 9. Production Planning (End-to-End Delivery)

Since the goal is an end-to-end deliverable, not just a working demo, treat these as first-class deliverables alongside the code:

### 9.1 Deployment Architecture
- **Containerization**: one container per service — `inference-engine` (Python: sniffer + flow aggregator + model + SHAP + mitigation), `backend` (Node/Express/Socket.io), `frontend` (React/Next.js, served via Nginx in production build), `timeseries-db` (InfluxDB/MongoDB), optionally `reporting` as a scheduled job container.
- **Networking**: internal Docker network for service-to-service traffic; only the frontend/Nginx port exposed externally. The inference engine needs host network mode or elevated capabilities (`NET_ADMIN`, `NET_RAW`) to sniff packets and manage iptables — document this clearly as a deliberate, scoped exception, not a general security posture.
- **Orchestration**: `docker-compose.yml` is sufficient at this scale; Kubernetes would be over-engineering for a solo minor project and isn't recommended here.

### 9.2 Configuration & Secrets
- All thresholds (98% confidence gate, flow timeout `T`), ports, and DB credentials in `.env`, never hardcoded.
- Provide `.env.example` with safe placeholder values so the repo is usable out of the box for evaluators.

### 9.3 Security Hardening (for the demo/production package itself)
- Scope the mitigation subprocess's privileges as narrowly as possible (e.g., a sudoers rule limited to the specific `iptables` command pattern, rather than full root).
- Validate/sanitize any user input on API endpoints (especially anything that reaches a `subprocess` call — command injection is a real risk here given the firewall integration).
- Rate-limit or debounce the auto-mitigation action to prevent a flapping/oscillating block-unblock loop.
- Document clearly, in the report and README, that this is a lab/academic-scope IPS and explicitly not intended to be pointed at production or shared networks.

### 9.4 Observability
- Centralized logs per container (`docker compose logs -f`), structured JSON logging in the Python and Node services for easier debugging during the demo.
- A simple `/health` endpoint on the Node backend for a quick "is everything up" check before a live demo.

### 9.5 Documentation Deliverables
- **README**: architecture diagram (from the synopsis §2.1, rendered cleanly), setup/run instructions, how to reproduce model training, how to run the lab attack simulation safely.
- **Architecture & data-flow diagrams**: for the final report/thesis document (can be built from the Dependencies Map above).
- **Model card**: dataset, architecture, metrics, known limitations (from Phase 2).
- **Demo script + backup video**: rehearsed walkthrough plus a recorded fallback in case live packet capture or the VM lab misbehaves during evaluation — this is the single highest-value risk mitigation for a live technical demo.

### 9.6 Post-Project Maintenance Plan (even for an academic deliverable, worth stating in the report)
- Note versioning strategy for the model (retraining cadence if deployed longer-term).
- Note how new attack categories would be added (retrain + update the label schema + update SHAP background set).
- Note what would need to change to move from "lab-scope IPS" to anything closer to production (proper RBAC, audit logging to an immutable store, human-in-the-loop approval workflow for blocks instead of full autonomy, testing against a much larger and more diverse traffic sample).

---

## 10. Resource Allocation (Solo)

| Role (all you) | Hours/Week | Primary Focus by Week |
|-----------------|------------|------------------------|
| ML Engineer | ~35h in Wks 1–3 | Data, model, XAI |
| Backend/Systems Engineer | ~30h in Wks 4, 6 | Ingestion bridge, mitigation, persistence |
| Full-Stack/Frontend Engineer | ~35h in Wk 5 | Node backend, React dashboard |
| Integration/QA + DevOps | ~35h in Wks 7–8 | Testing, Docker, deployment, docs |

---

## Success Metrics (for evaluation/report)

- Model: Accuracy, Precision, Recall, F1-score, False Positive Rate (overall + per-class), with particular emphasis on attack-class Recall.
- System: end-to-end detection latency (capture → dashboard render), measured and reported.
- XAI: qualitative assessment of explanation fidelity/usefulness (a few worked examples in the report).
- IPS: correct auto-block rate at the 98% threshold, false-block rate on benign simulated traffic, override functionality verified.
- Delivery: `docker compose up` reproducibility on a clean machine — this is your strongest "end-to-end" evidence for evaluators.
