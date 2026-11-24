require('dotenv').config({ path: '../.env' });
const express = require('express');
const { createServer } = require('http');
const { Server } = require('socket.io');
const cors = require('cors');
const morgan = require('morgan');

// ─── App & Server Setup ─────────────────────────────────────────────────────

const app = express();
const httpServer = createServer(app);

const CORS_ORIGIN = process.env.CORS_ORIGIN || 'http://localhost:3000';

app.use(cors({ origin: CORS_ORIGIN }));
app.use(express.json());
app.use(morgan('dev'));

const io = new Server(httpServer, {
  cors: {
    origin: CORS_ORIGIN,
    methods: ['GET', 'POST'],
  },
});

// ─── In-Memory State ─────────────────────────────────────────────────────────

const MAX_ALERTS = 10_000;

/** @type {Array<Object>} Newest alerts are pushed to the end. */
const alertStore = [];

/** Index for O(1) lookup by alert_id. */
const alertIndex = new Map();

let connectedClients = 0;
let mitigationEnabled = true;
let lastAlertAt = null;

// ─── Helper: Compute Live Stats ─────────────────────────────────────────────

/**
 * Build a stats snapshot from the current alert store.
 * @returns {Object} Dashboard statistics payload.
 */
function computeStats() {
  const total = alertStore.length;

  const byLabel = {};
  const byAction = {};
  let confidenceSum = 0;
  let latencySum = 0;
  let latencyCount = 0;

  for (const a of alertStore) {
    // Count by label
    byLabel[a.label] = (byLabel[a.label] || 0) + 1;

    // Count by action
    byAction[a.action] = (byAction[a.action] || 0) + 1;

    // Accumulate confidence
    confidenceSum += a.confidence ?? 0;

    // Accumulate latency (guard against missing field)
    if (a.inference_latency_ms != null) {
      latencySum += a.inference_latency_ms;
      latencyCount++;
    }
  }

  // Recent rate: alerts received in the last 5 minutes, expressed as per-minute
  const fiveMinAgo = Date.now() - 5 * 60 * 1000;
  let recentCount = 0;
  // Walk from newest (end) backwards — stop early once we pass the window
  for (let i = alertStore.length - 1; i >= 0; i--) {
    const ts = new Date(alertStore[i].timestamp).getTime();
    if (ts >= fiveMinAgo) {
      recentCount++;
    } else {
      break;
    }
  }

  return {
    total_alerts: total,
    by_label: byLabel,
    by_action: byAction,
    recent_rate: parseFloat((recentCount / 5).toFixed(2)),
    avg_confidence: total > 0 ? parseFloat((confidenceSum / total).toFixed(4)) : 0,
    avg_latency_ms: latencyCount > 0 ? parseFloat((latencySum / latencyCount).toFixed(2)) : 0,
    connected_clients: connectedClients,
    uptime: parseFloat(process.uptime().toFixed(2)),
    last_alert_at: lastAlertAt,
    mitigation_enabled: mitigationEnabled,
  };
}

// ─── Helper: Build Timeline (last 60 minutes, bucketed per minute) ──────────

/**
 * Returns an array of { minute, count, attacks } objects for the last 60 min.
 */
function computeTimeline() {
  const now = new Date();
  const sixtyMinAgo = now.getTime() - 60 * 60 * 1000;

  // Initialise 60 buckets keyed by minute string
  /** @type {Map<string, {count: number, attacks: number}>} */
  const buckets = new Map();
  for (let m = 0; m < 60; m++) {
    const d = new Date(now.getTime() - m * 60 * 1000);
    d.setSeconds(0, 0);
    const key = d.toISOString().replace(/:\d{2}\.\d{3}Z$/, 'Z'); // e.g. "2026-11-08T15:00Z"
    if (!buckets.has(key)) {
      buckets.set(key, { count: 0, attacks: 0 });
    }
  }

  // Walk backwards and fill buckets
  for (let i = alertStore.length - 1; i >= 0; i--) {
    const a = alertStore[i];
    const ts = new Date(a.timestamp).getTime();
    if (ts < sixtyMinAgo) break;

    const d = new Date(ts);
    d.setSeconds(0, 0);
    const key = d.toISOString().replace(/:\d{2}\.\d{3}Z$/, 'Z');

    const bucket = buckets.get(key);
    if (bucket) {
      bucket.count++;
      if (a.label !== 'Benign') {
        bucket.attacks++;
      }
    }
  }

  // Convert to sorted array (oldest first)
  const timeline = [];
  for (const [minute, data] of buckets) {
    timeline.push({ minute, count: data.count, attacks: data.attacks });
  }
  timeline.sort((a, b) => (a.minute < b.minute ? -1 : 1));

  return timeline;
}

// ─── Helper: Store an Alert ─────────────────────────────────────────────────

/**
 * Adds an alert to the in-memory store, enforcing the cap.
 * @param {Object} alert
 */
function storeAlert(alert) {
  // Evict oldest if at capacity
  while (alertStore.length >= MAX_ALERTS) {
    const evicted = alertStore.shift();
    if (evicted) alertIndex.delete(evicted.alert_id);
  }

  alertStore.push(alert);
  if (alert.alert_id) {
    alertIndex.set(alert.alert_id, alert);
  }
  lastAlertAt = alert.timestamp || new Date().toISOString();
}

// ─── Socket.io ──────────────────────────────────────────────────────────────

io.on('connection', (socket) => {
  connectedClients++;
  console.log(`[WS] Client connected (${connectedClients} total)`);

  socket.on('new_alert', (alertData) => {
    try {
      console.log(
        `[ALERT] ${alertData.label} (${(alertData.confidence * 100).toFixed(1)}%) from ${alertData.src_ip}`
      );

      // Persist to in-memory store
      storeAlert(alertData);

      // Broadcast the alert to all connected clients
      io.emit('alert', alertData);

      // Broadcast updated stats so dashboards can live-update
      io.emit('stats_update', computeStats());
    } catch (err) {
      console.error('[WS] Error processing new_alert:', err.message);
    }
  });

  socket.on('disconnect', () => {
    connectedClients--;
    console.log(`[WS] Client disconnected (${connectedClients} total)`);
  });
});

// ─── REST: Health ───────────────────────────────────────────────────────────

app.get('/health', (_req, res) => {
  res.json({
    status: 'ok',
    uptime: parseFloat(process.uptime().toFixed(2)),
    connected_clients: connectedClients,
    timestamp: new Date().toISOString(),
  });
});

// ─── REST: Alerts ───────────────────────────────────────────────────────────

/**
 * GET /api/alerts
 * Query params: label, action, limit (default 50), offset (default 0)
 * Returns newest first.
 */
app.get('/api/alerts', (req, res) => {
  try {
    const { label, action } = req.query;
    const limit = Math.max(1, Math.min(parseInt(req.query.limit, 10) || 50, 1000));
    const offset = Math.max(0, parseInt(req.query.offset, 10) || 0);

    // Start from a reversed view (newest first)
    let filtered = [...alertStore].reverse();

    if (label) {
      filtered = filtered.filter((a) => a.label === label);
    }
    if (action) {
      filtered = filtered.filter((a) => a.action === action);
    }

    const total = filtered.length;
    const page = filtered.slice(offset, offset + limit);

    res.json({ total, limit, offset, alerts: page });
  } catch (err) {
    console.error('[API] GET /api/alerts error:', err.message);
    res.status(500).json({ error: 'Internal server error' });
  }
});

/**
 * GET /api/alerts/:id
 * Returns a single alert by alert_id.
 */
app.get('/api/alerts/:id', (req, res) => {
  const alert = alertIndex.get(req.params.id);
  if (!alert) {
    return res.status(404).json({ error: 'Alert not found' });
  }
  res.json(alert);
});

/**
 * DELETE /api/alerts
 * Clears all stored alerts (useful for testing/reset).
 */
app.delete('/api/alerts', (_req, res) => {
  const count = alertStore.length;
  alertStore.length = 0;
  alertIndex.clear();
  lastAlertAt = null;
  console.log(`[API] Cleared ${count} alerts`);
  res.json({ message: `Cleared ${count} alerts` });
});

// ─── REST: Stats ────────────────────────────────────────────────────────────

/**
 * GET /api/stats
 * Returns live dashboard statistics.
 */
app.get('/api/stats', (_req, res) => {
  res.json(computeStats());
});

/**
 * GET /api/stats/timeline
 * Returns alerts bucketed by minute for the last 60 minutes.
 */
app.get('/api/stats/timeline', (_req, res) => {
  res.json({ timeline: computeTimeline() });
});

// ─── REST: Mitigation ───────────────────────────────────────────────────────

/**
 * POST /api/mitigation/toggle
 * Body: { "enabled": true | false }
 * Stores and returns the current mitigation state.
 */
app.post('/api/mitigation/toggle', (req, res) => {
  const { enabled } = req.body;

  if (typeof enabled !== 'boolean') {
    return res.status(400).json({
      error: 'Request body must include "enabled" as a boolean.',
    });
  }

  mitigationEnabled = enabled;
  console.log(`[MITIGATION] Auto-mitigation ${mitigationEnabled ? 'ENABLED' : 'DISABLED'}`);

  res.json({ mitigation_enabled: mitigationEnabled });
});

// ─── Start Server ───────────────────────────────────────────────────────────

const PORT = process.env.BACKEND_PORT || 3001;
const HOST = process.env.BACKEND_HOST || '0.0.0.0';

httpServer.listen(PORT, HOST, () => {
  console.log(`\n========================================`);
  console.log(`  SENTINEL-AI Backend`);
  console.log(`  Listening on http://${HOST}:${PORT}`);
  console.log(`  Max in-memory alerts: ${MAX_ALERTS.toLocaleString()}`);
  console.log(`========================================\n`);
});
