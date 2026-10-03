/**
 * SENTINEL-AI — Backend Server
 * 
 * Express + Socket.io server that:
 * 1. Receives alert events from the Python inference bridge (Socket.io)
 * 2. Broadcasts alerts to connected dashboard clients (WebSocket)
 * 3. Exposes REST API for historical queries and system control
 * 
 * @module server
 */

require('dotenv').config({ path: '../.env' });

const express = require('express');
const { createServer } = require('http');
const { Server } = require('socket.io');
const cors = require('cors');
const morgan = require('morgan');

const app = express();
const httpServer = createServer(app);

// -------------------- Middleware --------------------
app.use(cors({ origin: process.env.CORS_ORIGIN || 'http://localhost:3000' }));
app.use(express.json());
app.use(morgan('dev'));

// -------------------- Socket.io --------------------
const io = new Server(httpServer, {
    cors: {
        origin: process.env.CORS_ORIGIN || 'http://localhost:3000',
        methods: ['GET', 'POST'],
    },
});

// Track connected clients
let connectedClients = 0;

io.on('connection', (socket) => {
    connectedClients++;
    console.log(`[WS] Client connected (${connectedClients} total)`);

    // Listen for alerts from the Python inference bridge
    socket.on('new_alert', (alertData) => {
        console.log(`[ALERT] ${alertData.label} (${(alertData.confidence * 100).toFixed(1)}%) from ${alertData.src_ip}`);
        // Broadcast to all connected dashboard clients
        io.emit('alert', alertData);
    });

    socket.on('disconnect', () => {
        connectedClients--;
        console.log(`[WS] Client disconnected (${connectedClients} total)`);
    });
});

// -------------------- REST API --------------------

// Health check
app.get('/health', (req, res) => {
    res.json({
        status: 'ok',
        uptime: process.uptime(),
        connected_clients: connectedClients,
        timestamp: new Date().toISOString(),
    });
});

// Placeholder: Historical alerts query
app.get('/api/alerts', (req, res) => {
    // TODO (Phase 6): Query from InfluxDB/MongoDB
    res.json({ message: 'Historical alerts endpoint — not yet implemented (Phase 6)' });
});

// Placeholder: Mitigation control
app.post('/api/mitigation/toggle', (req, res) => {
    // TODO (Phase 6): Toggle auto-mitigation on/off
    res.json({ message: 'Mitigation toggle — not yet implemented (Phase 6)' });
});

// -------------------- Start Server --------------------
const PORT = process.env.BACKEND_PORT || 3001;
const HOST = process.env.BACKEND_HOST || '0.0.0.0';

httpServer.listen(PORT, HOST, () => {
    console.log(`\n========================================`);
    console.log(`  SENTINEL-AI Backend`);
    console.log(`  Listening on http://${HOST}:${PORT}`);
    console.log(`  Health: http://${HOST}:${PORT}/health`);
    console.log(`========================================\n`);
});
