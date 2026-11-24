import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { io } from 'socket.io-client';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:3001';
const MAX_ALERTS = 500;
const TIMELINE_MINUTES = 60;

function buildTimeline(alerts) {
  const now = Date.now();
  const buckets = [];
  const index = new Map();

  for (let m = TIMELINE_MINUTES - 1; m >= 0; m--) {
    const d = new Date(now - m * 60 * 1000);
    d.setSeconds(0, 0);
    const minute = d.toISOString();
    const bucket = { minute, count: 0, attacks: 0 };
    buckets.push(bucket);
    index.set(minute, bucket);
  }

  const windowStart = now - TIMELINE_MINUTES * 60 * 1000;
  for (const alert of alerts) {
    const ts = new Date(alert.timestamp).getTime();
    if (Number.isNaN(ts) || ts < windowStart) continue;
    const d = new Date(ts);
    d.setSeconds(0, 0);
    const bucket = index.get(d.toISOString());
    if (!bucket) continue;
    bucket.count += 1;
    if (alert.label && alert.label !== 'Benign') {
      bucket.attacks += 1;
    }
  }

  return buckets;
}

export default function useSocket() {
  const [isConnected, setIsConnected] = useState(false);
  const [alerts, setAlerts] = useState([]);
  const [stats, setStats] = useState(null);
  const socketRef = useRef(null);
  const alertSetRef = useRef(new Set());

  const fetchInitialData = useCallback(async () => {
    try {
      const [alertsRes, statsRes] = await Promise.all([
        fetch(`${API_URL}/api/alerts?limit=${MAX_ALERTS}`),
        fetch(`${API_URL}/api/stats`),
      ]);

      if (alertsRes.ok) {
        const data = await alertsRes.json();
        const alertList = Array.isArray(data) ? data : data.alerts || [];
        alertSetRef.current = new Set(alertList.map((a) => a.alert_id));
        setAlerts(alertList);
      }

      if (statsRes.ok) {
        const data = await statsRes.json();
        setStats(data);
      }
    } catch (err) {
      console.error('[SENTINEL] Failed to fetch initial data:', err);
    }
  }, []);

  useEffect(() => {
    fetchInitialData();

    const socket = io(API_URL, {
      transports: ['websocket', 'polling'],
      reconnectionAttempts: Infinity,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
    });
    socketRef.current = socket;

    socket.on('connect', () => {
      setIsConnected(true);
    });

    socket.on('disconnect', () => {
      setIsConnected(false);
    });

    socket.on('alert', (alert) => {
      if (!alert || !alert.alert_id) return;
      if (alertSetRef.current.has(alert.alert_id)) return;

      alertSetRef.current.add(alert.alert_id);
      setAlerts((prev) => {
        const next = [alert, ...prev];
        if (next.length > MAX_ALERTS) {
          const removed = next.slice(MAX_ALERTS);
          removed.forEach((a) => alertSetRef.current.delete(a.alert_id));
          return next.slice(0, MAX_ALERTS);
        }
        return next;
      });
    });

    socket.on('stats_update', (newStats) => {
      if (newStats) {
        setStats(newStats);
      }
    });

    return () => {
      socket.off('connect');
      socket.off('disconnect');
      socket.off('alert');
      socket.off('stats_update');
      socket.disconnect();
    };
  }, [fetchInitialData]);

  const timeline = useMemo(() => buildTimeline(alerts), [alerts]);

  return { isConnected, alerts, stats, timeline };
}