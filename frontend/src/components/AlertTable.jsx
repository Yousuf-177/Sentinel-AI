import React, { useState, useRef, useEffect } from 'react';
import AlertDetail from './AlertDetail';

const MAX_DISPLAY = 200;

const LABEL_BADGE_CLASS = {
  Benign: 'badge--benign',
  'DoS/DDoS': 'badge--dos',
  PortScan: 'badge--portscan',
  BruteForce: 'badge--bruteforce',
  WebAttack: 'badge--webattack',
  Infiltration: 'badge--infiltration',
};

const ACTION_BADGE_CLASS = {
  ALLOW: 'badge--allow',
  ALERT: 'badge--alert',
  MITIGATE: 'badge--mitigate',
};

export default function AlertTable({ alerts }) {
  const [selectedAlert, setSelectedAlert] = useState(null);
  const [newAlertIds, setNewAlertIds] = useState(new Set());
  const prevCountRef = useRef(alerts.length);

  // Track newly arrived alerts for animation
  useEffect(() => {
    if (alerts.length > prevCountRef.current) {
      const newCount = alerts.length - prevCountRef.current;
      const ids = new Set();
      for (let i = 0; i < newCount && i < alerts.length; i++) {
        ids.add(alerts[i].alert_id);
      }
      setNewAlertIds(ids);
      // Clear animation flag after animation completes
      const timer = setTimeout(() => setNewAlertIds(new Set()), 900);
      return () => clearTimeout(timer);
    }
    prevCountRef.current = alerts.length;
  }, [alerts]);

  const displayAlerts = alerts.slice(0, MAX_DISPLAY);

  return (
    <>
      <div className="alert-table-wrapper">
        <div className="alert-table-header">
          <div className="alert-table-header__title">
            <span className="alert-table-header__live-dot" />
            Live Alert Feed
          </div>
          <span className="alert-table-header__count">
            {alerts.length} alert{alerts.length !== 1 ? 's' : ''}
          </span>
        </div>

        <div className="alert-table-scroll">
          <table className="alert-table">
            <thead>
              <tr>
                <th className="col-time">Time</th>
                <th className="col-src">Source</th>
                <th className="col-dst">Destination</th>
                <th className="col-label">Label</th>
                <th className="col-conf">Confidence</th>
                <th className="col-action">Action</th>
                <th className="col-detail">Details</th>
              </tr>
            </thead>
            <tbody>
              {displayAlerts.length === 0 ? (
                <tr>
                  <td colSpan={7} className="alert-table__empty">
                    No alerts received yet. Waiting for traffic…
                  </td>
                </tr>
              ) : (
                displayAlerts.map((alert) => (
                  <AlertRow
                    key={alert.alert_id}
                    alert={alert}
                    isNew={newAlertIds.has(alert.alert_id)}
                    onView={() => setSelectedAlert(alert)}
                  />
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {selectedAlert && (
        <AlertDetail
          alert={selectedAlert}
          onClose={() => setSelectedAlert(null)}
        />
      )}
    </>
  );
}

function AlertRow({ alert, isNew, onView }) {
  const time = formatTime(alert.timestamp);
  const src = `${alert.src_ip || '—'}:${alert.src_port ?? '—'}`;
  const dst = `${alert.dst_ip || '—'}:${alert.dst_port ?? '—'}`;
  const conf = alert.confidence != null ? alert.confidence : null;
  const confPct = conf != null ? `${(conf * 100).toFixed(1)}%` : '—';
  const confClass =
    conf != null
      ? conf >= 0.95
        ? 'conf--high'
        : conf >= 0.8
        ? 'conf--medium'
        : 'conf--low'
      : '';

  const labelClass = LABEL_BADGE_CLASS[alert.label] || '';
  const actionClass = ACTION_BADGE_CLASS[alert.action] || '';

  return (
    <tr className={isNew ? 'alert-table__row--new' : ''}>
      <td className="col-time text-mono">{time}</td>
      <td className="col-src text-mono">{src}</td>
      <td className="col-dst text-mono">{dst}</td>
      <td className="col-label">
        <span className={`badge ${labelClass}`}>{alert.label || '—'}</span>
      </td>
      <td className={`col-conf ${confClass}`}>{confPct}</td>
      <td className="col-action">
        <span className={`badge ${actionClass}`}>{alert.action || '—'}</span>
      </td>
      <td className="col-detail">
        <button className="btn" onClick={onView}>
          View
        </button>
      </td>
    </tr>
  );
}

function formatTime(ts) {
  if (!ts) return '—';
  try {
    const d = new Date(ts);
    if (isNaN(d.getTime())) return ts;
    return d.toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  } catch {
    return ts;
  }
}
