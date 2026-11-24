import React, { useEffect, useCallback } from 'react';

const LABEL_COLORS = {
  Benign: '#10b981',
  'DoS/DDoS': '#ef4444',
  PortScan: '#f59e0b',
  BruteForce: '#8b5cf6',
  WebAttack: '#ec4899',
  Infiltration: '#06b6d4',
};

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

export default function AlertDetail({ alert, onClose }) {
  const handleKeyDown = useCallback(
    (e) => {
      if (e.key === 'Escape') onClose();
    },
    [onClose]
  );

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown);
    document.body.style.overflow = 'hidden';
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = '';
    };
  }, [handleKeyDown]);

  const handleOverlayClick = (e) => {
    if (e.target === e.currentTarget) onClose();
  };

  if (!alert) return null;

  const conf = alert.confidence != null ? alert.confidence : null;
  const confPct = conf != null ? `${(conf * 100).toFixed(1)}%` : '—';
  const confColor =
    conf != null
      ? conf >= 0.95
        ? '#10b981'
        : conf >= 0.8
        ? '#f59e0b'
        : '#ef4444'
      : '#64748b';
  const labelColor = LABEL_COLORS[alert.label] || '#64748b';

  const explanation = alert.explanation;
  const topFeatures = explanation?.top_features || [];
  const maxShap =
    topFeatures.length > 0
      ? Math.max(...topFeatures.map((f) => Math.abs(f.shap_value)), 0.01)
      : 1;

  return (
    <div className="modal-overlay" onClick={handleOverlayClick}>
      <div className="modal" role="dialog" aria-modal="true">
        <button className="modal__close" onClick={onClose} aria-label="Close">
          ✕
        </button>

        <div className="modal__header">
          <h2 className="modal__title">Alert Details</h2>
          <p className="modal__subtitle">ID: {alert.alert_id || '—'}</p>
        </div>

        <div className="modal__body">
          {/* Flow visualization */}
          <div className="detail-flow">
            <span>
              {alert.src_ip || '?'}:{alert.src_port ?? '?'}
            </span>
            <span className="detail-flow__arrow">→</span>
            <span>
              {alert.dst_ip || '?'}:{alert.dst_port ?? '?'}
            </span>
          </div>

          {/* Metadata grid */}
          <div className="detail-grid">
            <div className="detail-item">
              <div className="detail-item__label">Label</div>
              <div className="detail-item__value detail-item__value--large">
                <span
                  className={`badge ${LABEL_BADGE_CLASS[alert.label] || ''}`}
                  style={{ fontSize: '1rem', padding: '4px 14px' }}
                >
                  {alert.label || '—'}
                </span>
              </div>
            </div>

            <div className="detail-item">
              <div className="detail-item__label">Confidence</div>
              <div
                className="detail-item__value detail-item__value--large"
                style={{ color: confColor }}
              >
                {confPct}
              </div>
            </div>

            <div className="detail-item">
              <div className="detail-item__label">Action</div>
              <div className="detail-item__value">
                <span
                  className={`badge ${ACTION_BADGE_CLASS[alert.action] || ''}`}
                  style={{ fontSize: '0.9rem', padding: '4px 14px' }}
                >
                  {alert.action || '—'}
                </span>
              </div>
            </div>

            <div className="detail-item">
              <div className="detail-item__label">Mitigation Status</div>
              <div className="detail-item__value" style={{ textTransform: 'capitalize' }}>
                {alert.mitigation_status || '—'}
              </div>
            </div>

            <div className="detail-item">
              <div className="detail-item__label">Timestamp</div>
              <div className="detail-item__value detail-item__value--mono">
                {alert.timestamp
                  ? new Date(alert.timestamp).toLocaleString()
                  : '—'}
              </div>
            </div>

            <div className="detail-item">
              <div className="detail-item__label">Protocol</div>
              <div className="detail-item__value">{alert.protocol || '—'}</div>
            </div>

            <div className="detail-item">
              <div className="detail-item__label">Flow ID</div>
              <div className="detail-item__value detail-item__value--mono">
                {alert.flow_id || '—'}
              </div>
            </div>

            <div className="detail-item">
              <div className="detail-item__label">Inference Latency</div>
              <div className="detail-item__value">
                {alert.inference_latency_ms != null
                  ? `${alert.inference_latency_ms.toFixed(2)} ms`
                  : '—'}
              </div>
            </div>
          </div>

          {/* SHAP Explanation */}
          <div className="explanation-section">
            <h3 className="explanation-section__title">
              🧠 XAI Explanation (SHAP)
            </h3>

            {explanation ? (
              <>
                {/* Human-readable text */}
                {explanation.human_readable && (
                  <div className="explanation-text">
                    {explanation.human_readable}
                  </div>
                )}

                {/* SHAP bar chart */}
                {topFeatures.length > 0 ? (
                  <div className="shap-bars">
                    {topFeatures.map((feat, idx) => {
                      const absVal = Math.abs(feat.shap_value);
                      const widthPct = (absVal / maxShap) * 45; // max 45% of track width
                      const isPositive = feat.shap_value >= 0;

                      return (
                        <div className="shap-bar" key={idx}>
                          <span className="shap-bar__feature" title={feat.feature}>
                            {feat.feature}
                          </span>
                          <div className="shap-bar__track">
                            <div className="shap-bar__center-line" />
                            <div
                              className={`shap-bar__fill ${
                                isPositive
                                  ? 'shap-bar__fill--positive'
                                  : 'shap-bar__fill--negative'
                              }`}
                              style={{ width: `${widthPct}%` }}
                            />
                          </div>
                          <span
                            className={`shap-bar__value ${
                              isPositive
                                ? 'shap-bar__value--positive'
                                : 'shap-bar__value--negative'
                            }`}
                          >
                            {isPositive ? '+' : ''}
                            {feat.shap_value.toFixed(3)}
                          </span>
                          <span className="shap-bar__meta">
                            {feat.pct_vs_baseline || ''}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="explanation-unavailable">
                    No feature attributions available.
                  </div>
                )}
              </>
            ) : (
              <div className="explanation-unavailable">
                Explanation not available (flow was ALLOW or SHAP queue was
                full)
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
