import React from 'react';

export default function StatsCards({ stats, alerts }) {
  // Compute stats from alerts if stats object is not available
  const totalAlerts = stats?.total_alerts ?? alerts.length;

  const attackAlerts = stats?.by_label
    ? Object.entries(stats.by_label).reduce(
        (sum, [label, count]) => (label !== 'Benign' ? sum + count : sum),
        0
      )
    : alerts.filter((a) => a.label && a.label !== 'Benign').length;

  const mitigations = stats?.by_action
    ? stats.by_action.MITIGATE || 0
    : alerts.filter((a) => a.action === 'MITIGATE').length;

  const avgLatency =
    stats?.avg_latency_ms ??
    (() => {
      const withLatency = alerts.filter(
        (a) => a.inference_latency_ms != null
      );
      if (withLatency.length === 0) return 0;
      const sum = withLatency.reduce(
        (acc, a) => acc + a.inference_latency_ms,
        0
      );
      return sum / withLatency.length;
    })();

  const cards = [
    {
      icon: '📊',
      label: 'Total Alerts',
      value: formatNumber(totalAlerts),
      color: 'blue',
    },
    {
      icon: '🔴',
      label: 'Attack Alerts',
      value: formatNumber(attackAlerts),
      color: 'red',
    },
    {
      icon: '🛡️',
      label: 'Mitigations',
      value: formatNumber(mitigations),
      color: 'purple',
    },
    {
      icon: '⚡',
      label: 'Avg Latency',
      value: `${avgLatency.toFixed(1)}ms`,
      color: 'green',
    },
  ];

  return (
    <div className="stats-row">
      {cards.map((card) => (
        <div key={card.label} className="stat-card">
          <div className="stat-card__header">
            <span className={`stat-card__icon stat-card__icon--${card.color}`}>
              {card.icon}
            </span>
          </div>
          <span className={`stat-card__value stat-card__value--${card.color}`}>
            {card.value}
          </span>
          <span className="stat-card__label">{card.label}</span>
        </div>
      ))}
    </div>
  );
}

function formatNumber(n) {
  if (n == null) return '0';
  if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`;
  if (n >= 1000) return `${(n / 1000).toFixed(1)}K`;
  return String(n);
}