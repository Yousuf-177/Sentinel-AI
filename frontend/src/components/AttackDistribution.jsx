import React, { useMemo } from 'react';
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
} from 'chart.js';
import { Doughnut } from 'react-chartjs-2';

ChartJS.register(ArcElement, Tooltip, Legend);

const LABEL_COLORS = {
  Benign: '#10b981',
  'DoS/DDoS': '#ef4444',
  PortScan: '#f59e0b',
  BruteForce: '#8b5cf6',
  WebAttack: '#ec4899',
  Infiltration: '#06b6d4',
};

const LABEL_ORDER = [
  'Benign',
  'DoS/DDoS',
  'PortScan',
  'BruteForce',
  'WebAttack',
  'Infiltration',
];

export default function AttackDistribution({ alerts, stats }) {
  const chartData = useMemo(() => {
    const serverCounts = stats?.by_label || stats?.label_counts;
    let counts = {};

    if (serverCounts && typeof serverCounts === 'object') {
      counts = serverCounts;
    } else {
      alerts.forEach((a) => {
        const label = a.label || 'Unknown';
        counts[label] = (counts[label] || 0) + 1;
      });
    }

    const labels = LABEL_ORDER.filter(
      (l) => counts[l] != null && counts[l] > 0
    );
    // Include any extra labels not in LABEL_ORDER
    Object.keys(counts).forEach((l) => {
      if (!labels.includes(l) && counts[l] > 0) {
        labels.push(l);
      }
    });

    const data = labels.map((l) => counts[l] || 0);
    const backgroundColor = labels.map(
      (l) => LABEL_COLORS[l] || '#64748b'
    );

    return {
      labels,
      datasets: [
        {
          data,
          backgroundColor,
          borderColor: '#1e293b',
          borderWidth: 2,
          hoverOffset: 6,
        },
      ],
    };
  }, [alerts, stats]);

  const options = useMemo(
    () => ({
      responsive: true,
      maintainAspectRatio: false,
      cutout: '60%',
      plugins: {
        legend: {
          display: true,
          position: 'right',
          labels: {
            color: '#94a3b8',
            boxWidth: 12,
            boxHeight: 12,
            padding: 12,
            font: { size: 11 },
            usePointStyle: true,
            pointStyleWidth: 10,
          },
        },
        tooltip: {
          backgroundColor: '#1e293b',
          titleColor: '#f1f5f9',
          bodyColor: '#94a3b8',
          borderColor: '#334155',
          borderWidth: 1,
          padding: 10,
          cornerRadius: 8,
          callbacks: {
            label: function (context) {
              const total = context.dataset.data.reduce(
                (a, b) => a + b,
                0
              );
              const pct =
                total > 0
                  ? ((context.parsed / total) * 100).toFixed(1)
                  : 0;
              return ` ${context.label}: ${context.parsed} (${pct}%)`;
            },
          },
        },
      },
    }),
    []
  );

  const hasData =
    chartData.datasets[0].data.length > 0 &&
    chartData.datasets[0].data.some((d) => d > 0);

  return (
    <div className="chart-card">
      <h3 className="chart-card__title">Attack Distribution</h3>
      <div className="chart-card__body chart-card__body--doughnut">
        {hasData ? (
          <Doughnut data={chartData} options={options} />
        ) : (
          <span className="text-muted">No alert data yet</span>
        )}
      </div>
    </div>
  );
}