import React, { useMemo } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip);

const BIN_COUNT = 10;

function binIndex(confidence) {
  if (confidence >= 1) return BIN_COUNT - 1;
  if (confidence <= 0) return 0;
  return Math.min(BIN_COUNT - 1, Math.floor(confidence * BIN_COUNT));
}

export default function ConfidenceHistogram({ alerts }) {
  const chartData = useMemo(() => {
    const counts = new Array(BIN_COUNT).fill(0);
    alerts.forEach((alert) => {
      if (alert.confidence == null || Number.isNaN(Number(alert.confidence))) return;
      counts[binIndex(Number(alert.confidence))] += 1;
    });

    return {
      labels: counts.map((_, i) => `${i * 10}–${(i + 1) * 10}%`),
      datasets: [
        {
          label: 'Alerts',
          data: counts,
          backgroundColor: 'rgba(59, 130, 246, 0.75)',
          borderColor: '#3b82f6',
          borderWidth: 1,
          borderRadius: 4,
        },
      ],
    };
  }, [alerts]);

  const options = useMemo(
    () => ({
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#1e293b',
          titleColor: '#f1f5f9',
          bodyColor: '#94a3b8',
          borderColor: '#334155',
          borderWidth: 1,
          padding: 10,
          cornerRadius: 8,
        },
      },
      scales: {
        x: {
          grid: { color: 'rgba(51, 65, 85, 0.3)', drawBorder: false },
          ticks: {
            color: '#64748b',
            font: { size: 10 },
            maxRotation: 0,
          },
        },
        y: {
          beginAtZero: true,
          grid: { color: 'rgba(51, 65, 85, 0.3)', drawBorder: false },
          ticks: {
            color: '#64748b',
            font: { size: 10 },
            precision: 0,
          },
        },
      },
    }),
    []
  );

  const hasData = chartData.datasets[0].data.some((count) => count > 0);

  return (
    <div className="chart-card">
      <h3 className="chart-card__title">Confidence Histogram</h3>
      <div className="chart-card__body chart-card__body--histogram">
        {hasData ? (
          <Bar data={chartData} options={options} />
        ) : (
          <div className="flex-center" style={{ height: '100%' }}>
            <span className="text-muted">No alert data yet</span>
          </div>
        )}
      </div>
    </div>
  );
}