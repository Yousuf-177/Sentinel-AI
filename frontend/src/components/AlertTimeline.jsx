import React, { useMemo } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend
);

export default function AlertTimeline({ timeline }) {
  const chartData = useMemo(() => {
    if (!timeline || timeline.length === 0) {
      // Generate empty 60-min timeline
      const labels = [];
      const now = new Date();
      for (let i = 59; i >= 0; i--) {
        const t = new Date(now.getTime() - i * 60000);
        labels.push(
          t.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        );
      }
      return {
        labels,
        datasets: [
          {
            label: 'Total Alerts',
            data: new Array(60).fill(0),
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            fill: true,
            tension: 0.4,
            pointRadius: 0,
            pointHoverRadius: 4,
            borderWidth: 2,
          },
          {
            label: 'Attack Alerts',
            data: new Array(60).fill(0),
            borderColor: '#ef4444',
            backgroundColor: 'rgba(239, 68, 68, 0.08)',
            fill: true,
            tension: 0.4,
            pointRadius: 0,
            pointHoverRadius: 4,
            borderWidth: 2,
          },
        ],
      };
    }

    const labels = timeline.map((t) => {
      if (t.minute || t.timestamp || t.time) {
        const d = new Date(t.minute || t.timestamp || t.time);
        return isNaN(d.getTime())
          ? t.minute || t.timestamp || t.time
          : d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      }
      return '';
    });

    const totalData = timeline.map(
      (t) => t.total ?? t.count ?? t.total_alerts ?? 0
    );
    const attackData = timeline.map(
      (t) => t.attacks ?? t.attack_count ?? t.attack_alerts ?? 0
    );

    return {
      labels,
      datasets: [
        {
          label: 'Total Alerts',
          data: totalData,
          borderColor: '#3b82f6',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          fill: true,
          tension: 0.4,
          pointRadius: 0,
          pointHoverRadius: 4,
          borderWidth: 2,
        },
        {
          label: 'Attack Alerts',
          data: attackData,
          borderColor: '#ef4444',
          backgroundColor: 'rgba(239, 68, 68, 0.08)',
          fill: true,
          tension: 0.4,
          pointRadius: 0,
          pointHoverRadius: 4,
          borderWidth: 2,
        },
      ],
    };
  }, [timeline]);

  const options = useMemo(
    () => ({
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'index',
        intersect: false,
      },
      plugins: {
        legend: {
          display: true,
          position: 'top',
          align: 'end',
          labels: {
            color: '#94a3b8',
            boxWidth: 12,
            boxHeight: 3,
            padding: 16,
            font: { size: 11 },
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
        },
      },
      scales: {
        x: {
          grid: { color: 'rgba(51, 65, 85, 0.3)', drawBorder: false },
          ticks: {
            color: '#64748b',
            font: { size: 10 },
            maxRotation: 0,
            maxTicksLimit: 12,
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

  return (
    <div className="chart-card">
      <h3 className="chart-card__title">Alert Timeline (Last 60 min)</h3>
      <div className="chart-card__body chart-card__body--timeline">
        <Line data={chartData} options={options} />
      </div>
    </div>
  );
}
