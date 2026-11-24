import React, { useState } from 'react';
import useSocket from './hooks/useSocket';
import StatsCards from './components/StatsCards';
import AlertTimeline from './components/AlertTimeline';
import AttackDistribution from './components/AttackDistribution';
import ConfidenceHistogram from './components/ConfidenceHistogram';
import AlertTable from './components/AlertTable';

const TABS = [
  { id: 'live', label: 'Live Feed' },
  { id: 'historical', label: 'Historical' },
  { id: 'settings', label: 'Settings' },
];

export default function App() {
  const { isConnected, alerts, stats, timeline } = useSocket();
  const [tab, setTab] = useState('live');

  return (
    <>
      <header className="navbar">
        <div className="navbar__brand">
          <span className="navbar__brand-icon" aria-hidden="true">
            🛡
          </span>
          <span className="navbar__brand-name">SENTINEL-AI</span>
        </div>
        <div className="navbar__actions">
          <div className="conn-status">
            <span
              className={`conn-status__dot${
                isConnected ? ' conn-status__dot--connected' : ''
              }`}
            />
            <span>{isConnected ? 'Live' : 'Disconnected'}</span>
          </div>
        </div>
      </header>

      <main className="dashboard">
        <nav className="tabs" aria-label="Dashboard sections">
          {TABS.map((item) => (
            <button
              key={item.id}
              type="button"
              className={`tabs__btn${tab === item.id ? ' tabs__btn--active' : ''}`}
              aria-current={tab === item.id ? 'page' : undefined}
              onClick={() => setTab(item.id)}
            >
              {item.label}
            </button>
          ))}
        </nav>

        {tab === 'live' && (
          <>
            <StatsCards stats={stats} alerts={alerts} />
            <div className="charts-row">
              <AlertTimeline timeline={timeline} />
              <AttackDistribution alerts={alerts} stats={stats} />
            </div>
            <div className="charts-row charts-row--single">
              <ConfidenceHistogram alerts={alerts} />
            </div>
            <AlertTable alerts={alerts} />
          </>
        )}

        {tab === 'historical' && (
          <section className="panel">
            <h2 className="panel__title">Historical analysis</h2>
            <p>
              Time-range queries arrive in Phase 6, once classified flows are
              stored in the time-series database.
            </p>
          </section>
        )}

        {tab === 'settings' && (
          <section className="panel">
            <h2 className="panel__title">Settings</h2>
            <p>
              The auto-block switch arrives in Phase 6. The inference service
              does not read the backend mitigation flag yet, so this screen
              does not change firewall behavior.
            </p>
            <p className="mt-8">
              Server flag:{' '}
              <span className="text-mono">
                {stats == null
                  ? 'unknown'
                  : stats.mitigation_enabled
                  ? 'enabled'
                  : 'disabled'}
              </span>
            </p>
          </section>
        )}
      </main>
    </>
  );
}