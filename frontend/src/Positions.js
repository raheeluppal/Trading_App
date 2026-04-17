import React, { useState, useEffect } from 'react';
import { getAccountSummary, getClosedPositions, getPositions, getPositionStats } from './api';
import './Positions.css';

function Positions() {
  const [positions, setPositions] = useState({
    open_positions: [],
    total_open: 0
  });
  const [stats, setStats] = useState(null);
  const [closedTrades, setClosedTrades] = useState([]);
  const [accountSummary, setAccountSummary] = useState({
    equity: 0,
    cash: 0,
    buying_power: 0,
    portfolio_value: 0,
    starting_equity: 100000,
    account_pnl: 0,
    daily_pnl: 0,
    daily_pnl_percent: 0,
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchPositions = async () => {
      try {
        setLoading(true);
        const [posRes, statsRes, closedRes, accountRes] = await Promise.all([
          getPositions(),
          getPositionStats(),
          getClosedPositions(),
          getAccountSummary(),
        ]);
        setPositions(posRes);
        setStats(statsRes);
        setClosedTrades(closedRes?.closed_positions || []);
        setAccountSummary(accountRes);
      } catch (error) {
        console.error('Error fetching positions:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchPositions();
    const interval = setInterval(fetchPositions, 5000); // Update every 5 seconds

    const handleTradeExecuted = () => {
      fetchPositions();
    };
    window.addEventListener("trade:executed", handleTradeExecuted);

    return () => {
      clearInterval(interval);
      window.removeEventListener("trade:executed", handleTradeExecuted);
    };
  }, []);

  const getExitColor = (pnl_percent, hold_minutes) => {
    // Green: approaching profit target
    if (pnl_percent >= 1.5) return '#4ade80';
    // Yellow: neutral
    if (pnl_percent >= -0.5) return '#facc15';
    // Red: approaching stop loss
    return '#ef4444';
  };

  const getExitStatus = (pos) => {
    const pnl = pos.pnl_percent;
    const hold = pos.hold_time_minutes;
    const maxHold = pos.max_hold_minutes;

    // Check which exit condition is closest
    const profitToTarget = pos.profit_target - pnl;
    const lossToStop = pnl - pos.stop_loss;
    const timeToExit = maxHold - hold;

    let status = [];
    if (profitToTarget <= 0.5) {
      status.push(`💰 Profit Target: ${profitToTarget.toFixed(2)}%`);
    }
    if (lossToStop <= 0.5) {
      status.push(`⛔ Stop Loss: ${lossToStop.toFixed(2)}%`);
    }
    if (timeToExit <= 10) {
      status.push(`⏰ Time Exit: ${timeToExit.toFixed(0)} min`);
    }

    return status;
  };

  if (loading && !positions.open_positions.length) {
    return <div className="positions-container">Loading positions...</div>;
  }

  const accountPnl = Number(accountSummary.account_pnl || 0);
  const dailyPnl = Number(accountSummary.daily_pnl || 0);
  const dailyPnlPercent = Number(accountSummary.daily_pnl_percent || 0);

  return (
    <div className="positions-container">
      <div className="positions-header">
        <h2>📊 Active Positions & Trading</h2>
        <div className="positions-summary">
          <div className="summary-stat">
            <span className="stat-label">Total Balance</span>
            <span className="stat-value">${Number(accountSummary.equity || 0).toFixed(2)}</span>
          </div>
          <div className="summary-stat">
            <span className="stat-label">Buying Power</span>
            <span className="stat-value">${Number(accountSummary.buying_power || 0).toFixed(2)}</span>
          </div>
          <div className="summary-stat">
            <span className="stat-label">Open Trades</span>
            <span className="stat-value">{positions.total_open}</span>
          </div>
          <div className="summary-stat">
            <span className="stat-label">Trade History</span>
            <span className="stat-value">{closedTrades.length}</span>
          </div>
          {stats && (
            <>
              <div className="summary-stat">
                <span className="stat-label">Win Rate</span>
                <span className="stat-value">{stats.statistics.win_rate}%</span>
              </div>
              <div className="summary-stat">
                <span className="stat-label">Total P&L</span>
                <span className={`stat-value ${accountPnl >= 0 ? 'positive' : 'negative'}`}>
                  ${accountPnl.toFixed(2)}
                </span>
              </div>
              <div className="summary-stat">
                <span className="stat-label">Daily P&L</span>
                <span className={`stat-value ${dailyPnl >= 0 ? 'positive' : 'negative'}`}>
                  ${dailyPnl.toFixed(2)} ({dailyPnlPercent >= 0 ? '+' : ''}{dailyPnlPercent.toFixed(2)}%)
                </span>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Open Positions */}
      {positions.open_positions.length > 0 ? (
        <div className="open-positions">
          <h3>🟢 Open Positions ({positions.total_open})</h3>
          <div className="positions-grid">
            {positions.open_positions.map((pos) => (
              <div 
                key={pos.ticker} 
                className="position-card"
                style={{ borderLeftColor: getExitColor(pos.pnl_percent, pos.hold_time_minutes) }}
              >
                <div className="position-header">
                  <span className="ticker-badge">{pos.ticker}</span>
                  <span className={`pnl-badge ${pos.pnl_percent >= 0 ? 'profit' : 'loss'}`}>
                    {pos.pnl_percent >= 0 ? '📈' : '📉'} {pos.pnl_percent.toFixed(2)}%
                  </span>
                </div>

                <div className="position-details">
                  <div className="detail-row">
                    <span className="detail-label">Entry</span>
                    <span className="detail-value">${pos.entry_price.toFixed(2)}</span>
                  </div>
                  <div className="detail-row">
                    <span className="detail-label">Current</span>
                    <span className="detail-value">${pos.current_price.toFixed(2)}</span>
                  </div>
                  <div className="detail-row">
                    <span className="detail-label">P&L</span>
                    <span className={`detail-value ${pos.pnl_dollars >= 0 ? 'profit' : 'loss'}`}>
                      ${pos.pnl_dollars.toFixed(2)}
                    </span>
                  </div>
                  <div className="detail-row">
                    <span className="detail-label">Hold Time</span>
                    <span className="detail-value">{pos.hold_time_minutes.toFixed(1)} min</span>
                  </div>
                </div>

                {/* Exit Targets */}
                <div className="exit-targets">
                  <div className="target-item">
                    <span className="target-label">💰 Profit Target</span>
                    <div className="target-bar">
                      <div 
                        className="target-progress profit"
                        style={{ width: `${Math.min((pos.pnl_percent / pos.profit_target) * 100, 100)}%` }}
                      />
                    </div>
                    <span className="target-value">{pos.pnl_percent.toFixed(2)}% / {pos.profit_target.toFixed(2)}%</span>
                  </div>

                  <div className="target-item">
                    <span className="target-label">⛔ Stop Loss</span>
                    <div className="target-bar">
                      <div 
                        className="target-progress loss"
                        style={{ width: `${Math.max((Math.abs(pos.pnl_percent) / Math.abs(pos.stop_loss)) * 100, 0)}%` }}
                      />
                    </div>
                    <span className="target-value">{pos.pnl_percent.toFixed(2)}% / {pos.stop_loss.toFixed(2)}%</span>
                  </div>

                  <div className="target-item">
                    <span className="target-label">⏰ Time Exit</span>
                    <div className="target-bar">
                      <div 
                        className="target-progress time"
                        style={{ width: `${(pos.hold_time_minutes / pos.max_hold_minutes) * 100}%` }}
                      />
                    </div>
                    <span className="target-value">{pos.hold_time_minutes.toFixed(1)} / {pos.max_hold_minutes.toFixed(0)} min</span>
                  </div>
                </div>

                {/* Exit Alerts */}
                <div className="exit-alerts">
                  {getExitStatus(pos).map((status, idx) => (
                    <div key={idx} className="alert-item">⚠️ {status}</div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="no-positions">
          <p>📭 No open positions. Waiting for BUY signals...</p>
        </div>
      )}

      {/* Trading Statistics */}
      {stats && (
        <div className="trading-stats">
          <h3>📈 Trading Statistics</h3>
          <div className="stats-grid">
            <div className="stat-item">
              <span className="stat-name">Total Trades</span>
              <span className="stat-number">{stats.statistics.total_trades}</span>
            </div>
            <div className="stat-item">
              <span className="stat-name">Winning Trades</span>
              <span className="stat-number profit">{stats.statistics.winning_trades}</span>
            </div>
            <div className="stat-item">
              <span className="stat-name">Losing Trades</span>
              <span className="stat-number loss">{stats.statistics.losing_trades}</span>
            </div>
            <div className="stat-item">
              <span className="stat-name">Win Rate</span>
              <span className="stat-number">{stats.statistics.win_rate}%</span>
            </div>
            <div className="stat-item">
              <span className="stat-name">Avg Hold Time</span>
              <span className="stat-number">{stats.statistics.avg_hold_minutes} min</span>
            </div>
            <div className="stat-item">
              <span className="stat-name">Exit by Profit</span>
              <span className="stat-number profit">{stats.statistics.profit_target_exits}</span>
            </div>
            <div className="stat-item">
              <span className="stat-name">Exit by Stop Loss</span>
              <span className="stat-number loss">{stats.statistics.stop_loss_exits}</span>
            </div>
            <div className="stat-item">
              <span className="stat-name">Exit by Time</span>
              <span className="stat-number">{stats.statistics.time_exits}</span>
            </div>
            <div className="stat-item full-width">
              <span className="stat-name">Tracked Closed-Trade P&L</span>
              <span className={`stat-number ${stats.statistics.total_pnl >= 0 ? 'profit' : 'loss'}`}>
                ${stats.statistics.total_pnl}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Trade History */}
      <div className="trade-history">
        <h3>🧾 Trade History</h3>
        {closedTrades.length === 0 ? (
          <div className="no-positions">No closed trades yet.</div>
        ) : (
          <div className="history-table-wrapper">
            <table className="history-table">
              <thead>
                <tr>
                  <th>Ticker</th>
                  <th>Entry</th>
                  <th>Exit</th>
                  <th>P&L %</th>
                  <th>P&L $</th>
                  <th>Exit Reason</th>
                  <th>Exit Time</th>
                </tr>
              </thead>
              <tbody>
                {closedTrades.map((trade, idx) => (
                  <tr key={`${trade.ticker}-${trade.exit_time || idx}-${idx}`}>
                    <td>{trade.ticker}</td>
                    <td>${Number(trade.entry_price || 0).toFixed(2)}</td>
                    <td>${Number(trade.exit_price || 0).toFixed(2)}</td>
                    <td className={Number(trade.pnl_percent) >= 0 ? "profit-text" : "loss-text"}>
                      {Number(trade.pnl_percent || 0).toFixed(2)}%
                    </td>
                    <td className={Number(trade.pnl_dollars) >= 0 ? "profit-text" : "loss-text"}>
                      ${Number(trade.pnl_dollars || 0).toFixed(2)}
                    </td>
                    <td>{trade.exit_reason || "N/A"}</td>
                    <td>{trade.exit_time ? new Date(trade.exit_time).toLocaleString() : "N/A"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

export default Positions;
