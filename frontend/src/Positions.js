import React, { useEffect, useState } from "react";
import { getAccountSummary, getPositions, getTradeLog } from "./api";
import "./Positions.css";

function Positions() {
  const [positions, setPositions] = useState([]);
  const [summary, setSummary] = useState(null);
  const [trades, setTrades] = useState([]);

  useEffect(() => {
    let active = true;
    const refresh = async () => {
      try {
        const [open, account, log] = await Promise.all([
          getPositions(), getAccountSummary(), getTradeLog(300),
        ]);
        if (!active) return;
        setPositions(open.open_positions || []);
        setSummary(account);
        setTrades(log.trades || []);
      } catch (error) {
        console.error("Unable to load broker positions", error);
      }
    };
    refresh();
    const timer = setInterval(refresh, 5000);
    window.addEventListener("trade:executed", refresh);
    return () => {
      active = false;
      clearInterval(timer);
      window.removeEventListener("trade:executed", refresh);
    };
  }, []);

  return (
    <div className="positions-container">
      <div className="positions-header">
        <h2>Stock positions</h2>
        <div className="positions-summary">
          <div className="summary-stat"><span className="stat-label">Equity</span><span className="stat-value">${Number(summary?.equity || 0).toFixed(2)}</span></div>
          <div className="summary-stat"><span className="stat-label">Buying power</span><span className="stat-value">${Number(summary?.buying_power || 0).toFixed(2)}</span></div>
          <div className="summary-stat"><span className="stat-label">Open positions</span><span className="stat-value">{positions.length}</span></div>
          <div className="summary-stat"><span className="stat-label">Saved exit rows</span><span className="stat-value">{trades.length}</span></div>
        </div>
      </div>
      <p>Entry window: 9:30–11:00 a.m. ET. Existing positions may be reduced after 11:00. No automatic order or exit is placed by this app.</p>
      {positions.length ? (
        <div className="open-positions">
          <h3>Open broker positions</h3>
          <div className="positions-grid">
            {positions.map((pos) => (
              <div key={pos.ticker} className="position-card">
                <div className="position-header"><span className="ticker-badge">{pos.ticker}</span><span>{String(pos.side || "").toUpperCase()}</span></div>
                <div className="position-details">
                  <div className="detail-row"><span className="detail-label">Shares</span><span>{Math.abs(Number(pos.qty || 0))}</span></div>
                  <div className="detail-row"><span className="detail-label">Entry</span><span>${Number(pos.entry_price || 0).toFixed(2)}</span></div>
                  <div className="detail-row"><span className="detail-label">Current</span><span>${Number(pos.current_price || 0).toFixed(2)}</span></div>
                  <div className="detail-row"><span className="detail-label">Unrealized P&amp;L</span><span>${Number(pos.pnl_dollars || 0).toFixed(2)} ({Number(pos.pnl_percent || 0).toFixed(2)}%)</span></div>
                </div>
              </div>
            ))}
          </div>
          <p>Broker holdings do not carry a verified playbook stop or target here; manage those levels explicitly.</p>
        </div>
      ) : <p>No open positions in the six-symbol strategy universe.</p>}
      <div className="trade-history">
        <h3>Saved trade log</h3>
        <p>These are historical saved rows, not a replay of the current strategy.</p>
        {trades.length ? (
          <div className="history-table-wrapper"><table className="history-table"><thead><tr><th>Exit time</th><th>Ticker</th><th>Shares</th><th>Entry</th><th>Exit</th><th>P&amp;L</th><th>Reason</th></tr></thead><tbody>
            {trades.map((trade) => <tr key={trade.id}><td>{trade.exit_time || "—"}</td><td>{trade.ticker}</td><td>{trade.qty_shares}</td><td>${Number(trade.entry_price || 0).toFixed(2)}</td><td>${Number(trade.exit_price || 0).toFixed(2)}</td><td>${Number(trade.pnl_dollars || 0).toFixed(2)}</td><td>{trade.exit_reason_detail || "—"}</td></tr>)}
          </tbody></table></div>
        ) : <p>No saved exits.</p>}
      </div>
    </div>
  );
}

export default Positions;
