import React, { useEffect, useState } from "react";
import { getHistory, getSignalUniverse } from "./api";
import Chart from "./Chart";
import "./Dashboard.css";

function Dashboard() {
  const [signals, setSignals] = useState({});
  const [allSignals, setAllSignals] = useState({});
  const [topVolumeTickers, setTopVolumeTickers] = useState([]);
  const [availableTickers, setAvailableTickers] = useState([]);
  const [viewMode, setViewMode] = useState("top10");
  const [tickerFilter, setTickerFilter] = useState("");
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [selectedTicker, setSelectedTicker] = useState(null);

  useEffect(() => {
    const fetchDashboardData = async () => {
      const [universeData, historyData] = await Promise.all([
        getSignalUniverse(tickerFilter),
        getHistory(),
      ]);
      const topSignals = {};
      (universeData.top_volume_tickers || []).forEach((ticker) => {
        if (universeData.signals?.[ticker]) {
          topSignals[ticker] = universeData.signals[ticker];
        }
      });
      setSignals(topSignals);
      setAllSignals(universeData.signals || {});
      setTopVolumeTickers(universeData.top_volume_tickers || []);
      setAvailableTickers(universeData.available_tickers || []);
      setHistory(historyData.history || []);
      setLastUpdate(new Date().toLocaleTimeString());
      setLoading(false);
    };

    fetchDashboardData();

    const interval = setInterval(async () => {
      const [universeData, historyData] = await Promise.all([
        getSignalUniverse(tickerFilter),
        getHistory(),
      ]);
      const topSignals = {};
      (universeData.top_volume_tickers || []).forEach((ticker) => {
        if (universeData.signals?.[ticker]) {
          topSignals[ticker] = universeData.signals[ticker];
        }
      });
      setSignals(topSignals);
      setAllSignals(universeData.signals || {});
      setTopVolumeTickers(universeData.top_volume_tickers || []);
      setAvailableTickers(universeData.available_tickers || []);
      setHistory(historyData.history || []);
      setLastUpdate(new Date().toLocaleTimeString());
    }, 5000);

    return () => clearInterval(interval);
  }, [tickerFilter]);

  useEffect(() => {
    const quickTickers = availableTickers.slice(0, 10);
    const handler = (event) => {
      const idx = Number(event.key) - 1;
      if (idx >= 0 && idx < quickTickers.length) {
        setSelectedTicker(quickTickers[idx]);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  const getSignalColor = (signal) => {
    switch (signal) {
      case "BUY":
        return "#4CAF50";
      case "WAIT":
        return "#FF9800";
      case "ERROR":
        return "#f44336";
      default:
        return "#9E9E9E";
    }
  };

  const getProbabilityColor = (prob) => {
    if (prob > 0.75) return "#4CAF50";
    if (prob > 0.5) return "#FFC107";
    return "#f44336";
  };

  const visibleHistory = selectedTicker
    ? history.filter((record) => record.ticker === selectedTicker)
    : history;

  const visibleSignals =
    viewMode === "top10"
      ? signals
      : Object.fromEntries(
          Object.entries(allSignals).filter(([ticker]) =>
            ticker.toUpperCase().includes(tickerFilter.trim().toUpperCase())
          )
        );

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <h1>📈 Live Trading Signals</h1>
        <div className="header-info">
          <span className="status-indicator">● Live</span>
          <span className="last-update">
            {lastUpdate ? `Last update: ${lastUpdate}` : "Loading..."}
          </span>
        </div>
      </header>

      <div className="signals-controls">
        <div className="signals-controls-left">
          <button
            type="button"
            className={viewMode === "top10" ? "active" : ""}
            onClick={() => setViewMode("top10")}
          >
            Top 10 Volume
          </button>
          <button
            type="button"
            className={viewMode === "all" ? "active" : ""}
            onClick={() => setViewMode("all")}
          >
            All Signals
          </button>
        </div>
        <input
          type="text"
          placeholder="Filter ticker (e.g., NVDA)"
          value={tickerFilter}
          onChange={(e) => setTickerFilter(e.target.value)}
        />
      </div>

      {loading ? (
        <div className="loading">Loading signals...</div>
      ) : Object.keys(visibleSignals).length === 0 ? (
        <div className="no-data">
          <p>No signals match this filter right now.</p>
          <p>Run: <code>uvicorn main:app --reload</code></p>
        </div>
      ) : (
        <>
          {/* Signal Cards */}
          <div className="signals-grid">
            {Object.keys(visibleSignals).map((ticker) => (
              <div
                key={ticker}
                className={`signal-card ${selectedTicker === ticker ? "active" : ""}`}
                onClick={() => setSelectedTicker(selectedTicker === ticker ? null : ticker)}
              >
                <div className="ticker-header">
                  <h2>{ticker}</h2>
                </div>

                <div className="signal-content">
                  <div className="probability-section">
                    <label>Probability</label>
                    <div className="probability-bar">
                      <div
                        className="probability-fill"
                        style={{
                          width: `${(visibleSignals[ticker].probability || 0) * 100}%`,
                          backgroundColor: getProbabilityColor(
                            visibleSignals[ticker].probability || 0
                          ),
                        }}
                      ></div>
                    </div>
                    <p className="probability-value">
                      {(visibleSignals[ticker].probability || 0).toFixed(2)}
                    </p>
                  </div>

                  <div className="signal-section">
                    <label>Signal</label>
                    <div
                      className="signal-badge"
                      style={{
                        backgroundColor: getSignalColor(visibleSignals[ticker].signal),
                      }}
                    >
                      {visibleSignals[ticker].signal || "LOADING"}
                    </div>
                  </div>

                  <div className="expand-hint">
                    {selectedTicker === ticker ? "Click to hide" : "Click for chart"}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Chart Section */}
          {selectedTicker && (
            <div className="chart-section">
              <h2>📊 {selectedTicker} Technical Analysis</h2>
              <Chart ticker={selectedTicker} onTickerChange={setSelectedTicker} />
            </div>
          )}

          <div className="history-section">
            <h2>
              🗂 Signal History {selectedTicker ? `- ${selectedTicker}` : "(All Tickers)"}
            </h2>
            {visibleHistory.length === 0 ? (
              <p className="history-empty">No historical records available yet.</p>
            ) : (
              <div className="history-table-wrapper">
                <table className="history-table">
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>Ticker</th>
                      <th>Signal</th>
                      <th>Probability</th>
                    </tr>
                  </thead>
                  <tbody>
                    {visibleHistory.map((record, idx) => (
                      <tr key={`${record.ticker}-${record.timestamp}-${idx}`}>
                        <td>{new Date(record.timestamp).toLocaleString()}</td>
                        <td>{record.ticker}</td>
                        <td>
                          <span
                            className="signal-badge history-signal"
                            style={{ backgroundColor: getSignalColor(record.signal) }}
                          >
                            {record.signal}
                          </span>
                        </td>
                        <td>{(record.probability || 0).toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </>
      )}

      <footer className="dashboard-footer">
        <p>🤖 AI Signal Engine | Updates every 5 seconds | XGBoost Model with Bollinger Bands</p>
      </footer>
    </div>
  );
}

export default Dashboard;
