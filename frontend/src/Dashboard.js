import React, { useEffect, useState } from "react";
import { getSignals } from "./api";
import Chart from "./Chart";
import "./Dashboard.css";

function Dashboard() {
  const [signals, setSignals] = useState({});
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [selectedTicker, setSelectedTicker] = useState(null);

  useEffect(() => {
    // Fetch signals immediately on load
    const fetchSignals = async () => {
      const data = await getSignals();
      setSignals(data);
      setLastUpdate(new Date().toLocaleTimeString());
      setLoading(false);
    };

    fetchSignals();

    // Set up interval to fetch every 5 seconds
    const interval = setInterval(async () => {
      const data = await getSignals();
      setSignals(data);
      setLastUpdate(new Date().toLocaleTimeString());
    }, 5000);

    return () => clearInterval(interval);
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

      {loading ? (
        <div className="loading">Loading signals...</div>
      ) : Object.keys(signals).length === 0 ? (
        <div className="no-data">
          <p>No signals available. Make sure the backend is running.</p>
          <p>Run: <code>uvicorn main:app --reload</code></p>
        </div>
      ) : (
        <>
          {/* Signal Cards */}
          <div className="signals-grid">
            {Object.keys(signals).map((ticker) => (
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
                          width: `${(signals[ticker].probability || 0) * 100}%`,
                          backgroundColor: getProbabilityColor(
                            signals[ticker].probability || 0
                          ),
                        }}
                      ></div>
                    </div>
                    <p className="probability-value">
                      {(signals[ticker].probability || 0).toFixed(2)}
                    </p>
                  </div>

                  <div className="signal-section">
                    <label>Signal</label>
                    <div
                      className="signal-badge"
                      style={{
                        backgroundColor: getSignalColor(signals[ticker].signal),
                      }}
                    >
                      {signals[ticker].signal || "LOADING"}
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
              <Chart ticker={selectedTicker} />
            </div>
          )}
        </>
      )}

      <footer className="dashboard-footer">
        <p>🤖 AI Signal Engine | Updates every 5 seconds | XGBoost Model with Bollinger Bands</p>
      </footer>
    </div>
  );
}

export default Dashboard;
