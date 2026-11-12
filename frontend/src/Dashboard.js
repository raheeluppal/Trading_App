import React, { useEffect, useState } from "react";
import { getSignalUniverse } from "./api";
import Chart from "./Chart";
import "./Dashboard.css";

function Dashboard() {
  const [signals, setSignals] = useState({});
  const [allSignals, setAllSignals] = useState({});
  const [availableTickers, setAvailableTickers] = useState([]);
  const [viewMode, setViewMode] = useState("top10");
  const [tickerFilter, setTickerFilter] = useState("");
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [selectedTicker, setSelectedTicker] = useState(null);

  useEffect(() => {
    const fetchDashboardData = async () => {
      const universeData = await getSignalUniverse(tickerFilter);
      const topSignals = {};
      (universeData.top_volume_tickers || []).forEach((ticker) => {
        if (universeData.signals?.[ticker]) {
          topSignals[ticker] = universeData.signals[ticker];
        }
      });
      setSignals(topSignals);
      setAllSignals(universeData.signals || {});
      setAvailableTickers(universeData.available_tickers || []);
      setLastUpdate(new Date().toLocaleTimeString());
      setLoading(false);
    };

    fetchDashboardData();

    const interval = setInterval(async () => {
      const universeData = await getSignalUniverse(tickerFilter);
      const topSignals = {};
      (universeData.top_volume_tickers || []).forEach((ticker) => {
        if (universeData.signals?.[ticker]) {
          topSignals[ticker] = universeData.signals[ticker];
        }
      });
      setSignals(topSignals);
      setAllSignals(universeData.signals || {});
      setAvailableTickers(universeData.available_tickers || []);
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
  }, [availableTickers]);

  const getSignalColor = (signal) => {
    switch (signal) {
      case "BUY":
        return "#4CAF50";
      case "PUT":
        return "#b71c1c";
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
        <h1>Live trading signals</h1>
        <div className="header-info">
          <span className="status-indicator">Live</span>
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
                    <label>Playbook checklist</label>
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
                      {visibleSignals[ticker].probability === 1 ? "Confirmed" : "Not confirmed"}
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
                      {visibleSignals[ticker].signal === "BUY" ? "CALL IDEA" : visibleSignals[ticker].signal === "PUT" ? "PUT IDEA" : visibleSignals[ticker].signal || "LOADING"}
                    </div>
                  </div>
                  <p>{visibleSignals[ticker].setup?.name || visibleSignals[ticker].setup?.reason}</p>
                  {visibleSignals[ticker].setup?.entry != null && (
                    <p>
                      Entry {visibleSignals[ticker].setup.entry.toFixed(2)} · Stop {visibleSignals[ticker].setup.stop.toFixed(2)} · Target {visibleSignals[ticker].setup.target.toFixed(2)}
                    </p>
                  )}

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
              <h2>{selectedTicker} — chart</h2>
              <Chart ticker={selectedTicker} onTickerChange={setSelectedTicker} />
            </div>
          )}

        </>
      )}

      <footer className="dashboard-footer">
        <p>
          Visual playbook signals · closed-candle checks · review before trading
        </p>
      </footer>
    </div>
  );
}

export default Dashboard;
