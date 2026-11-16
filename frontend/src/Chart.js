import React, { useEffect, useMemo, useState } from "react";
import { Chart as ReactChart } from "react-chartjs-2";
import {
  Chart as ChartJS, CategoryScale, LinearScale, LineController, LineElement,
  PointElement, BarController, BarElement, Tooltip, Legend,
} from "chart.js";
import { CandlestickController, CandlestickElement } from "chartjs-chart-financial";
import { addMarkedZone, deleteMarkedZone, getChartData, getMarkedZones } from "./api";
import { TRADE_TICKERS } from "./tickers";
import "./Chart.css";

ChartJS.register(CategoryScale, LinearScale, LineController, LineElement, PointElement,
  BarController, BarElement, CandlestickController, CandlestickElement, Tooltip, Legend);

const kinds = ["support", "resistance", "demand", "supply", "trendline_support",
  "trendline_resistance", "ath_retest", "daily_gap"];

function Chart({ ticker, onTickerChange }) {
  const [interval, setInterval] = useState("1m");
  const [candles, setCandles] = useState([]);
  const [zones, setZones] = useState([]);
  const [zoneForm, setZoneForm] = useState({ kind: "support", low: "", high: "" });
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    const refresh = async () => {
      const [chart, marked] = await Promise.all([
        getChartData(ticker, interval, 240), getMarkedZones(ticker),
      ]);
      if (active) {
        setCandles(chart?.data || []);
        setZones(marked?.zones || []);
      }
    };
    refresh();
    const timer = setInterval(refresh, 15000);
    return () => { active = false; clearInterval(timer); };
  }, [ticker, interval]);

  const chartData = useMemo(() => {
    const labels = candles.map((bar) => new Date(bar.timestamp).toLocaleTimeString("en-US", {
      timeZone: "America/New_York", hour: "2-digit", minute: "2-digit",
    }));
    const line = (label, key, color) => ({ type: "line", label, yAxisID: "price",
      data: candles.map((bar, i) => ({ x: i, y: bar[key] })), borderColor: color, pointRadius: 0, borderWidth: 1.5, spanGaps: true });
    const datasets = [
      { type: "candlestick", label: `${ticker} stock`, yAxisID: "price",
        data: candles.map((bar, i) => ({ x: i, o: bar.open, h: bar.high, l: bar.low, c: bar.close })) },
      line("EMA 9", "ema9", "#eab308"), line("EMA 21", "ema21", "#a78bfa"),
      line("VWAP", "vwap", "#38bdf8"),
      { type: "bar", label: "Volume", yAxisID: "volume", data: candles.map((bar, i) => ({ x: i, y: bar.volume })),
        backgroundColor: "#64748b88" },
    ];
    zones.forEach((zone) => {
      const color = ["demand", "support", "trendline_support", "ath_retest"].includes(zone.kind)
        ? "#22c55e" : "#ef4444";
      datasets.push({ type: "line", label: `${zone.kind} #${zone.id} low`, yAxisID: "price",
        data: candles.map((_, i) => ({ x: i, y: zone.low })), borderColor: color, borderDash: [5, 3],
        borderWidth: 1, pointRadius: 0 });
      if (zone.high !== zone.low) datasets.push({ type: "line", label: `${zone.kind} #${zone.id} high`,
        yAxisID: "price", data: candles.map((_, i) => ({ x: i, y: zone.high })), borderColor: color,
        borderDash: [5, 3], borderWidth: 1, pointRadius: 0 });
    });
    return { labels, datasets };
  }, [candles, zones, ticker]);

  const saveZone = async (event) => {
    event.preventDefault();
    setError("");
    const low = Number(zoneForm.low);
    const high = Number(zoneForm.high || zoneForm.low);
    if (!Number.isFinite(low) || !Number.isFinite(high) || low <= 0 || high < low) {
      setError("Enter positive price bounds, with high at or above low."); return;
    }
    try {
      await addMarkedZone(ticker, { kind: zoneForm.kind, low, high });
      const marked = await getMarkedZones(ticker);
      setZones(marked.zones || []);
      setZoneForm({ kind: zoneForm.kind, low: "", high: "" });
    } catch (cause) { setError(cause.response?.data?.detail || cause.message); }
  };

  const removeZone = async (id) => {
    try {
      await deleteMarkedZone(ticker, id);
      setZones((current) => current.filter((zone) => zone.id !== id));
    } catch (cause) { setError(cause.response?.data?.detail || cause.message); }
  };

  return <div className="chart-container">
    <div className="chart-header"><h3>Playbook chart</h3>
      <select value={ticker} onChange={(event) => onTickerChange(event.target.value)}>
        {TRADE_TICKERS.map((symbol) => <option key={symbol} value={symbol}>{symbol}</option>)}
      </select>
      <select value={interval} onChange={(event) => setInterval(event.target.value)}>
        {["1m", "5m", "1h", "1d"].map((value) => <option key={value} value={value}>{value}</option>)}
      </select>
    </div>
    <div style={{ height: 480 }}><ReactChart type="candlestick" data={chartData} options={{
      responsive: true, maintainAspectRatio: false, parsing: false,
      scales: { x: { type: "category", ticks: { maxTicksLimit: 12 } },
        price: { type: "linear", position: "right" },
        volume: { type: "linear", position: "left", display: false, max: Math.max(1, ...candles.map((bar) => bar.volume)) * 4 } },
      plugins: { legend: { display: true }, tooltip: { mode: "index", intersect: false } },
    }} /></div>
    <h4>Marked strategy zones</h4>
    <p>Drawn support, resistance, supply/demand, trendline and ATH areas must be entered here to affect signals. Price bounds are not inferred from the chart.</p>
    <form onSubmit={saveZone}>
      <select value={zoneForm.kind} onChange={(event) => setZoneForm({ ...zoneForm, kind: event.target.value })}>
        {kinds.map((kind) => <option key={kind} value={kind}>{kind.replaceAll("_", " ")}</option>)}
      </select>
      <input aria-label="Zone low price" type="number" step="0.01" min="0.01" placeholder="Low price" value={zoneForm.low} onChange={(event) => setZoneForm({ ...zoneForm, low: event.target.value })} />
      <input aria-label="Zone high price" type="number" step="0.01" min="0.01" placeholder="High price (optional)" value={zoneForm.high} onChange={(event) => setZoneForm({ ...zoneForm, high: event.target.value })} />
      <button type="submit">Add zone</button>
    </form>
    {error && <p role="alert">{error}</p>}
    <div>{zones.map((zone) => <div key={zone.id}>{zone.kind.replaceAll("_", " ")}: ${zone.low.toFixed(2)}–${zone.high.toFixed(2)} <button onClick={() => removeZone(zone.id)}>Remove</button></div>)}</div>
  </div>;
}

export default Chart;
