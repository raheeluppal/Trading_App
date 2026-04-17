import React, { useEffect, useMemo, useState } from "react";
import { Bar, Chart as ReactChart, Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from "chart.js";
import {
  CandlestickController,
  CandlestickElement,
} from "chartjs-chart-financial";
import zoomPlugin from "chartjs-plugin-zoom";
import {
  createAlertRule,
  deleteAlertRule,
  getAlertEvents,
  getAlertRules,
  getChartData,
  getClosedPositions,
  getOrders,
  getPositions,
} from "./api";
import {
  intervalBarsMap,
  indicatorPresets,
  loadChartState,
  loadLayouts,
  saveChartState,
  saveLayouts,
} from "./chartState";
import "./Chart.css";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  CandlestickController,
  CandlestickElement,
  Title,
  Tooltip,
  Legend,
  Filler
  ,
  zoomPlugin
);

function Chart({ ticker, onTickerChange }) {
  const savedState = loadChartState();
  const [chartData, setChartData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [windowSize, setWindowSize] = useState(savedState.windowSize || "50");
  const [interval, setIntervalValue] = useState(savedState.interval || "1m");
  const [chartType, setChartType] = useState(savedState.chartType || "candlestick");
  const [enabledIndicators, setEnabledIndicators] = useState(
    savedState.enabledIndicators || indicatorPresets.default
  );
  const [activePreset, setActivePreset] = useState(savedState.indicatorPreset || "default");
  const [hoverSnapshot, setHoverSnapshot] = useState(null);
  const [tradeOverlay, setTradeOverlay] = useState({
    openPositions: [],
    closedPositions: [],
    openOrders: [],
  });
  const [drawings, setDrawings] = useState({ horizontal: [], trend: [], zones: [], fib: [] });
  const [alertRules, setAlertRules] = useState([]);
  const [alertEvents, setAlertEvents] = useState([]);
  const [alertForm, setAlertForm] = useState({
    metric: "price",
    condition: "above",
    threshold: "",
    cooldown_seconds: 300,
  });
  const [compareTicker, setCompareTicker] = useState("");
  const [compareData, setCompareData] = useState(null);
  const [layouts, setLayouts] = useState(loadLayouts());
  const [showOverlayDetails, setShowOverlayDetails] = useState(false);
  const [chartRef, setChartRef] = useState(null);
  const marketTimeFormatter = useMemo(
    () =>
      new Intl.DateTimeFormat("en-US", {
        hour: "2-digit",
        minute: "2-digit",
        hour12: true,
      }),
    []
  );
  const [windowOffset, setWindowOffset] = useState(0);

  const getPollingIntervalMs = () => (document.hidden ? 15000 : 5000);

  useEffect(() => {
    saveChartState({
      interval,
      windowSize,
      chartType,
      enabledIndicators,
      indicatorPreset: activePreset,
    });
  }, [interval, windowSize, chartType, enabledIndicators, activePreset]);

  useEffect(() => {
    const drawingKey = `chart_drawings_${ticker}_${interval}`;
    try {
      const raw = localStorage.getItem(drawingKey);
      if (raw) {
        setDrawings(JSON.parse(raw));
      } else {
        setDrawings({ horizontal: [], trend: [], zones: [], fib: [] });
      }
    } catch (error) {
      setDrawings({ horizontal: [], trend: [], zones: [], fib: [] });
    }
  }, [ticker, interval]);

  useEffect(() => {
    const fetchCompare = async () => {
      if (!compareTicker || compareTicker === ticker) {
        setCompareData(null);
        return;
      }
      const data = await getChartData(compareTicker, interval, intervalBarsMap[interval]);
      setCompareData(data?.data || null);
    };
    fetchCompare();
  }, [compareTicker, interval, ticker]);

  useEffect(() => {
    const drawingKey = `chart_drawings_${ticker}_${interval}`;
    try {
      localStorage.setItem(drawingKey, JSON.stringify(drawings));
    } catch (error) {
      // Ignore local storage write failures.
    }
  }, [drawings, ticker, interval]);

  useEffect(() => {
    const fetchTradeOverlay = async () => {
      const [positionsRes, closedRes, ordersRes] = await Promise.all([
        getPositions(),
        getClosedPositions(),
        getOrders(),
      ]);

      const openPositions = (positionsRes?.open_positions || []).filter(
        (pos) => pos.ticker === ticker
      );
      const closedPositions = (closedRes?.closed_positions || []).filter(
        (trade) => trade.ticker === ticker
      );
      const openOrders = (ordersRes?.orders || []).filter(
        (order) => order.symbol === ticker
      );

      setTradeOverlay({
        openPositions,
        closedPositions,
        openOrders,
      });
    };

    fetchTradeOverlay();
    let overlayInterval = setInterval(fetchTradeOverlay, getPollingIntervalMs());
    const handleTradeExecuted = () => fetchTradeOverlay();
    const handleVisibilityChange = () => {
      clearInterval(overlayInterval);
      overlayInterval = setInterval(fetchTradeOverlay, getPollingIntervalMs());
    };
    window.addEventListener("trade:executed", handleTradeExecuted);
    document.addEventListener("visibilitychange", handleVisibilityChange);

    return () => {
      clearInterval(overlayInterval);
      window.removeEventListener("trade:executed", handleTradeExecuted);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [ticker]);

  useEffect(() => {
    const fetchAlerts = async () => {
      const [rulesRes, eventsRes] = await Promise.all([getAlertRules(), getAlertEvents()]);
      setAlertRules((rulesRes?.rules || []).filter((rule) => rule.ticker === ticker));
      setAlertEvents((eventsRes?.events || []).filter((event) => event.ticker === ticker).slice(0, 8));
    };

    fetchAlerts();
    let alertInterval = setInterval(fetchAlerts, getPollingIntervalMs());
    const handleVisibilityChange = () => {
      clearInterval(alertInterval);
      alertInterval = setInterval(fetchAlerts, getPollingIntervalMs());
    };
    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => {
      clearInterval(alertInterval);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [ticker]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const data = await getChartData(ticker, interval, intervalBarsMap[interval]);
        setChartData(data);
        setLoading(false);
      } catch (error) {
        console.error("Error fetching chart data:", error);
        setLoading(false);
      }
    };

    fetchData();
    let dataInterval = setInterval(fetchData, getPollingIntervalMs());
    const handleVisibilityChange = () => {
      clearInterval(dataInterval);
      dataInterval = setInterval(fetchData, getPollingIntervalMs());
    };
    document.addEventListener("visibilitychange", handleVisibilityChange);

    return () => {
      clearInterval(dataInterval);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [ticker, interval]);

  const data_points = useMemo(() => {
    const points = chartData?.data || [];
    if (windowSize === "all") return points;
    const size = Number(windowSize);
    const end = Math.max(size, points.length - windowOffset);
    const start = Math.max(0, end - size);
    return points.slice(start, end);
  }, [chartData, windowSize, windowOffset]);

  const maxWindowOffset = useMemo(() => {
    if (windowSize === "all") return 0;
    const points = chartData?.data || [];
    return Math.max(0, points.length - Number(windowSize));
  }, [chartData, windowSize]);

  useEffect(() => {
    setWindowOffset(0);
  }, [ticker, interval, windowSize]);

  const labels = useMemo(
    () =>
      data_points.map((d, idx) => {
        // Feed timestamps are effectively ET wall-clock values for this UI.
        // Remove timezone suffix to avoid browser UTC conversion shifts.
        const normalizedTimestamp = String(d.timestamp || "")
          .replace("Z", "")
          .replace(/([+-]\d{2}:\d{2})$/, "");
        const parsedDate = new Date(normalizedTimestamp);
        if (!Number.isNaN(parsedDate.getTime())) {
          return marketTimeFormatter.format(parsedDate);
        }
        return `${idx + 1}`;
      }),
    [data_points, marketTimeFormatter]
  );
  const latest = data_points[data_points.length - 1];
  const previous = data_points[Math.max(data_points.length - 2, 0)];
  const priceChangePct =
    previous && previous.close
      ? ((latest.close - previous.close) / previous.close) * 100
      : 0;

  // Candlestick Chart with Bollinger Bands
  const priceData = useMemo(() => ({
    labels,
    datasets: [
      {
        label: chartType === "candlestick" ? "Candles" : "Price",
        type: chartType === "candlestick" ? "candlestick" : "line",
        data: data_points.map((d, i) => ({
          x: i,
          o: d.open,
          h: d.high,
          l: d.low,
          c: d.close,
          y: d.close,
        })),
        borderColor: "rgba(30, 41, 59, 0.8)",
        backgroundColor:
          chartType === "area" ? "rgba(37,99,235,0.15)" : "rgba(37,99,235,0.05)",
        pointRadius: 0,
        tension: chartType === "line" || chartType === "area" ? 0.2 : 0,
        fill: chartType === "area",
        color: {
          up: "#16a34a",
          down: "#dc2626",
          unchanged: "#64748b",
        },
      },
      {
        label: "BB Upper",
        type: "line",
        data: data_points.map((d, i) => ({ x: i, y: d.bb_upper })),
        borderColor: "rgba(239, 68, 68, 0.7)",
        borderWidth: 1,
        borderDash: [5, 5],
        fill: false,
        pointRadius: 0,
        tension: 0.4,
        hidden: !enabledIndicators.bollinger,
      },
      {
        label: "BB Middle",
        type: "line",
        data: data_points.map((d, i) => ({ x: i, y: d.bb_middle })),
        borderColor: "rgba(14, 165, 233, 0.8)",
        borderWidth: 1,
        fill: false,
        pointRadius: 0,
        tension: 0.35,
        hidden: !enabledIndicators.bollinger,
      },
      {
        label: "BB Lower",
        type: "line",
        data: data_points.map((d, i) => ({ x: i, y: d.bb_lower })),
        borderColor: "rgba(34, 197, 94, 0.7)",
        borderWidth: 1,
        borderDash: [5, 5],
        fill: "-2",
        backgroundColor: "rgba(37, 99, 235, 0.08)",
        pointRadius: 0,
        tension: 0.4,
        hidden: !enabledIndicators.bollinger,
      },
      {
        label: "EMA 20",
        type: "line",
        data: data_points.map((d, i) => ({ x: i, y: d.ema_20 })),
        borderColor: "rgba(124, 58, 237, 0.9)",
        borderWidth: 1.4,
        fill: false,
        pointRadius: 0,
        tension: 0.25,
        hidden: !enabledIndicators.smaEma,
      },
      {
        label: "SMA 20",
        type: "line",
        data: data_points.map((d, i) => ({ x: i, y: d.sma_20 })),
        borderColor: "rgba(234, 88, 12, 0.9)",
        borderWidth: 1.4,
        fill: false,
        pointRadius: 0,
        tension: 0.25,
        hidden: !enabledIndicators.smaEma,
      },
      {
        label: "VWAP",
        type: "line",
        data: data_points.map((d, i) => ({ x: i, y: d.vwap })),
        borderColor: "rgba(8, 145, 178, 0.9)",
        borderWidth: 1.3,
        fill: false,
        pointRadius: 0,
        tension: 0.2,
        hidden: !enabledIndicators.vwap,
      },
      ...tradeOverlay.openPositions.flatMap((pos) => [
        {
          label: `Entry ${pos.ticker}`,
          type: "line",
          data: data_points.map((_, i) => ({ x: i, y: Number(pos.entry_price) })),
          borderColor: "rgba(16, 185, 129, 0.65)",
          borderWidth: 1.2,
          borderDash: [4, 4],
          pointRadius: 0,
          fill: false,
          hidden: !showOverlayDetails,
        },
        {
          label: `Stop ${pos.ticker}`,
          type: "line",
          data: data_points.map((_, i) => ({ x: i, y: Number(pos.current_stop || pos.initial_stop || 0) })),
          borderColor: "rgba(239, 68, 68, 0.7)",
          borderWidth: 1.2,
          borderDash: [3, 3],
          pointRadius: 0,
          fill: false,
          hidden: !showOverlayDetails,
        },
        {
          label: `Target ${pos.ticker}`,
          type: "line",
          data: data_points.map((_, i) => ({ x: i, y: Number(pos.profit_target || 0) })),
          borderColor: "rgba(59, 130, 246, 0.6)",
          borderWidth: 1.2,
          borderDash: [2, 5],
          pointRadius: 0,
          fill: false,
          hidden: !showOverlayDetails,
        },
      ]),
      ...tradeOverlay.closedPositions.map((trade) => ({
        label: `Exit ${trade.ticker}`,
        type: "line",
        data: data_points.map((_, i) => ({
          x: i,
          y: i === data_points.length - 1 ? Number(trade.exit_price || 0) : null,
        })),
        borderColor: "rgba(245, 158, 11, 0.9)",
        pointRadius: 5,
        pointHoverRadius: 6,
        showLine: false,
        hidden: !showOverlayDetails,
      })),
      ...tradeOverlay.openOrders
        .filter((order) => order.stop_price || order.limit_price)
        .map((order) => ({
          label: `${order.type} Order`,
          type: "line",
          data: data_points.map((_, i) => ({
            x: i,
            y: Number(order.stop_price || order.limit_price || 0),
          })),
          borderColor: "rgba(168, 85, 247, 0.6)",
          borderWidth: 1,
          borderDash: [2, 2],
          pointRadius: 0,
          fill: false,
          hidden: !showOverlayDetails,
        })),
      ...drawings.horizontal.map((line, idx) => ({
        label: `HLine ${idx + 1}`,
        type: "line",
        data: data_points.map((_, i) => ({ x: i, y: Number(line.price) })),
        borderColor: "rgba(30, 64, 175, 0.65)",
        borderWidth: 1,
        pointRadius: 0,
        fill: false,
        hidden: !showOverlayDetails,
      })),
      ...drawings.trend.map((line, idx) => ({
        label: `Trend ${idx + 1}`,
        type: "line",
        data: data_points.map((_, i) => {
          const slope = (Number(line.endPrice) - Number(line.startPrice)) / Math.max(data_points.length - 1, 1);
          return { x: i, y: Number(line.startPrice) + slope * i };
        }),
        borderColor: "rgba(2, 132, 199, 0.8)",
        borderWidth: 1.2,
        pointRadius: 0,
        fill: false,
        hidden: !showOverlayDetails,
      })),
      ...drawings.zones.flatMap((zone, idx) => [
        {
          label: `ZoneTop ${idx + 1}`,
          type: "line",
          data: data_points.map((_, i) => ({ x: i, y: Number(zone.high) })),
          borderColor: "rgba(217, 119, 6, 0.5)",
          borderWidth: 1,
          pointRadius: 0,
          fill: false,
          hidden: !showOverlayDetails,
        },
        {
          label: `ZoneBot ${idx + 1}`,
          type: "line",
          data: data_points.map((_, i) => ({ x: i, y: Number(zone.low) })),
          borderColor: "rgba(217, 119, 6, 0.5)",
          borderWidth: 1,
          pointRadius: 0,
          fill: false,
          hidden: !showOverlayDetails,
        },
      ]),
      ...drawings.fib.flatMap((fib, idx) => {
        const low = Number(fib.low);
        const high = Number(fib.high);
        const levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1];
        return levels.map((level) => ({
          label: `Fib ${idx + 1} ${level}`,
          type: "line",
          data: data_points.map((_, i) => ({ x: i, y: low + (high - low) * level })),
          borderColor: "rgba(124, 58, 237, 0.45)",
          borderWidth: 1,
          pointRadius: 0,
          fill: false,
          hidden: !showOverlayDetails,
        }));
      }),
      ...(compareData && compareData.length > 1
        ? [
            {
              label: `Compare ${compareTicker}`,
              type: "line",
              data: (() => {
                const normalized = compareData.slice(-data_points.length);
                const base = Number(normalized[0]?.close || 1);
                return normalized.map((d, i) => ({
                  x: i,
                  y: (Number(d.close) / base) * Number(data_points[0]?.close || 1),
                }));
              })(),
              borderColor: "rgba(20, 184, 166, 0.85)",
              borderWidth: 1.5,
              borderDash: [6, 4],
              pointRadius: 0,
              fill: false,
            },
          ]
        : []),
    ],
  }), [labels, chartType, enabledIndicators, tradeOverlay, drawings, compareData, compareTicker, data_points, showOverlayDetails]);

  const volumeData = useMemo(() => ({
    labels,
    datasets: [
      {
        label: "Volume",
        data: data_points.map((d, i) => ({ x: i, y: Number(d.volume || 0) })),
        backgroundColor: data_points.map((d) =>
          d.close >= d.open ? "rgba(34, 197, 94, 0.6)" : "rgba(239, 68, 68, 0.6)"
        ),
        borderWidth: 0,
        borderRadius: 2,
      },
    ],
  }), [labels, data_points]);

  // RSI Chart
  const rsiData = useMemo(() => ({
    labels,
    datasets: [
      {
        label: "RSI (14)",
        data: data_points.map((d, i) => ({ x: i, y: Number(d.rsi ?? NaN) })),
        borderColor: "#ff9800",
        backgroundColor: "rgba(249, 115, 22, 0.12)",
        borderWidth: 2,
        fill: true,
        pointRadius: 0,
        tension: 0.28,
        hidden: !enabledIndicators.rsi,
      },
      {
        label: "Overbought (70)",
        data: data_points.map((_, i) => ({ x: i, y: 70 })),
        borderColor: "rgba(244, 67, 54, 0.3)",
        borderWidth: 1,
        borderDash: [5, 5],
        fill: false,
        pointRadius: 0,
        hidden: !enabledIndicators.rsi,
      },
      {
        label: "Oversold (30)",
        data: data_points.map((_, i) => ({ x: i, y: 30 })),
        borderColor: "rgba(76, 175, 80, 0.3)",
        borderWidth: 1,
        borderDash: [5, 5],
        fill: false,
        pointRadius: 0,
        hidden: !enabledIndicators.rsi,
      },
    ],
  }), [labels, data_points, enabledIndicators]);

  // MACD Chart
  const macdData = useMemo(() => ({
    labels,
    datasets: [
      {
        label: "MACD",
        data: data_points.map((d, i) => ({ x: i, y: Number(d.macd ?? NaN) })),
        borderColor: "#667eea",
        borderWidth: 2,
        fill: false,
        pointRadius: 0,
        tension: 0.25,
        hidden: !enabledIndicators.macd,
      },
      {
        label: "Signal",
        data: data_points.map((d, i) => ({ x: i, y: Number(d.macd_signal ?? NaN) })),
        borderColor: "#f97316",
        borderWidth: 2,
        fill: false,
        pointRadius: 0,
        tension: 0.25,
        hidden: !enabledIndicators.macd,
      },
    ],
  }), [labels, data_points, enabledIndicators]);

  const stochData = useMemo(() => ({
    labels,
    datasets: [
      {
        label: "Stoch %K",
        data: data_points.map((d, i) => ({ x: i, y: Number(d.stoch_k ?? NaN) })),
        borderColor: "#8b5cf6",
        borderWidth: 2,
        fill: false,
        pointRadius: 0,
        tension: 0.25,
      },
      {
        label: "Stoch %D",
        data: data_points.map((d, i) => ({ x: i, y: Number(d.stoch_d ?? NaN) })),
        borderColor: "#f97316",
        borderWidth: 2,
        fill: false,
        pointRadius: 0,
        tension: 0.25,
      },
      {
        label: "80",
        data: data_points.map((_, i) => ({ x: i, y: 80 })),
        borderColor: "rgba(239, 68, 68, 0.35)",
        borderDash: [5, 5],
        pointRadius: 0,
      },
      {
        label: "20",
        data: data_points.map((_, i) => ({ x: i, y: 20 })),
        borderColor: "rgba(34, 197, 94, 0.35)",
        borderDash: [5, 5],
        pointRadius: 0,
      },
    ],
  }), [labels, data_points]);

  const atrData = useMemo(() => ({
    labels,
    datasets: [
      {
        label: "ATR (14)",
        data: data_points.map((d, i) => ({ x: i, y: Number(d.atr ?? NaN) })),
        borderColor: "#0ea5e9",
        backgroundColor: "rgba(14, 165, 233, 0.12)",
        borderWidth: 2,
        fill: true,
        pointRadius: 0,
        tension: 0.25,
      },
    ],
  }), [labels, data_points]);

  const chartOptions = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    normalized: true,
    interaction: {
      mode: "index",
      intersect: false,
    },
    plugins: {
      legend: {
        position: "top",
        labels: {
          color: "#334155",
          font: { size: 11, weight: "600" },
          padding: 8,
          usePointStyle: true,
          filter: (item) =>
            !item.text.startsWith("Zone") &&
            !item.text.startsWith("Fib") &&
            !item.text.startsWith("HLine") &&
            !item.text.startsWith("Trend"),
        },
      },
      title: {
        display: true,
        color: "#0f172a",
        font: { size: 13, weight: "bold" },
        padding: 12,
      },
      tooltip: {
        backgroundColor: "rgba(15,23,42,0.95)",
        titleColor: "#e2e8f0",
        bodyColor: "#cbd5e1",
        borderColor: "#334155",
        borderWidth: 1,
      },
      zoom: {
        pan: {
          enabled: true,
          mode: "x",
          threshold: 5,
        },
        zoom: {
          wheel: {
            enabled: true,
            modifierKey: "ctrl",
          },
          pinch: {
            enabled: true,
          },
          drag: {
            enabled: true,
            backgroundColor: "rgba(37, 99, 235, 0.12)",
            borderColor: "rgba(37, 99, 235, 0.5)",
            borderWidth: 1,
          },
          mode: "x",
        },
      },
    },
    scales: {
      y: {
        grid: {
          color: "rgba(148, 163, 184, 0.2)",
        },
        ticks: {
          color: "#475569",
        },
      },
      x: {
        type: "linear",
        grid: {
          display: false,
        },
        ticks: {
          color: "#64748b",
          maxTicksLimit: 8,
          callback: (value) => labels[Math.round(value)] || "",
        },
      },
    },
  }), [labels]);

  if (loading) return <div className="chart-loading">Loading chart...</div>;
  if (!chartData || !chartData.data) return <div className="chart-loading">No data</div>;
  if (data_points.length === 0) return <div className="chart-loading">No chart points</div>;

  const addHorizontal = () => {
    const price = Number(latest?.close || 0);
    setDrawings((prev) => ({ ...prev, horizontal: [...prev.horizontal, { price }] }));
  };

  const addTrend = () => {
    const startPrice = Number(previous?.close || latest?.close || 0);
    const endPrice = Number(latest?.close || 0);
    setDrawings((prev) => ({
      ...prev,
      trend: [...prev.trend, { startPrice, endPrice }],
    }));
  };

  const addZone = () => {
    const center = Number(latest?.close || 0);
    setDrawings((prev) => ({
      ...prev,
      zones: [...prev.zones, { low: center * 0.995, high: center * 1.005 }],
    }));
  };

  const addFib = () => {
    const low = Number(Math.min(...data_points.map((p) => p.low)));
    const high = Number(Math.max(...data_points.map((p) => p.high)));
    setDrawings((prev) => ({ ...prev, fib: [...prev.fib, { low, high }] }));
  };

  const clearDrawings = () => setDrawings({ horizontal: [], trend: [], zones: [], fib: [] });

  const submitAlert = async () => {
    const threshold = Number(alertForm.threshold);
    if (Number.isNaN(threshold)) return;
    await createAlertRule({
      ticker,
      metric: alertForm.metric,
      condition: alertForm.condition,
      threshold,
      cooldown_seconds: Number(alertForm.cooldown_seconds || 300),
    });
    const rulesRes = await getAlertRules();
    setAlertRules((rulesRes?.rules || []).filter((rule) => rule.ticker === ticker));
    setAlertForm((prev) => ({ ...prev, threshold: "" }));
  };

  const saveLayout = () => {
    const name = window.prompt("Layout name?");
    if (!name) return;
    const nextLayouts = {
      ...layouts,
      [name]: { interval, windowSize, chartType, enabledIndicators, activePreset, compareTicker },
    };
    setLayouts(nextLayouts);
    saveLayouts(nextLayouts);
  };

  const loadLayout = (name) => {
    const layout = layouts[name];
    if (!layout) return;
    setIntervalValue(layout.interval || "1m");
    setWindowSize(layout.windowSize || "50");
    setChartType(layout.chartType || "candlestick");
    setEnabledIndicators(layout.enabledIndicators || indicatorPresets.default);
    setActivePreset(layout.activePreset || "default");
    setCompareTicker(layout.compareTicker || "");
  };

  const zoomIn = () => chartRef?.zoom(1.2);
  const zoomOut = () => chartRef?.zoom(0.8);
  const resetZoom = () => chartRef?.resetZoom();

  return (
    <div className="chart-container">
      <div className="chart-toolbar">
        <div className="chart-toolbar-left">
          <span className="chart-ticker">{ticker}</span>
          <span className="chart-price">${latest.close.toFixed(2)}</span>
          <span className={`chart-change ${priceChangePct >= 0 ? "up" : "down"}`}>
            {priceChangePct >= 0 ? "▲" : "▼"} {Math.abs(priceChangePct).toFixed(2)}%
          </span>
        </div>
        <div className="chart-window-buttons">
          {["SPY", "TSLA", "AMZN", "MSFT"].map((symbol) => (
            <button
              key={symbol}
              type="button"
              className={ticker === symbol ? "active" : ""}
              onClick={() => onTickerChange && onTickerChange(symbol)}
            >
              {symbol}
            </button>
          ))}
        </div>
        <div className="chart-window-buttons">
          {["1m", "5m", "15m", "1h", "4h", "1d"].map((option) => (
            <button
              key={option}
              type="button"
              className={interval === option ? "active" : ""}
              onClick={() => setIntervalValue(option)}
            >
              {option}
            </button>
          ))}
          {["20", "50", "all"].map((option) => (
            <button
              key={option}
              type="button"
              className={windowSize === option ? "active" : ""}
              onClick={() => setWindowSize(option)}
            >
              {option === "all" ? "All" : `${option} bars`}
            </button>
          ))}
          {windowSize !== "all" && (
            <>
              <button
                type="button"
                disabled={windowOffset >= maxWindowOffset}
                onClick={() =>
                  setWindowOffset((prev) => Math.min(maxWindowOffset, prev + 10))
                }
              >
                Older ◀
              </button>
              <button
                type="button"
                disabled={windowOffset <= 0}
                onClick={() => setWindowOffset((prev) => Math.max(0, prev - 10))}
              >
                Newer ▶
              </button>
              <span style={{ fontSize: "0.72rem", color: "#475569", alignSelf: "center" }}>
                Offset: {windowOffset}/{maxWindowOffset}
              </span>
            </>
          )}
        </div>
        <div className="chart-window-buttons">
          {["candlestick", "line", "area"].map((option) => (
            <button
              key={option}
              type="button"
              className={chartType === option ? "active" : ""}
              onClick={() => setChartType(option)}
            >
              {option}
            </button>
          ))}
        </div>
        <div className="chart-window-buttons">
          <select value={compareTicker} onChange={(e) => setCompareTicker(e.target.value)}>
            <option value="">Compare</option>
            {["SPY", "TSLA", "AMZN", "MSFT"].filter((s) => s !== ticker).map((symbol) => (
              <option key={symbol} value={symbol}>{symbol}</option>
            ))}
          </select>
          <button type="button" onClick={saveLayout}>Save Layout</button>
          <select onChange={(e) => loadLayout(e.target.value)} defaultValue="">
            <option value="" disabled>Load Layout</option>
            {Object.keys(layouts).map((name) => (
              <option key={name} value={name}>{name}</option>
            ))}
          </select>
        </div>
        <div className="chart-window-buttons">
          {Object.keys(indicatorPresets).map((preset) => (
            <button
              key={preset}
              type="button"
              className={activePreset === preset ? "active" : ""}
              onClick={() => {
                setActivePreset(preset);
                setEnabledIndicators(indicatorPresets[preset]);
              }}
            >
              {preset}
            </button>
          ))}
        </div>
        <div className="chart-window-buttons">
          {[
            ["bollinger", "BB"],
            ["smaEma", "SMA/EMA"],
            ["vwap", "VWAP"],
            ["volume", "VOL"],
            ["rsi", "RSI"],
            ["macd", "MACD"],
            ["stoch", "STOCH"],
            ["atr", "ATR"],
          ].map(([key, label]) => (
            <button
              key={key}
              type="button"
              className={enabledIndicators[key] ? "active" : ""}
              onClick={() =>
                setEnabledIndicators((prev) => ({
                  ...prev,
                  [key]: !prev[key],
                }))
              }
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      <div className="chart-toolbar">
        <div className="chart-window-buttons">
          <button type="button" onClick={zoomIn}>Zoom +</button>
          <button type="button" onClick={zoomOut}>Zoom -</button>
          <button type="button" onClick={resetZoom}>Reset Zoom</button>
          <button
            type="button"
            className={showOverlayDetails ? "active" : ""}
            onClick={() => setShowOverlayDetails((prev) => !prev)}
          >
            Overlays
          </button>
          <button type="button" onClick={addHorizontal}>Add H-Line</button>
          <button type="button" onClick={addTrend}>Add Trend</button>
          <button type="button" onClick={addZone}>Add Zone</button>
          <button type="button" onClick={addFib}>Add Fib</button>
          <button type="button" onClick={clearDrawings}>Clear Drawings</button>
        </div>
        <div className="chart-window-buttons">
          <select
            value={alertForm.metric}
            onChange={(e) => setAlertForm((prev) => ({ ...prev, metric: e.target.value }))}
          >
            <option value="price">Price</option>
            <option value="rsi">RSI</option>
            <option value="probability">Probability %</option>
          </select>
          <select
            value={alertForm.condition}
            onChange={(e) => setAlertForm((prev) => ({ ...prev, condition: e.target.value }))}
          >
            <option value="above">Above</option>
            <option value="below">Below</option>
          </select>
          <input
            type="number"
            value={alertForm.threshold}
            placeholder="Threshold"
            onChange={(e) => setAlertForm((prev) => ({ ...prev, threshold: e.target.value }))}
          />
          <button type="button" onClick={submitAlert}>Add Alert</button>
        </div>
      </div>

      <div
        className="chart-wrapper"
        onWheelCapture={(e) => {
          if (e.ctrlKey) return; // let plugin zoom handle Ctrl+wheel
          if (windowSize === "all") return;
          e.preventDefault();
          const delta = e.deltaY > 0 ? 3 : -3;
          setWindowOffset((prev) =>
            Math.max(0, Math.min(maxWindowOffset, prev + delta))
          );
        }}
      >
        <ReactChart
          ref={(instance) => {
            if (instance) setChartRef(instance);
          }}
          type="candlestick"
          data={priceData}
          options={{
            ...chartOptions,
            onHover: (_, elements) => {
              if (!elements || elements.length === 0) return;
              const index = elements[0].index;
              const point = data_points[index];
              if (!point) return;
              setHoverSnapshot({
                time: labels[index],
                open: point.open,
                high: point.high,
                low: point.low,
                close: point.close,
                rsi: point.rsi,
                macd: point.macd,
              });
            },
            plugins: {
              ...chartOptions.plugins,
              title: { ...chartOptions.plugins.title, text: "Candlesticks & Bollinger Bands" },
            },
          }}
        />
      </div>

      {hoverSnapshot && (
        <div className="chart-hover-readout">
          <span>{hoverSnapshot.time}</span>
          <span>O: {hoverSnapshot.open.toFixed(2)}</span>
          <span>H: {hoverSnapshot.high.toFixed(2)}</span>
          <span>L: {hoverSnapshot.low.toFixed(2)}</span>
          <span>C: {hoverSnapshot.close.toFixed(2)}</span>
          <span>RSI: {(hoverSnapshot.rsi || 0).toFixed(2)}</span>
          <span>MACD: {(hoverSnapshot.macd || 0).toFixed(4)}</span>
        </div>
      )}

      {(alertRules.length > 0 || alertEvents.length > 0) && (
        <div className="chart-hover-readout">
          {alertRules.slice(0, 4).map((rule) => (
            <span key={rule.id}>
              Rule: {rule.metric} {rule.condition} {rule.threshold}
              <button
                type="button"
                className="mini-action"
                onClick={async () => {
                  await deleteAlertRule(rule.id);
                  const rulesRes = await getAlertRules();
                  setAlertRules((rulesRes?.rules || []).filter((r) => r.ticker === ticker));
                }}
              >
                x
              </button>
            </span>
          ))}
          {alertEvents.slice(0, 3).map((event) => (
            <span key={event.id}>Alert: {event.message}</span>
          ))}
        </div>
      )}

      <div className="chart-row">
        {enabledIndicators.volume && (
          <div className="chart-wrapper chart-half">
          <Bar
            data={volumeData}
            options={{
              ...chartOptions,
              plugins: {
                ...chartOptions.plugins,
                title: { ...chartOptions.plugins.title, text: "Volume" },
              },
              scales: {
                ...chartOptions.scales,
                y: {
                  ...chartOptions.scales.y,
                  ticks: {
                    color: "#475569",
                    callback: (value) => `${(value / 1000000).toFixed(1)}M`,
                  },
                },
              },
            }}
          />
          </div>
        )}

        {enabledIndicators.rsi && (
          <div className="chart-wrapper chart-half">
          <Line
            data={rsiData}
            options={{
              ...chartOptions,
              plugins: {
                ...chartOptions.plugins,
                title: { ...chartOptions.plugins.title, text: "RSI Indicator" },
              },
              scales: {
                ...chartOptions.scales,
                y: { ...chartOptions.scales.y, max: 100, min: 0 },
              },
            }}
          />
          </div>
        )}

        {enabledIndicators.macd && (
          <div className="chart-wrapper chart-half">
          <Line
            data={macdData}
            options={{
              ...chartOptions,
              plugins: {
                ...chartOptions.plugins,
                title: { ...chartOptions.plugins.title, text: "MACD Indicator" },
              },
            }}
          />
          </div>
        )}

        {enabledIndicators.stoch && (
          <div className="chart-wrapper chart-half">
            <Line
              data={stochData}
              options={{
                ...chartOptions,
                plugins: {
                  ...chartOptions.plugins,
                  title: { ...chartOptions.plugins.title, text: "Stochastic Oscillator" },
                },
                scales: {
                  ...chartOptions.scales,
                  y: { ...chartOptions.scales.y, min: 0, max: 100 },
                },
              }}
            />
          </div>
        )}

        {enabledIndicators.atr && (
          <div className="chart-wrapper chart-half">
            <Line
              data={atrData}
              options={{
                ...chartOptions,
                plugins: {
                  ...chartOptions.plugins,
                  title: { ...chartOptions.plugins.title, text: "Average True Range" },
                },
              }}
            />
          </div>
        )}
      </div>
    </div>
  );
}

export default Chart;
