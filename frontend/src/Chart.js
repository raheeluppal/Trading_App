import React, { useEffect, useState } from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from "chart.js";
import { getChartData } from "./api";
import "./Chart.css";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

function Chart({ ticker }) {
  const [chartData, setChartData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const data = await getChartData(ticker);
        setChartData(data);
        setLoading(false);
      } catch (error) {
        console.error("Error fetching chart data:", error);
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 5000);

    return () => clearInterval(interval);
  }, [ticker]);

  if (loading) return <div className="chart-loading">Loading chart...</div>;
  if (!chartData || !chartData.data) return <div className="chart-loading">No data</div>;

  const data_points = chartData.data.slice(-50); // Last 50 candles

  const labels = data_points.map((d, i) => i);

  // Price Chart with Bollinger Bands
  const priceData = {
    labels,
    datasets: [
      {
        label: "Close Price",
        data: data_points.map((d) => d.close),
        borderColor: "#667eea",
        backgroundColor: "rgba(102, 126, 234, 0.05)",
        borderWidth: 2,
        fill: false,
        tension: 0.4,
      },
      {
        label: "Upper Band",
        data: data_points.map((d) => d.bb_upper),
        borderColor: "rgba(244, 67, 54, 0.5)",
        borderWidth: 1,
        borderDash: [5, 5],
        fill: false,
        tension: 0.4,
      },
      {
        label: "Lower Band",
        data: data_points.map((d) => d.bb_lower),
        borderColor: "rgba(76, 175, 80, 0.5)",
        borderWidth: 1,
        borderDash: [5, 5],
        fill: "-1",
        backgroundColor: "rgba(102, 126, 234, 0.05)",
        tension: 0.4,
      },
    ],
  };

  // RSI Chart
  const rsiData = {
    labels,
    datasets: [
      {
        label: "RSI (14)",
        data: data_points.map((d) => d.rsi),
        borderColor: "#ff9800",
        backgroundColor: "rgba(255, 152, 0, 0.1)",
        borderWidth: 2,
        fill: true,
        tension: 0.4,
      },
      {
        label: "Overbought (70)",
        data: Array(data_points.length).fill(70),
        borderColor: "rgba(244, 67, 54, 0.3)",
        borderWidth: 1,
        borderDash: [5, 5],
        fill: false,
        pointRadius: 0,
      },
      {
        label: "Oversold (30)",
        data: Array(data_points.length).fill(30),
        borderColor: "rgba(76, 175, 80, 0.3)",
        borderWidth: 1,
        borderDash: [5, 5],
        fill: false,
        pointRadius: 0,
      },
    ],
  };

  // MACD Chart
  const macdData = {
    labels,
    datasets: [
      {
        label: "MACD",
        data: data_points.map((d) => d.macd),
        borderColor: "#667eea",
        borderWidth: 2,
        fill: false,
        tension: 0.4,
      },
      {
        label: "Signal",
        data: data_points.map((d) => d.macd_signal),
        borderColor: "#ff9800",
        borderWidth: 2,
        fill: false,
        tension: 0.4,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: true,
    interaction: {
      mode: "index",
      intersect: false,
    },
    plugins: {
      legend: {
        position: "top",
        labels: {
          font: { size: 12 },
          padding: 15,
        },
      },
      title: {
        display: true,
        font: { size: 14, weight: "bold" },
        padding: 20,
      },
    },
    scales: {
      y: {
        grid: {
          color: "rgba(0, 0, 0, 0.05)",
        },
      },
      x: {
        grid: {
          display: false,
        },
      },
    },
  };

  return (
    <div className="chart-container">
      <div className="chart-wrapper">
        <Line
          data={priceData}
          options={{
            ...chartOptions,
            plugins: {
              ...chartOptions.plugins,
              title: { ...chartOptions.plugins.title, text: "Price & Bollinger Bands" },
            },
          }}
        />
      </div>

      <div className="chart-row">
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
      </div>
    </div>
  );
}

export default Chart;
