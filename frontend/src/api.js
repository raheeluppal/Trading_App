import axios from "axios";

const API_URL = "http://localhost:8000";

export const getSignals = async () => {
  try {
    const res = await axios.get(`${API_URL}/signals`);
    return res.data;
  } catch (error) {
    console.error("Error fetching signals:", error);
    return {};
  }
};

export const getSignalUniverse = async (query = "") => {
  try {
    const res = await axios.get(`${API_URL}/signals/universe`, {
      params: { query },
    });
    return res.data;
  } catch (error) {
    console.error("Error fetching signal universe:", error);
    return { top_volume_tickers: [], signals: {}, available_tickers: [] };
  }
};

export const getChartData = async (ticker, interval = "1m", bars = 120) => {
  try {
    const res = await axios.get(`${API_URL}/chart/${ticker}`, {
      params: { interval, bars },
    });
    return res.data;
  } catch (error) {
    console.error("Error fetching chart data:", error);
    return null;
  }
};

export const getHistory = async () => {
  try {
    const res = await axios.get(`${API_URL}/history`);
    return res.data;
  } catch (error) {
    console.error("Error fetching history:", error);
    return { history: [], total_records: 0 };
  }
};

export const getOrders = async () => {
  try {
    const res = await axios.get(`${API_URL}/orders`);
    return res.data;
  } catch (error) {
    console.error("Error fetching orders:", error);
    return { orders: [], total_pending: 0 };
  }
};

export const getPositions = async () => {
  try {
    const res = await axios.get(`${API_URL}/positions`);
    return res.data;
  } catch (error) {
    console.error("Error fetching positions:", error);
    return { open_positions: [], total_open: 0 };
  }
};

export const getPositionStats = async () => {
  try {
    const res = await axios.get(`${API_URL}/positions/stats`);
    return res.data;
  } catch (error) {
    console.error("Error fetching position stats:", error);
    return null;
  }
};

export const getClosedPositions = async () => {
  try {
    const res = await axios.get(`${API_URL}/positions/closed`);
    return res.data;
  } catch (error) {
    console.error("Error fetching closed positions:", error);
    return { closed_positions: [], total_closed: 0 };
  }
};

export const getTradeLog = async (limit = 200) => {
  try {
    const res = await axios.get(`${API_URL}/trades/log`, {
      params: { limit },
    });
    return res.data;
  } catch (error) {
    console.error("Error fetching trade log:", error);
    return { trades: [], total_returned: 0 };
  }
};

export const getAccountSummary = async () => {
  try {
    const res = await axios.get(`${API_URL}/account/summary`);
    return res.data;
  } catch (error) {
    console.error("Error fetching account summary:", error);
    return {
      equity: 0,
      cash: 0,
      buying_power: 0,
      portfolio_value: 0,
    };
  }
};

export const placeOrder = async (ticker, qty, orderType = "BUY") => {
  try {
    const res = await axios.post(`${API_URL}/orders/place`, null, {
      params: {
        ticker,
        qty: parseInt(qty),
        order_type: orderType.toUpperCase()
      }
    });
    return res.data;
  } catch (error) {
    console.error("Error placing order:", error);
    return { success: false, error: error.message };
  }
};

export const healthCheck = async () => {
  try {
    const res = await axios.get(`${API_URL}/health`);
    return res.data;
  } catch (error) {
    console.error("Error checking health:", error);
    return null;
  }
};

export const getAlertRules = async () => {
  try {
    const res = await axios.get(`${API_URL}/alerts/rules`);
    return res.data;
  } catch (error) {
    console.error("Error fetching alert rules:", error);
    return { rules: [] };
  }
};

export const createAlertRule = async (payload) => {
  try {
    const res = await axios.post(`${API_URL}/alerts/rules`, null, { params: payload });
    return res.data;
  } catch (error) {
    console.error("Error creating alert rule:", error);
    return { success: false, error: error.message };
  }
};

export const deleteAlertRule = async (ruleId) => {
  try {
    const res = await axios.delete(`${API_URL}/alerts/rules/${ruleId}`);
    return res.data;
  } catch (error) {
    console.error("Error deleting alert rule:", error);
    return { success: false, error: error.message };
  }
};

export const getAlertEvents = async () => {
  try {
    const res = await axios.get(`${API_URL}/alerts/events`);
    return res.data;
  } catch (error) {
    console.error("Error fetching alert events:", error);
    return { events: [], total: 0 };
  }
};
