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

export const getMarkedZones = async (ticker) =>
  (await axios.get(`${API_URL}/zones/${ticker}`)).data;

export const addMarkedZone = async (ticker, zone) =>
  (await axios.post(`${API_URL}/zones/${ticker}`, zone)).data;

export const deleteMarkedZone = async (ticker, id) =>
  (await axios.delete(`${API_URL}/zones/${ticker}/${id}`)).data;

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
