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

export const getChartData = async (ticker) => {
  try {
    const res = await axios.get(`${API_URL}/chart/${ticker}`);
    return res.data;
  } catch (error) {
    console.error("Error fetching chart data:", error);
    return null;
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
