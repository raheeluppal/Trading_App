import React, { useState, useEffect } from "react";
import { getOrders } from "./api";
import "./Orders.css";

const Orders = () => {
  const [orders, setOrders] = useState([]);
  const [totalPending, setTotalPending] = useState(0);
  const [lastUpdate, setLastUpdate] = useState(new Date());
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchOrders = async () => {
      setIsLoading(true);
      const data = await getOrders();
      if (data && data.orders) {
        setOrders(data.orders);
        setTotalPending(data.total_pending || 0);
      }
      setLastUpdate(new Date());
      setIsLoading(false);
    };

    fetchOrders();
    const interval = setInterval(fetchOrders, 5000); // Poll every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const getOrderStatusColor = (status) => {
    if (status === "filled") return "filled";
    if (status === "partially_filled") return "partial";
    if (status === "pending_new" || status === "accepted") return "pending";
    if (status === "canceled" || status === "expired") return "canceled";
    return "pending";
  };

  const getOrderTypeColor = (orderType) => {
    if (orderType === "BUY") return "buy";
    if (orderType === "SELL") return "sell";
    if (orderType === "STOP") return "stop";
    return "neutral";
  };

  const groupedOrders = {
    buy: orders.filter((o) => o.type === "BUY"),
    sell: orders.filter((o) => o.type === "SELL"),
    stop: orders.filter((o) => o.type === "STOP"),
  };

  return (
    <div className="orders-container">
      <h2>📋 Pending Orders</h2>

      {/* Summary Stats */}
      <div className="orders-summary">
        <div className="summary-card">
          <div className="summary-label">Total Pending</div>
          <div className="summary-value">{totalPending}</div>
        </div>
        <div className="summary-card">
          <div className="summary-label">Buy Orders</div>
          <div className="summary-value" style={{ color: "#10b981" }}>
            {groupedOrders.buy.length}
          </div>
        </div>
        <div className="summary-card">
          <div className="summary-label">Sell Orders</div>
          <div className="summary-value" style={{ color: "#ef4444" }}>
            {groupedOrders.sell.length}
          </div>
        </div>
        <div className="summary-card">
          <div className="summary-label">Stop Orders</div>
          <div className="summary-value" style={{ color: "#f59e0b" }}>
            {groupedOrders.stop.length}
          </div>
        </div>
        <div className="summary-card">
          <div className="summary-label">Last Update</div>
          <div className="summary-time">
            {lastUpdate.toLocaleTimeString()}
          </div>
        </div>
      </div>

      {/* Orders Grid */}
      {isLoading ? (
        <div className="loading-state">Loading orders...</div>
      ) : totalPending === 0 ? (
        <div className="empty-state">
          <p>No pending orders</p>
          <p style={{ fontSize: "0.9em", color: "#9ca3af" }}>
            Orders will appear here when BUY signals are triggered
          </p>
        </div>
      ) : (
        <div className="orders-grid">
          {orders.map((order) => (
            <div
              key={order.id}
              className={`order-card order-${getOrderTypeColor(order.type)}`}
            >
              <div className="order-header">
                <div className="order-symbol">{order.symbol}</div>
                <div className={`order-type-badge ${getOrderTypeColor(order.type)}`}>
                  {order.type}
                </div>
                <div
                  className={`order-status-badge ${getOrderStatusColor(
                    order.status
                  )}`}
                >
                  {order.status === "pending_new"
                    ? "Pending"
                    : order.status === "partially_filled"
                    ? "Partial"
                    : order.status === "accepted"
                    ? "Accepted"
                    : order.status.charAt(0).toUpperCase() + order.status.slice(1)}
                </div>
              </div>

              <div className="order-details">
                <div className="detail-row">
                  <span className="label">Quantity:</span>
                  <span className="value">{order.qty} shares</span>
                </div>

                {order.type === "BUY" || order.type === "SELL" ? (
                  <div className="detail-row">
                    <span className="label">Market Order</span>
                    <span className="value">📊</span>
                  </div>
                ) : (
                  <>
                    {order.stop_price && (
                      <div className="detail-row">
                        <span className="label">Stop Price:</span>
                        <span className="value">${order.stop_price.toFixed(2)}</span>
                      </div>
                    )}
                  </>
                )}

                <div className="detail-row">
                  <span className="label">Filled:</span>
                  <span className="value">
                    {order.filled_qty} / {order.qty} (
                    {((order.filled_qty / order.qty) * 100).toFixed(0)}%)
                  </span>
                </div>

                {order.submitted_at && (
                  <div className="detail-row">
                    <span className="label">Submitted:</span>
                    <span className="value">
                      {new Date(order.submitted_at).toLocaleTimeString()}
                    </span>
                  </div>
                )}
              </div>

              {/* Progress Bar */}
              <div className="order-progress">
                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{
                      width: `${(order.filled_qty / order.qty) * 100}%`,
                    }}
                  ></div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default Orders;
