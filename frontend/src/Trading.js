import React, { useState } from "react";
import { placeOrder } from "./api";
import "./Trading.css";

function Trading() {
  const tickers = ["SPY", "TSLA", "AMZN", "MSFT"];
  const [selectedTicker, setSelectedTicker] = useState("SPY");
  const [quantity, setQuantity] = useState(1);
  const [orderType, setOrderType] = useState("BUY");
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState(null);

  const handlePlaceOrder = async () => {
    if (quantity < 1 || quantity > 100) {
      setMessage({ type: "error", text: "Quantity must be between 1 and 100" });
      return;
    }

    setIsLoading(true);
    const result = await placeOrder(selectedTicker, quantity, orderType);
    setIsLoading(false);

    if (result.success) {
      setMessage({
        type: "success",
        text: `✅ ${orderType} order placed: ${quantity} shares of ${selectedTicker}`,
      });
      setQuantity(1);
      setTimeout(() => setMessage(null), 5000);
    } else {
      setMessage({
        type: "error",
        text: `❌ Error: ${result.error || "Failed to place order"}`,
      });
    }
  };

  return (
    <div className="trading-container">
      <h2>📊 Manual Trade Execution</h2>
      <p className="subtitle">Place custom BUY/SELL orders manually</p>

      <div className="trading-panel">
        {/* Ticker Selection */}
        <div className="trading-section">
          <label>Select Stock</label>
          <div className="ticker-buttons">
            {tickers.map((ticker) => (
              <button
                key={ticker}
                className={`ticker-btn ${selectedTicker === ticker ? "active" : ""}`}
                onClick={() => setSelectedTicker(ticker)}
              >
                {ticker}
              </button>
            ))}
          </div>
        </div>

        {/* Order Type Selection */}
        <div className="trading-section">
          <label>Order Type</label>
          <div className="order-type-buttons">
            <button
              className={`order-btn buy ${orderType === "BUY" ? "active" : ""}`}
              onClick={() => setOrderType("BUY")}
            >
              🟢 BUY
            </button>
            <button
              className={`order-btn sell ${orderType === "SELL" ? "active" : ""}`}
              onClick={() => setOrderType("SELL")}
            >
              🔴 SELL
            </button>
          </div>
        </div>

        {/* Quantity Input */}
        <div className="trading-section">
          <label>Quantity (shares)</label>
          <div className="quantity-input">
            <button
              className="qty-btn"
              onClick={() => setQuantity(Math.max(1, quantity - 1))}
            >
              −
            </button>
            <input
              type="number"
              min="1"
              max="100"
              value={quantity}
              onChange={(e) =>
                setQuantity(Math.max(1, Math.min(100, parseInt(e.target.value) || 1)))
              }
            />
            <button
              className="qty-btn"
              onClick={() => setQuantity(Math.min(100, quantity + 1))}
            >
              +
            </button>
          </div>
          <small>Max 100 shares per order</small>
        </div>

        {/* Action Buttons */}
        <div className="action-buttons">
          <button
            className={`place-order-btn ${orderType.toLowerCase()} ${isLoading ? "loading" : ""}`}
            onClick={handlePlaceOrder}
            disabled={isLoading}
          >
            {isLoading ? "Placing Order..." : `Place ${orderType} Order`}
          </button>
        </div>

        {/* Message Display */}
        {message && (
          <div className={`message ${message.type}`}>
            {message.text}
          </div>
        )}

        {/* Order Summary */}
        <div className="order-summary">
          <div className="summary-row">
            <span>Stock:</span>
            <strong>{selectedTicker}</strong>
          </div>
          <div className="summary-row">
            <span>Type:</span>
            <strong className={orderType.toLowerCase()}>{orderType}</strong>
          </div>
          <div className="summary-row">
            <span>Quantity:</span>
            <strong>{quantity} shares</strong>
          </div>
        </div>
      </div>

      {/* Info Box */}
      <div className="info-box">
        <h4>ℹ️ How to Trade</h4>
        <ul>
          <li>Select a stock (SPY, TSLA, AMZN, MSFT)</li>
          <li>Choose BUY or SELL</li>
          <li>Enter quantity (1-100 shares)</li>
          <li>Click "Place Order" to execute</li>
          <li>Orders appear in the 📋 Orders tab</li>
          <li>Positions appear in the 💼 Positions tab</li>
        </ul>
      </div>
    </div>
  );
}

export default Trading;
