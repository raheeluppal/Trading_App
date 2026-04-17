import React, { useState } from "react";
import { placeOrder } from "./api";
import { TRADE_TICKERS } from "./tickers";
import "./QuickTradeBar.css";

const TICKERS = TRADE_TICKERS;

function QuickTradeBar() {
  const [ticker, setTicker] = useState(TICKERS[0]);
  const [qty, setQty] = useState(1);
  const [side, setSide] = useState("BUY");
  const [submitting, setSubmitting] = useState(false);
  const [message, setMessage] = useState(null);

  const submitOrder = async () => {
    const parsedQty = Number(qty);
    if (!Number.isInteger(parsedQty) || parsedQty < 1 || parsedQty > 100) {
      setMessage({ type: "error", text: "Quantity must be 1-100 shares." });
      return;
    }

    setSubmitting(true);
    const result = await placeOrder(ticker, parsedQty, side);
    setSubmitting(false);

    if (result.success) {
      setMessage({
        type: "success",
        text: `${side} placed: ${parsedQty} ${ticker}`,
      });
      window.dispatchEvent(new CustomEvent("trade:executed"));
    } else {
      setMessage({
        type: "error",
        text: result.error || "Failed to place order.",
      });
    }

    setTimeout(() => setMessage(null), 3000);
  };

  return (
    <div className="quick-trade-bar">
      <div className="quick-trade-controls">
        <select value={ticker} onChange={(e) => setTicker(e.target.value)}>
          {TICKERS.map((symbol) => (
            <option key={symbol} value={symbol}>
              {symbol}
            </option>
          ))}
        </select>

        <input
          type="number"
          min="1"
          max="100"
          value={qty}
          onChange={(e) => setQty(e.target.value)}
        />

        <button
          className={`quick-trade-side ${side === "BUY" ? "active-buy" : ""}`}
          onClick={() => setSide("BUY")}
          type="button"
        >
          BUY
        </button>
        <button
          className={`quick-trade-side ${side === "SELL" ? "active-sell" : ""}`}
          onClick={() => setSide("SELL")}
          type="button"
        >
          SELL
        </button>

        <button
          className={`quick-trade-submit ${side === "BUY" ? "buy" : "sell"}`}
          onClick={submitOrder}
          disabled={submitting}
          type="button"
        >
          {submitting ? "Submitting..." : `Quick ${side}`}
        </button>
      </div>

      {message && (
        <div className={`quick-trade-message ${message.type}`}>{message.text}</div>
      )}
    </div>
  );
}

export default QuickTradeBar;
