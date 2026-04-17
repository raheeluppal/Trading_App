# Trading Bot - Live Signal Engine

A complete trading bot system with live price data feed, ML-based signal engine, FastAPI backend, and React dashboard.

## System Overview

The system consists of 4 main components:

1. **Data Feed** - Alpaca streaming API for live stock prices
2. **Signal Engine** - XGBoost ML model that generates buy/sell signals every minute
3. **Backend API** - FastAPI server that coordinates data and signals
4. **Frontend Dashboard** - React app displaying live probabilities and trading signals

## Project Structure

```
trading-bot/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── data_feed.py         # Alpaca data integration
│   ├── features.py          # Technical indicator calculations
│   ├── model.py             # XGBoost model loader and predictor
│   └── trained_model.json   # Trained model (to be added)
├── frontend/
│   ├── src/
│   │   ├── App.js           # Main React app
│   │   ├── Dashboard.js     # Signal display component
│   │   └── api.js           # API communication helper
│   └── public/
│       └── index.html       # HTML entry point
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Backend Setup

### Prerequisites
- Python 3.8+
- pip

### Installation

1. Navigate to the project directory:
```bash
cd c:\Trading_app
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure Alpaca API keys:
   - Edit `backend/data_feed.py`
   - Replace `YOUR_KEY` and `YOUR_SECRET` with your Alpaca API credentials
   - Get keys from: https://app.alpaca.markets

5. Prepare trained model:
   - Place your trained XGBoost model as `backend/trained_model.json`
   - Or the code will use an untrained model for testing

### Running the Backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

### API Endpoints

- `GET /health` - Health check
- `GET /signals` - Get latest trading signals for all tickers

Response format:
```json
{
  "SPY": {
    "probability": 0.82,
    "signal": "BUY"
  },
  "TSLA": {
    "probability": 0.45,
    "signal": "WAIT"
  }
}
```

## Frontend Setup

### Prerequisites
- Node.js 14+ and npm

### Installation

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

### Running the Frontend

```bash
npm start
```

The app will open at `http://localhost:3000`

### Build for Production

```bash
npm run build
```

## Trading Signals

The system monitors 4 major stocks: **SPY, TSLA, AMZN, MSFT**

### Signal Generation
- Every 60 seconds, the system:
  1. Fetches 50 minutes of 1-minute candle data
  2. Calculates technical indicators (RSI, MACD, SMA)
  3. Feeds them into the XGBoost model
  4. Generates buy probability

### Signal Interpretation
- **BUY**: Probability > 75% - Strong buy signal
- **WAIT**: Probability ≤ 75% - Hold or sell

## Technical Indicators

The signal engine uses:
- **RSI (14)** - Relative Strength Index for momentum
- **MACD** - Moving Average Convergence Divergence for trend
- **SMA (20)** - Simple Moving Average for trend direction

## ML Model Training

The system expects a trained XGBoost binary classifier:
- **Input features**: RSI, MACD, MACD Signal, SMA 20
- **Output**: Probability of price increase (0-1)

### Model Retraining
Replace `trained_model.json` with your newly trained model and restart the backend.

## Development Workflow

1. **Collect historical data** using the Alpaca API
2. **Engineer features** using technical indicators
3. **Train XGBoost** model on labeled data
4. **Export model** as JSON
5. **Deploy** the backend and frontend
6. **Monitor** signals in the dashboard

## Environment Variables

For production deployment, set these environment variables:
- `ALPACA_API_KEY` - Alpaca API key
- `ALPACA_API_SECRET` - Alpaca API secret
- `API_BASE_URL` - Backend API URL (for frontend)

## Troubleshooting

### Backend Issues
- **Model not found**: Ensure `trained_model.json` exists in backend directory
- **Alpaca auth fails**: Check API credentials are correct
- **CORS errors**: Frontend should connect to `http://localhost:8000`

### Frontend Issues
- **Cannot connect to API**: Ensure backend is running on port 8000
- **Signals not updating**: Check browser console for API errors

## License

MIT License - feel free to use for educational purposes

## Next Steps

1. Get Alpaca API credentials
2. Prepare trained model or train one using historical data
3. Run backend: `uvicorn main:app --reload`
4. Run frontend: `npm start`
5. Monitor signals in the dashboard
