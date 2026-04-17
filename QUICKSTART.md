# Quick Start Guide - Trading Bot

## ⚡ 5-Minute Setup

### 1️⃣ Backend Setup
```bash
cd c:\Trading_app

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Edit backend/data_feed.py and add your Alpaca API keys
# Then run the backend
cd backend
uvicorn main:app --reload --port 8000
```

Backend will be ready at: `http://localhost:8000`

### 2️⃣ Frontend Setup (in a new terminal)
```bash
cd c:\Trading_app\frontend

# Install dependencies (first time only)
npm install

# Start development server
npm start
```

Frontend will open at: `http://localhost:3000`

## 📋 Checklist

- [ ] Python virtual environment created
- [ ] Python dependencies installed (`pip install -r requirements.txt`)
- [ ] Alpaca API keys added to `backend/data_feed.py`
- [ ] Backend running (`uvicorn main:app --reload`)
- [ ] Frontend npm dependencies installed (`npm install`)
- [ ] Frontend running (`npm start`)
- [ ] Dashboard visible at `http://localhost:3000`

## 🔑 Get Alpaca API Keys

1. Go to https://app.alpaca.markets
2. Sign up for a free paper trading account
3. Navigate to API Keys section
4. Copy API Key and Secret Key
5. Paste them in `backend/data_feed.py` (lines 7-8)

## 📊 What Should Happen

1. Backend starts fetching live market data every minute
2. XGBoost model generates buy/sell signals
3. Dashboard updates with live probabilities every 5 seconds
4. Signals show as:
   - 🟢 **BUY** (probability > 75%)
   - 🟠 **WAIT** (probability ≤ 75%)

## 🤖 Add Your Trained Model

Place your trained XGBoost model as:
```
backend/trained_model.json
```

The system will automatically load it on startup.

## 🔧 Troubleshooting

**Backend won't start:**
```bash
# Make sure port 8000 is free
netstat -ano | findstr :8000

# Kill process on port 8000 if needed
taskkill /PID <PID> /F
```

**Frontend can't connect to backend:**
- Check backend is running on `http://localhost:8000`
- Verify CORS is enabled (it is in main.py)

**No signals appear:**
- Check Alpaca API keys are correct
- Ensure market is open (trading hours)
- Check browser console for errors

## 📚 Project Files

```
Trading_app/
├── backend/
│   ├── main.py              ← FastAPI server
│   ├── data_feed.py         ← Alpaca integration
│   ├── features.py          ← Technical indicators
│   ├── model.py             ← XGBoost model
│   └── trained_model.json   ← Your trained model
├── frontend/
│   ├── src/
│   │   ├── App.js
│   │   ├── Dashboard.js     ← Main UI
│   │   └── api.js           ← API calls
│   └── public/
├── requirements.txt
├── README.md
└── QUICKSTART.md            ← This file
```

## 🚀 Next Steps

1. Set up the development environment ✅
2. Add real Alpaca credentials
3. Train your XGBoost model
4. Deploy to production (AWS, Heroku, etc.)

Happy trading! 📈
