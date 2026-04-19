import './App.css';
import { useState } from 'react';
import Dashboard from './Dashboard';
import Positions from './Positions';
import Orders from './Orders';
import Trading from './Trading';
import QuickTradeBar from './QuickTradeBar';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');

  return (
    <div className="App">
      <nav className="app-nav">
        <button
          className={`nav-tab ${activeTab === 'dashboard' ? 'active' : ''}`}
          onClick={() => setActiveTab('dashboard')}
        >
          Signals
        </button>
        <button
          className={`nav-tab ${activeTab === 'trading' ? 'active' : ''}`}
          onClick={() => setActiveTab('trading')}
        >
          Trade
        </button>
        <button
          className={`nav-tab ${activeTab === 'positions' ? 'active' : ''}`}
          onClick={() => setActiveTab('positions')}
        >
          Positions
        </button>
        <button
          className={`nav-tab ${activeTab === 'orders' ? 'active' : ''}`}
          onClick={() => setActiveTab('orders')}
        >
          Orders
        </button>
      </nav>
      <QuickTradeBar />

      {activeTab === 'dashboard' && <Dashboard />}
      {activeTab === 'trading' && <Trading />}
      {activeTab === 'positions' && <Positions />}
      {activeTab === 'orders' && <Orders />}
    </div>
  );
}

export default App;
