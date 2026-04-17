"""
Position Manager: Tracks open trades and exit signals.
- Entry: When model probability > 0.50 and generates BUY signal
- Exit: Based on profit target (+2%), stop loss (-1%), or time (1 hour)
"""

from datetime import datetime, timedelta
from typing import Dict, List

class Position:
    """Represents a single open trade with professional exit strategy."""
    
    def __init__(self, ticker, entry_price, entry_time, atr=None, qty=1, order_id=None):
        self.ticker = ticker
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.current_price = entry_price
        self.highest_price = entry_price  # Track peak for trailing stop
        self.status = "OPEN"  # OPEN, CLOSED_PROFIT, CLOSED_LOSS, CLOSED_TIME
        self.exit_reason = None
        self.exit_price = None
        self.exit_time = None
        self.atr = atr or (entry_price * 0.01)  # Default 1% ATR if not provided
        
        # Real order tracking
        self.qty = qty  # Number of shares
        self.order_id = order_id  # Entry buy order ID
        self.stop_order_id = None  # Stop loss order ID
        self.profit_order_ids = []  # Take profit order IDs for scale-outs
        
        # === PROFESSIONAL EXIT STRATEGY ===
        # 1. Initial stop: 1.5x ATR (volatility-adjusted)
        self.initial_stop_loss = entry_price - (self.atr * 1.5)
        self.current_stop_loss = self.initial_stop_loss
        
        # 2. Multiple profit targets (scale out strategy)
        self.targets = [
            {"level": 0.01, "percent": 0.25},   # +1% = exit 25% of position
            {"level": 0.02, "percent": 0.35},   # +2% = exit 35% of position
            {"level": 0.03, "percent": 0.40},   # +3% = exit remaining 40%
        ]
        self.target_status = {i: False for i in range(len(self.targets))}
        self.partial_exits = []
        
        # 3. Trailing stop (activate after +1%)
        self.trailing_stop_percent = 0.005  # 0.5% trail
        self.trailing_stop_active = False
        
        # 4. Risk/Reward management
        self.risk_amount = entry_price - self.initial_stop_loss
        self.target_reward = self.risk_amount * 2  # 1:2 risk/reward minimum
        
        # 5. Time-based exit (market close, 4 hours for intraday)
        self.max_hold_time = 14400  # 4 hours in seconds (one trading session)
        
        # 6. Breakeven stop (move stop to entry after +1.5%)
        self.breakeven_target = 0.015
        self.breakeven_activated = False
    
    def update_price(self, current_price, atr=None):
        """Update current price and check exit conditions."""
        self.current_price = current_price
        
        # Track highest price for trailing stop
        if current_price > self.highest_price:
            self.highest_price = current_price
        
        # Update ATR for dynamic stop adjustment
        if atr:
            self.atr = atr
    
    def get_pnl(self):
        """Get current profit/loss percentage."""
        if self.entry_price == 0:
            return 0
        return (self.current_price - self.entry_price) / self.entry_price
    
    def get_pnl_dollars(self):
        """Get profit/loss in dollars for full tracked position size."""
        return (self.current_price - self.entry_price) * self.qty
    
    def get_hold_time_seconds(self):
        """Get how long position has been held."""
        return (datetime.now() - self.entry_time).total_seconds()
    
    def get_hold_time_minutes(self):
        """Get hold time in minutes."""
        return self.get_hold_time_seconds() / 60
    
    def update_stops(self):
        """
        Update stops dynamically as price moves.
        Implements: trailing stop, breakeven, and profit-locking.
        """
        pnl = self.get_pnl()
        
        # 1. BREAKEVEN STOP: Move stop to entry at +1.5%
        if pnl >= self.breakeven_target and not self.breakeven_activated:
            self.current_stop_loss = self.entry_price
            self.breakeven_activated = True
        
        # 2. TRAILING STOP: Activate at +1%, trail by 0.5%
        if pnl >= 0.01:
            self.trailing_stop_active = True
            trailing_level = self.highest_price * (1 - self.trailing_stop_percent)
            if trailing_level > self.current_stop_loss:
                self.current_stop_loss = trailing_level
    
    def check_exit_signal(self, current_price, atr=None):
        """
        Check if any exit conditions are met.
        Returns: (should_exit: bool, exit_reason: str, exit_price: float, percent_exiting: float)
        """
        self.update_price(current_price, atr)
        self.update_stops()
        
        pnl = self.get_pnl()
        hold_time = self.get_hold_time_seconds()
        
        # 1. STOP LOSS: Initial or trailing stop hit
        if current_price <= self.current_stop_loss:
            return True, "STOP_LOSS", current_price, 1.0
        
        # 2. SCALE OUT: Hit profit targets progressively
        for i, target in enumerate(self.targets):
            if pnl >= target["level"] and not self.target_status[i]:
                self.target_status[i] = True
                self.partial_exits.append({
                    "price": current_price,
                    "percent": target["percent"],
                    "pnl": pnl,
                    "time": datetime.now()
                })
                return True, f"PROFIT_TARGET_{i+1}", current_price, target["percent"]
        
        # 3. TIME-BASED EXIT: Close at 4 hours (end of trading session)
        if hold_time >= self.max_hold_time:
            return True, "TIME_EXIT", current_price, 1.0
        
        return False, None, None, None
    
    def close_position(self, exit_reason, exit_price, percent_exiting=1.0, exit_time=None):
        """Close the position (or partially if scaling out)."""
        self.exit_reason = exit_reason
        self.exit_price = exit_price
        self.exit_time = exit_time or datetime.now()
        
        # Determine final status
        if percent_exiting < 1.0:
            self.status = "PARTIAL_CLOSE"
        elif exit_reason == "STOP_LOSS":
            self.status = "CLOSED_LOSS"
        elif "PROFIT_TARGET" in exit_reason:
            self.status = "CLOSED_PROFIT"
        else:
            self.status = "CLOSED_TIME"
    
    def to_dict(self):
        """Convert position to dictionary for API response."""
        pnl = self.get_pnl()
        hold_time_min = self.get_hold_time_minutes()
        
        # Calculate total size exited
        total_exited = sum(e["percent"] for e in self.partial_exits)
        remaining = 1.0 - total_exited
        
        # Calculate expected profit target (first uncompleted target)
        profit_target = next((t["level"] * 100 for i, t in enumerate(self.targets) if not self.target_status[i]), 3.0)
        
        return {
            "ticker": self.ticker,
            "status": self.status,
            "entry_price": round(self.entry_price, 2),
            "current_price": round(self.current_price, 2),
            "highest_price": round(self.highest_price, 2),
            "entry_time": self.entry_time.isoformat(),
            "hold_time_minutes": round(hold_time_min, 1),
            "pnl_percent": round(pnl * 100, 2),
            "pnl_dollars": round(self.get_pnl_dollars(), 2),
            
            # Risk Management
            "risk_dollars": round(self.risk_amount, 2),
            "initial_stop": round(self.initial_stop_loss, 2),
            "current_stop": round(self.current_stop_loss, 2),
            "atr": round(self.atr, 2),
            
            # Exit Levels (for dashboard compatibility)
            "profit_target": profit_target,
            "stop_loss": -1.5,  # Relative stop loss percentage
            
            # Stop Status
            "stop_distance_percent": round((self.current_price - self.current_stop_loss) / self.current_price * 100, 2),
            "breakeven_activated": self.breakeven_activated,
            "trailing_stop_active": self.trailing_stop_active,
            
            # Profit Targets
            "profit_targets": [
                {
                    "level_percent": t["level"] * 100,
                    "size_percent": t["percent"] * 100,
                    "hit": self.target_status[i],
                    "remaining_size": round(remaining * 100, 1)
                }
                for i, t in enumerate(self.targets)
            ],
            
            # Partial Exits
            "partial_exits": len(self.partial_exits),
            "total_exited_percent": round(total_exited * 100, 1),
            
            # Time Management
            "max_hold_minutes": round(self.max_hold_time / 60, 0),
            
            # Exit Info
            "exit_reason": self.exit_reason,
            "exit_price": round(self.exit_price, 2) if self.exit_price else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
        }


class PositionManager:
    """Manages all open and closed positions."""
    
    def __init__(self):
        self.open_positions: Dict[str, Position] = {}  # ticker -> Position
        self.closed_positions: List[Position] = []
    
    def open_position(self, ticker, entry_price, atr=None, qty=1, order_id=None):
        """Open a new position with volatility-adjusted stops."""
        position = Position(ticker, entry_price, datetime.now(), atr, qty, order_id)
        self.open_positions[ticker] = position
        return position
    
    def close_position(self, ticker, exit_reason, exit_price, percent_exiting=1.0):
        """Close an open position (or partially)."""
        if ticker in self.open_positions:
            position = self.open_positions[ticker]
            position.close_position(exit_reason, exit_price, percent_exiting)
            
            # Only remove from open if fully closed
            if percent_exiting >= 1.0:
                self.closed_positions.append(position)
                del self.open_positions[ticker]
            
            return position
        return None
    
    def update_prices(self, prices_dict, atr_dict=None):
        """
        Update prices for all open positions and check for exits.
        prices_dict: {"SPY": 450.25, "TSLA": 250.10, ...}
        atr_dict: {"SPY": 2.5, "TSLA": 3.2, ...}
        
        Returns: List of positions that should be closed
        """
        if atr_dict is None:
            atr_dict = {}
        
        positions_to_close = []
        
        for ticker, position in list(self.open_positions.items()):
            if ticker in prices_dict:
                atr = atr_dict.get(ticker)
                should_exit, exit_reason, exit_price, percent_exiting = position.check_exit_signal(prices_dict[ticker], atr)
                if should_exit:
                    positions_to_close.append({
                        "ticker": ticker,
                        "exit_reason": exit_reason,
                        "exit_price": exit_price,
                        "percent_exiting": percent_exiting,
                        "position": position
                    })
        
        return positions_to_close
    
    def get_open_positions(self):
        """Get all open positions as dictionaries."""
        return [pos.to_dict() for pos in self.open_positions.values()]
    
    def get_closed_positions(self):
        """Get all closed positions as dictionaries."""
        return [pos.to_dict() for pos in self.closed_positions]
    
    def get_statistics(self):
        """Get trading statistics."""
        if not self.closed_positions:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "profit_target_exits": 0,
                "stop_loss_exits": 0,
                "time_exits": 0,
                "total_pnl": 0,
                "win_rate": 0,
                "avg_hold_minutes": 0
            }
        
        winning = sum(1 for p in self.closed_positions if p.get_pnl() >= 0)
        losing = len(self.closed_positions) - winning
        profit_exits = sum(1 for p in self.closed_positions if p.exit_reason and "PROFIT_TARGET" in p.exit_reason)
        loss_exits = sum(1 for p in self.closed_positions if p.exit_reason == "STOP_LOSS")
        time_exits = sum(1 for p in self.closed_positions if p.exit_reason == "TIME_EXIT")
        total_pnl = sum(p.get_pnl_dollars() for p in self.closed_positions)
        avg_hold = sum(p.get_hold_time_minutes() for p in self.closed_positions) / len(self.closed_positions)
        
        return {
            "total_trades": len(self.closed_positions),
            "winning_trades": winning,
            "losing_trades": losing,
            "profit_target_exits": profit_exits,
            "stop_loss_exits": loss_exits,
            "time_exits": time_exits,
            "total_pnl": round(total_pnl, 2),
            "win_rate": round(winning / len(self.closed_positions) * 100, 1) if self.closed_positions else 0,
            "avg_hold_minutes": round(avg_hold, 1)
        }
