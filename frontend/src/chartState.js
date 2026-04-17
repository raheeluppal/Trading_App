const STORAGE_KEY = "trading_chart_preferences_v1";
const LAYOUTS_KEY = "trading_chart_layouts_v1";

const DEFAULT_STATE = {
  interval: "1m",
  windowSize: "50",
  chartType: "candlestick",
  enabledIndicators: {
    bollinger: true,
    smaEma: true,
    vwap: true,
    rsi: true,
    macd: true,
    stoch: false,
    atr: false,
    volume: true,
  },
  indicatorPreset: "default",
};

export const loadChartState = () => {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (!saved) return DEFAULT_STATE;
    return { ...DEFAULT_STATE, ...JSON.parse(saved) };
  } catch (error) {
    return DEFAULT_STATE;
  }
};

export const saveChartState = (nextState) => {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(nextState));
  } catch (error) {
    // Ignore storage failures (private mode / quota)
  }
};

export const indicatorPresets = {
  default: { bollinger: true, smaEma: true, vwap: true, rsi: true, macd: true, stoch: false, atr: false, volume: true },
  momentum: { bollinger: false, smaEma: false, vwap: false, rsi: true, macd: true, stoch: true, atr: true, volume: true },
  trend: { bollinger: true, smaEma: true, vwap: true, rsi: false, macd: true, stoch: false, atr: false, volume: true },
  scalping: { bollinger: true, smaEma: true, vwap: true, rsi: true, macd: false, stoch: true, atr: true, volume: true },
};

export const intervalBarsMap = {
  "1m": 180,
  "5m": 220,
  "15m": 250,
  "1h": 280,
  "4h": 220,
  "1d": 300,
};

export const loadLayouts = () => {
  try {
    const raw = localStorage.getItem(LAYOUTS_KEY);
    if (!raw) return {};
    return JSON.parse(raw);
  } catch (error) {
    return {};
  }
};

export const saveLayouts = (layouts) => {
  try {
    localStorage.setItem(LAYOUTS_KEY, JSON.stringify(layouts));
  } catch (error) {
    // Ignore storage write failures.
  }
};
