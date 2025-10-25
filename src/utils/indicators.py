"""
indicators.py
Enhanced Technical Indicators Module for Algorithmic Trading Bot
- Production-ready with comprehensive error handling
- Verified against standard formulas (Investopedia, TA-Lib, TradingView)
- Optimized for performance with vectorized operations
- Supports all Step 3 tasks of the algorithmic trading bot project
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Union, Dict
from datetime import datetime
import warnings
import logging
from binance.client import Client  # Assumes python-binance from Step 2

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/trading_bot.log', mode='a')
    ]
)
logger = logging.getLogger('TechnicalIndicators')

class TechnicalIndicators:
    """
    Technical Indicators Calculator for Algorithmic Trading
    - Calculates a comprehensive set of indicators for trading strategies
    - Handles OHLCV data with robust validation
    - Supports Binance Testnet integration
    - Aligns with Step 3 of the algorithmic trading bot project

    Example:
        from binance.client import Client
        client = Client(api_key, api_secret, testnet=True)
        ti = TechnicalIndicators(client=client)
        df = ti.fetch_klines('BTCUSDT', Client.KLINE_INTERVAL_1DAY, '1 year ago UTC')
        df_with_indicators = ti.calculate_all_indicators()
    """
    
    def __init__(self, df: Optional[pd.DataFrame] = None, client: Optional[Client] = None, 
                 log_level: str = 'INFO'):
        """
        Initialize with a DataFrame or Binance client.

        Args:
            df (pd.DataFrame, optional): DataFrame with columns: timestamp, open, high, low, close, volume
            client (Client, optional): Binance API client for fetching K-line data
            log_level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')

        Raises:
            ValueError: If required columns are missing, data is invalid, or timestamps are not datetime
        """
        self.logger = logging.getLogger('TechnicalIndicators')
        level = getattr(logging, log_level.upper(), logging.INFO)
        self.logger.setLevel(level)
        
        self.client = client
        if df is not None:
            self._validate_dataframe(df)
            self.df = df.copy()
            self.df.reset_index(drop=True, inplace=True)
            self.logger.info(f"✅ Initialized with {len(df)} data points")
            self.logger.info(f"📊 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        else:
            self.df = None
            self.logger.info("✅ Initialized without DataFrame; use fetch_klines to load data")

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate OHLCV DataFrame integrity."""
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if len(df) == 0:
            raise ValueError("DataFrame cannot be empty")
        
        # Validate timestamp
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='raise')
        except Exception as e:
            raise ValueError(f"Invalid timestamp format: {e}")
        
        # Check data length
        if len(df) < 200:
            self.logger.warning(f"⚠️ DataFrame has only {len(df)} rows; some indicators (e.g., SMA_200) require more data")
        
        # Validate OHLC data
        if (df[['open', 'high', 'low', 'close', 'volume']] < 0).any().any():
            self.logger.warning("⚠️ Found negative values in OHLCV data")
        
        invalid_hl = df['high'] < df['low']
        if invalid_hl.any():
            self.logger.warning(f"⚠️ Found {invalid_hl.sum()} rows where high < low")
        
        invalid_hc = df['high'] < df['close']
        invalid_lc = df['low'] > df['close']
        if invalid_hc.any() or invalid_lc.any():
            self.logger.warning(f"⚠️ Found OHLC inconsistencies in data")

    def _safe_divide(self, numerator: pd.Series, denominator: pd.Series, 
                     fill_value: float = np.nan) -> pd.Series:
        """Safely divide two series, handling division by zero."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = numerator / denominator.replace(0, np.nan)
        return result.fillna(fill_value) if not np.isnan(fill_value) else result

    def fetch_klines(self, symbol: str, interval: str, start_str: str) -> pd.DataFrame:
        """
        Fetch historical K-line data from Binance and store in DataFrame.
        Completes Task 3.2.

        Args:
            symbol (str): Trading pair (e.g., 'BTCUSDT')
            interval (str): K-line interval (e.g., Client.KLINE_INTERVAL_1DAY)
            start_str (str): Start time (e.g., '1 year ago UTC')

        Returns:
            pd.DataFrame: DataFrame with OHLCV data
        """
        if self.client is None:
            raise ValueError("Binance client not provided")
        
        self.logger.info(f"📡 Fetching K-line data for {symbol} (interval: {interval})")
        try:
            klines = self.client.get_historical_klines(symbol, interval, start_str)
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                'taker_buy_quote_asset_volume', 'ignore'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]  # Task 3.3: Select essential columns
            self._validate_dataframe(df)
            self.df = df
            self.logger.info(f"✅ Fetched {len(df)} K-lines for {symbol}")
            return df
        except Exception as e:
            self.logger.error(f"❌ Failed to fetch K-lines: {e}", exc_info=True)
            raise

    def calculate_sma(self, period: int = 20, column: str = 'close') -> pd.Series:
        """Calculate Simple Moving Average (SMA). Task 3.5."""
        if period < 1:
            raise ValueError("Period must be >= 1")
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found")
        
        self.logger.info(f"📈 Calculating SMA with period={period} on column={column}")
        sma = self.df[column].rolling(window=period, min_periods=period).mean()
        self.df[f'SMA_{period}'] = sma
        self.logger.debug(f"✅ SMA_{period} calculated. Valid values: {sma.notna().sum()}/{len(sma)}, Latest: {sma.iloc[-1]:.2f}")
        return sma

    def calculate_ema(self, period: int = 20, column: str = 'close') -> pd.Series:
        """Calculate Exponential Moving Average (EMA). Task 3.6."""
        if period < 1:
            raise ValueError("Period must be >= 1")
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found")
        
        self.logger.info(f"📈 Calculating EMA with period={period} on column={column}")
        ema = self.df[column].ewm(span=period, adjust=False, min_periods=period).mean()
        self.df[f'EMA_{period}'] = ema
        self.logger.debug(f"✅ EMA_{period} calculated. Valid values: {ema.notna().sum()}/{len(ema)}, Latest: {ema.iloc[-1]:.2f}")
        return ema

    def calculate_wma(self, period: int = 20, column: str = 'close') -> pd.Series:
        """Calculate Weighted Moving Average (WMA)."""
        if period < 1:
            raise ValueError("Period must be >= 1")
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found")
        
        self.logger.info(f"📈 Calculating WMA with period={period} on column={column}")
        weights = np.arange(1, period + 1)
        weights_sum = weights.sum()
        
        def weighted_mean(x):
            if len(x) < period:
                return np.nan
            return np.dot(x, weights) / weights_sum
        
        wma = self.df[column].rolling(window=period, min_periods=period).apply(weighted_mean, raw=True)
        self.df[f'WMA_{period}'] = wma
        self.logger.debug(f"✅ WMA_{period} calculated. Valid values: {wma.notna().sum()}/{len(wma)}, Latest: {wma.iloc[-1]:.2f}")
        return wma

    def calculate_vwap(self, reset_period: str = 'daily') -> pd.Series:
        """Calculate Volume Weighted Average Price (VWAP)."""
        if reset_period not in ['daily', 'cumulative']:
            raise ValueError("reset_period must be 'daily' or 'cumulative'")
        
        self.logger.info(f"📊 Calculating VWAP (reset_period={reset_period})")
        typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        
        if reset_period == 'daily':
            self.df['_date'] = self.df['timestamp'].dt.date
            self.df['_tp_volume'] = typical_price * self.df['volume']
            vwap = self.df.groupby('_date', group_keys=False).apply(
                lambda x: self._safe_divide(x['_tp_volume'].cumsum(), x['volume'].cumsum())
            )
            self.df.drop(['_date', '_tp_volume'], axis=1, inplace=True)
        else:
            tp_volume = typical_price * self.df['volume']
            vwap = self._safe_divide(tp_volume.cumsum(), self.df['volume'].cumsum())
        
        self.df['VWAP'] = vwap
        self.logger.debug(f"✅ VWAP calculated. Latest: {vwap.iloc[-1]:.4f}")
        return vwap

    def calculate_rsi(self, period: int = 14, column: str = 'close') -> pd.Series:
        """Calculate Relative Strength Index (RSI). Task 3.7."""
        if period < 1:
            raise ValueError("Period must be >= 1")
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found")
        
        self.logger.info(f"📊 Calculating RSI with period={period} on column={column}")
        delta = self.df[column].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        rs = self._safe_divide(avg_gain, avg_loss, fill_value=0)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.clip(0, 100)
        
        self.df[f'RSI_{period}'] = rsi
        self.logger.debug(f"✅ RSI_{period} calculated. Valid values: {rsi.notna().sum()}/{len(rsi)}, Latest: {rsi.iloc[-1]:.2f}")
        return rsi

    def calculate_macd(self, fast: int = 12, slow: int = 26, signal: int = 9, 
                      column: str = 'close') -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        if fast >= slow:
            raise ValueError("Fast period must be less than slow period")
        if signal < 1:
            raise ValueError("Signal period must be >= 1")
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found")
        
        self.logger.info(f"📊 Calculating MACD (fast={fast}, slow={slow}, signal={signal})")
        if len(self.df) < slow:
            self.logger.warning(f"⚠️ DataFrame length ({len(self.df)}) < slow period ({slow})")
        
        ema_fast = self.df[column].ewm(span=fast, adjust=False, min_periods=fast).mean()
        ema_slow = self.df[column].ewm(span=slow, adjust=False, min_periods=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
        histogram = macd_line - signal_line
        
        self.df['MACD'] = macd_line
        self.df['MACD_signal'] = signal_line
        self.df['MACD_histogram'] = histogram
        self.logger.debug(f"✅ MACD calculated. Latest - MACD: {macd_line.iloc[-1]:.4f}, Signal: {signal_line.iloc[-1]:.4f}")
        return macd_line, signal_line, histogram

    def calculate_stochastic(self, k_period: int = 14, d_period: int = 3, 
                           smooth_k: int = 1) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        if k_period < 1 or d_period < 1 or smooth_k < 1:
            raise ValueError("All periods must be >= 1")
        
        self.logger.info(f"📊 Calculating Stochastic (K={k_period}, D={d_period}, smooth={smooth_k})")
        low_min = self.df['low'].rolling(window=k_period, min_periods=k_period).min()
        high_max = self.df['high'].rolling(window=k_period, min_periods=k_period).max()
        stoch_k = 100 * self._safe_divide(self.df['close'] - low_min, high_max - low_min, fill_value=50)
        
        if smooth_k > 1:
            stoch_k = stoch_k.rolling(window=smooth_k, min_periods=smooth_k).mean()
        stoch_d = stoch_k.rolling(window=d_period, min_periods=d_period).mean()
        
        stoch_k = stoch_k.clip(0, 100)
        stoch_d = stoch_d.clip(0, 100)
        
        self.df['Stoch_%K'] = stoch_k
        self.df['Stoch_%D'] = stoch_d
        self.logger.debug(f"✅ Stochastic calculated. Latest - K: {stoch_k.iloc[-1]:.2f}, D: {stoch_d.iloc[-1]:.2f}")
        return stoch_k, stoch_d

    def calculate_roc(self, period: int = 12, column: str = 'close') -> pd.Series:
        """Calculate Rate of Change (ROC)."""
        if period < 1:
            raise ValueError("Period must be >= 1")
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found")
        
        self.logger.info(f"📊 Calculating ROC with period={period} on column={column}")
        price_shift = self.df[column].shift(period)
        roc = self._safe_divide(self.df[column] - price_shift, price_shift) * 100
        self.df[f'ROC_{period}'] = roc
        self.logger.debug(f"✅ ROC_{period} calculated. Latest: {roc.iloc[-1]:.2f}%")
        return roc

    def calculate_williams_r(self, period: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        if period < 1:
            raise ValueError("Period must be >= 1")
        
        self.logger.info(f"📊 Calculating Williams %R with period={period}")
        highest_high = self.df['high'].rolling(window=period, min_periods=period).max()
        lowest_low = self.df['low'].rolling(window=period, min_periods=period).min()
        williams_r = -100 * self._safe_divide(highest_high - self.df['close'], highest_high - lowest_low, fill_value=-50)
        williams_r = williams_r.clip(-100, 0)
        
        self.df[f'Williams_%R_{period}'] = williams_r
        self.logger.debug(f"✅ Williams %R_{period} calculated. Latest: {williams_r.iloc[-1]:.2f}")
        return williams_r

    def calculate_cci(self, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index (CCI)."""
        if period < 1:
            raise ValueError("Period must be >= 1")
        
        self.logger.info(f"📊 Calculating CCI with period={period}")
        typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        sma_tp = typical_price.rolling(window=period, min_periods=period).mean()
        
        def mad(x):
            return np.mean(np.abs(x - np.mean(x)))
        
        mean_deviation = typical_price.rolling(window=period, min_periods=period).apply(mad, raw=True)
        cci = self._safe_divide(typical_price - sma_tp, 0.015 * mean_deviation)
        self.df[f'CCI_{period}'] = cci
        self.logger.debug(f"✅ CCI_{period} calculated. Latest: {cci.iloc[-1]:.2f}")
        return cci

    def calculate_aroon(self, period: int = 25) -> Tuple[pd.Series, pd.Series]:
        """Calculate Aroon Indicator."""
        if period < 1:
            raise ValueError("Period must be >= 1")
        
        self.logger.info(f"📊 Calculating Aroon with period={period}")
        rolling_high_idx = self.df['high'].rolling(window=period, min_periods=period).apply(
            lambda x: len(x) - np.argmax(x) - 1, raw=True
        )
        rolling_low_idx = self.df['low'].rolling(window=period, min_periods=period).apply(
            lambda x: len(x) - np.argmin(x) - 1, raw=True
        )
        
        aroon_up = 100 * (period - rolling_high_idx) / period
        aroon_down = 100 * (period - rolling_low_idx) / period
        aroon_up = aroon_up.clip(0, 100)
        aroon_down = aroon_down.clip(0, 100)
        
        self.df[f'Aroon_Up_{period}'] = aroon_up
        self.df[f'Aroon_Down_{period}'] = aroon_down
        self.logger.debug(f"✅ Aroon calculated. Latest - Up: {aroon_up.iloc[-1]:.2f}, Down: {aroon_down.iloc[-1]:.2f}")
        return aroon_up, aroon_down

    def calculate_bollinger_bands(self, period: int = 20, std_dev: float = 2, 
                                 column: str = 'close') -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        if period < 2:
            raise ValueError("Period must be >= 2")
        if std_dev <= 0:
            raise ValueError("Standard deviation multiplier must be > 0")
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found")
        
        self.logger.info(f"📊 Calculating Bollinger Bands (period={period}, std_dev={std_dev})")
        middle_band = self.df[column].rolling(window=period, min_periods=period).mean()
        std = self.df[column].rolling(window=period, min_periods=period).std()
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)
        bb_width = self._safe_divide(upper_band - lower_band, middle_band)
        bb_percent_b = self._safe_divide(self.df[column] - lower_band, upper_band - lower_band)
        
        self.df['BB_middle'] = middle_band
        self.df['BB_upper'] = upper_band
        self.df['BB_lower'] = lower_band
        self.df['BB_width'] = bb_width
        self.df['BB_%B'] = bb_percent_b
        self.logger.debug(f"✅ Bollinger Bands calculated. Latest - Upper: {upper_band.iloc[-1]:.2f}, Middle: {middle_band.iloc[-1]:.2f}")
        return middle_band, upper_band, lower_band

    def calculate_atr(self, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR)."""
        if period < 1:
            raise ValueError("Period must be >= 1")
        
        self.logger.info(f"📊 Calculating ATR with period={period}")
        high_low = self.df['high'] - self.df['low']
        high_close_prev = abs(self.df['high'] - self.df['close'].shift(1))
        low_close_prev = abs(self.df['low'] - self.df['close'].shift(1))
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = true_range.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        
        self.df[f'ATR_{period}'] = atr
        self.logger.debug(f"✅ ATR_{period} calculated. Latest: {atr.iloc[-1]:.4f}")
        return atr

    def calculate_standard_deviation(self, period: int = 20, column: str = 'close') -> pd.Series:
        """Calculate Standard Deviation."""
        if period < 2:
            raise ValueError("Period must be >= 2")
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found")
        
        self.logger.info(f"📊 Calculating Standard Deviation with period={period}")
        std = self.df[column].rolling(window=period, min_periods=period).std()
        self.df[f'Std_Dev_{period}'] = std
        self.logger.debug(f"✅ Std_Dev_{period} calculated. Latest: {std.iloc[-1]:.4f}")
        return std

    def calculate_z_score(self, period: int = 20, column: str = 'close') -> pd.Series:
        """Calculate Z-Score."""
        if period < 2:
            raise ValueError("Period must be >= 2")
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found")
        
        self.logger.info(f"📊 Calculating Z-Score with period={period}")
        mean = self.df[column].rolling(window=period, min_periods=period).mean()
        std = self.df[column].rolling(window=period, min_periods=period).std()
        z_score = self._safe_divide(self.df[column] - mean, std)
        self.df[f'Z_Score_{period}'] = z_score
        self.logger.debug(f"✅ Z_Score_{period} calculated. Latest: {z_score.iloc[-1]:.2f}")
        return z_score

    def calculate_obv(self) -> pd.Series:
        """Calculate On-Balance Volume (OBV)."""
        self.logger.info("📊 Calculating On-Balance Volume (OBV)")
        price_diff = self.df['close'].diff()
        direction = np.sign(price_diff).fillna(0)
        obv = (direction * self.df['volume']).cumsum()
        self.df['OBV'] = obv
        self.logger.debug(f"✅ OBV calculated. Latest: {obv.iloc[-1]:.0f}")
        return obv

    def calculate_mfi(self, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index (MFI)."""
        if period < 1:
            raise ValueError("Period must be >= 1")
        
        self.logger.info(f"📊 Calculating MFI with period={period}")
        typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        money_flow = typical_price * self.df['volume']
        positive_flow = money_flow.where(typical_price.diff() > 0, 0)
        negative_flow = money_flow.where(typical_price.diff() < 0, 0)
        
        positive_flow_sum = positive_flow.rolling(window=period, min_periods=period).sum()
        negative_flow_sum = negative_flow.rolling(window=period, min_periods=period).sum()
        money_ratio = self._safe_divide(positive_flow_sum, negative_flow_sum, fill_value=0)
        mfi = 100 - (100 / (1 + money_ratio))
        mfi = mfi.clip(0, 100)
        
        self.df[f'MFI_{period}'] = mfi
        self.logger.debug(f"✅ MFI_{period} calculated. Latest: {mfi.iloc[-1]:.2f}")
        return mfi

    def calculate_cmf(self, period: int = 20) -> pd.Series:
        """Calculate Chaikin Money Flow (CMF)."""
        if period < 1:
            raise ValueError("Period must be >= 1")
        
        self.logger.info(f"📊 Calculating Chaikin Money Flow with period={period}")
        high_low_diff = self.df['high'] - self.df['low']
        multiplier = self._safe_divide(
            ((self.df['close'] - self.df['low']) - (self.df['high'] - self.df['close'])),
            high_low_diff, fill_value=0
        )
        money_flow_volume = multiplier * self.df['volume']
        cmf = self._safe_divide(
            money_flow_volume.rolling(window=period, min_periods=period).sum(),
            self.df['volume'].rolling(window=period, min_periods=period).sum()
        )
        cmf = cmf.clip(-1, 1)
        self.df[f'CMF_{period}'] = cmf
        self.logger.debug(f"✅ CMF_{period} calculated. Latest: {cmf.iloc[-1]:.4f}")
        return cmf

    def calculate_force_index(self, period: int = 13) -> pd.Series:
        """Calculate Force Index."""
        if period < 1:
            raise ValueError("Period must be >= 1")
        
        self.logger.info(f"📊 Calculating Force Index with period={period}")
        raw_force = (self.df['close'].diff() * self.df['volume'])
        force_index = raw_force.ewm(span=period, adjust=False, min_periods=period).mean()
        self.df[f'Force_Index_{period}'] = force_index
        self.logger.debug(f"✅ Force_Index_{period} calculated. Latest: {force_index.iloc[-1]:.2f}")
        return force_index

    def calculate_parabolic_sar(self, acceleration: float = 0.02, maximum: float = 0.2) -> pd.Series:
        """Calculate Parabolic SAR."""
        if acceleration <= 0 or maximum <= 0:
            raise ValueError("Acceleration and maximum must be > 0")
        if acceleration > maximum:
            raise ValueError("Acceleration must be <= maximum")
        
        self.logger.info(f"📊 Calculating Parabolic SAR (accel={acceleration}, max={maximum})")
        high = self.df['high'].values
        low = self.df['low'].values
        close = self.df['close'].values
        n = len(high)
        
        if n < 2:
            raise ValueError("Need at least 2 data points for Parabolic SAR")
        
        psar = np.zeros(n)
        trend = np.zeros(n)
        ep = np.zeros(n)
        af = np.zeros(n)
        
        psar[0] = high[0] if close[1] > close[0] else low[0]
        trend[0] = 1 if close[1] > close[0] else -1
        ep[0] = high[0] if trend[0] == 1 else low[0]
        af[0] = acceleration
        
        for i in range(1, n):
            psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
            if trend[i-1] == 1:
                if low[i] < psar[i]:
                    trend[i] = -1
                    psar[i] = ep[i-1]
                    ep[i] = low[i]
                    af[i] = acceleration
                else:
                    trend[i] = 1
                    if high[i] > ep[i-1]:
                        ep[i] = high[i]
                        af[i] = min(af[i-1] + acceleration, maximum)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
            else:
                if high[i] > psar[i]:
                    trend[i] = 1
                    psar[i] = ep[i-1]
                    ep[i] = high[i]
                    af[i] = acceleration
                else:
                    trend[i] = -1
                    if low[i] < ep[i-1]:
                        ep[i] = low[i]
                        af[i] = min(af[i-1] + acceleration, maximum)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
            
            if trend[i] == 1:
                lookback = [low[i-1]]
                if i >= 2:
                    lookback.append(low[i-2])
                psar[i] = min(psar[i], *lookback)
            else:
                lookback = [high[i-1]]
                if i >= 2:
                    lookback.append(high[i-2])
                psar[i] = max(psar[i], *lookback)
        
        psar_series = pd.Series(psar, index=self.df.index)
        self.df['Parabolic_SAR'] = psar_series
        self.df['PSAR_trend'] = pd.Series(trend, index=self.df.index)
        self.logger.debug(f"✅ Parabolic SAR calculated. Latest: {psar[-1]:.4f}")
        return psar_series

    def calculate_ichimoku(self, tenkan_period: int = 9, kijun_period: int = 26, 
                          senkou_period: int = 52) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """Calculate Ichimoku Cloud components."""
        if tenkan_period < 1 or kijun_period < 1 or senkou_period < 1:
            raise ValueError("All periods must be >= 1")
        
        self.logger.info(f"📊 Calculating Ichimoku Cloud (tenkan={tenkan_period}, kijun={kijun_period})")
        period_high_tenkan = self.df['high'].rolling(window=tenkan_period, min_periods=tenkan_period).max()
        period_low_tenkan = self.df['low'].rolling(window=tenkan_period, min_periods=tenkan_period).min()
        tenkan_sen = (period_high_tenkan + period_low_tenkan) / 2
        
        period_high_kijun = self.df['high'].rolling(window=kijun_period, min_periods=kijun_period).max()
        period_low_kijun = self.df['low'].rolling(window=kijun_period, min_periods=kijun_period).min()
        kijun_sen = (period_high_kijun + period_low_kijun) / 2
        
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)
        period_high_senkou = self.df['high'].rolling(window=senkou_period, min_periods=senkou_period).max()
        period_low_senkou = self.df['low'].rolling(window=senkou_period, min_periods=senkou_period).min()
        senkou_span_b = ((period_high_senkou + period_low_senkou) / 2).shift(kijun_period)
        chikou_span = self.df['close'].shift(-kijun_period)
        
        self.df['Ichimoku_Tenkan'] = tenkan_sen
        self.df['Ichimoku_Kijun'] = kijun_sen
        self.df['Ichimoku_Senkou_A'] = senkou_span_a
        self.df['Ichimoku_Senkou_B'] = senkou_span_b
        self.df['Ichimoku_Chikou'] = chikou_span
        self.logger.debug("✅ Ichimoku Cloud calculated")
        return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

    def calculate_adx(self, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Average Directional Index (ADX) with +DI and -DI."""
        if period < 1:
            raise ValueError("Period must be >= 1")
        
        self.logger.info(f"📊 Calculating ADX with period={period}")
        high_diff = self.df['high'].diff()
        low_diff = -self.df['low'].diff()
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        tr1 = self.df['high'] - self.df['low']
        tr2 = abs(self.df['high'] - self.df['close'].shift(1))
        tr3 = abs(self.df['low'] - self.df['close'].shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        plus_dm_smooth = pd.Series(plus_dm, index=self.df.index).ewm(
            alpha=1/period, min_periods=period, adjust=False
        ).mean()
        minus_dm_smooth = pd.Series(minus_dm, index=self.df.index).ewm(
            alpha=1/period, min_periods=period, adjust=False
        ).mean()
        true_range_smooth = true_range.ewm(
            alpha=1/period, min_periods=period, adjust=False
        ).mean()
        
        plus_di = 100 * self._safe_divide(plus_dm_smooth, true_range_smooth)
        minus_di = 100 * self._safe_divide(minus_dm_smooth, true_range_smooth)
        di_sum = plus_di + minus_di
        dx = 100 * self._safe_divide(abs(plus_di - minus_di), di_sum)
        adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        
        self.df[f'ADX_{period}'] = adx
        self.df[f'+DI_{period}'] = plus_di
        self.df[f'-DI_{period}'] = minus_di
        self.logger.debug(f"✅ ADX calculated. Latest - ADX: {adx.iloc[-1]:.2f}")
        return adx, plus_di, minus_di

    def calculate_keltner_channels(self, ema_period: int = 20, atr_period: int = 10, 
                                  atr_multiplier: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Keltner Channels."""
        if ema_period < 1 or atr_period < 1:
            raise ValueError("Periods must be >= 1")
        if atr_multiplier <= 0:
            raise ValueError("ATR multiplier must be > 0")
        
        self.logger.info(f"📊 Calculating Keltner Channels (EMA={ema_period}, ATR={atr_period})")
        middle_line = self.df['close'].ewm(span=ema_period, adjust=False, min_periods=ema_period).mean()
        high_low = self.df['high'] - self.df['low']
        high_close = abs(self.df['high'] - self.df['close'].shift(1))
        low_close = abs(self.df['low'] - self.df['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.ewm(alpha=1/atr_period, min_periods=atr_period, adjust=False).mean()
        
        upper_channel = middle_line + (atr_multiplier * atr)
        lower_channel = middle_line - (atr_multiplier * atr)
        
        self.df['KC_middle'] = middle_line
        self.df['KC_upper'] = upper_channel
        self.df['KC_lower'] = lower_channel
        self.logger.debug(f"✅ Keltner Channels calculated")
        return middle_line, upper_channel, lower_channel

    def calculate_donchian_channels(self, period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Donchian Channels."""
        if period < 1:
            raise ValueError("Period must be >= 1")
        
        self.logger.info(f"📊 Calculating Donchian Channels with period={period}")
        upper_band = self.df['high'].rolling(window=period, min_periods=period).max()
        lower_band = self.df['low'].rolling(window=period, min_periods=period).min()
        middle_band = (upper_band + lower_band) / 2
        
        self.df['DC_upper'] = upper_band
        self.df['DC_lower'] = lower_band
        self.df['DC_middle'] = middle_band
        self.logger.debug(f"✅ Donchian Channels calculated")
        return upper_band, lower_band, middle_band

    def calculate_supertrend(self, period: int = 10, multiplier: float = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Supertrend Indicator."""
        if period < 1:
            raise ValueError("Period must be >= 1")
        if multiplier <= 0:
            raise ValueError("Multiplier must be > 0")
        
        self.logger.info(f"📊 Calculating Supertrend (period={period}, multiplier={multiplier})")
        high_low = self.df['high'] - self.df['low']
        high_close = abs(self.df['high'] - self.df['close'].shift(1))
        low_close = abs(self.df['low'] - self.df['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        
        hl_avg = (self.df['high'] + self.df['low']) / 2
        basic_upper = hl_avg + (multiplier * atr)
        basic_lower = hl_avg - (multiplier * atr)
        
        final_upper = basic_upper.copy()
        final_lower = basic_lower.copy()
        
        for i in range(1, len(self.df)):
            if basic_upper.iloc[i] < final_upper.iloc[i-1] or self.df['close'].iloc[i-1] > final_upper.iloc[i-1]:
                final_upper.iloc[i] = basic_upper.iloc[i]
            else:
                final_upper.iloc[i] = final_upper.iloc[i-1]
            
            if basic_lower.iloc[i] > final_lower.iloc[i-1] or self.df['close'].iloc[i-1] < final_lower.iloc[i-1]:
                final_lower.iloc[i] = basic_lower.iloc[i]
            else:
                final_lower.iloc[i] = final_lower.iloc[i-1]
        
        supertrend = pd.Series(np.nan, index=self.df.index)
        direction = pd.Series(0, index=self.df.index)
        start_idx = period - 1
        
        if len(self.df) <= start_idx:
            self.logger.warning("⚠️ Insufficient data for Supertrend")
            self.df['Supertrend'] = supertrend
            self.df['Supertrend_direction'] = direction
            return supertrend, direction
        
        if self.df['close'].iloc[start_idx] > self.df['open'].iloc[start_idx]:
            supertrend.iloc[start_idx] = final_lower.iloc[start_idx]
            direction.iloc[start_idx] = 1
        else:
            supertrend.iloc[start_idx] = final_upper.iloc[start_idx]
            direction.iloc[start_idx] = -1
        
        for i in range(start_idx + 1, len(self.df)):
            prev_st = supertrend.iloc[i-1]
            if direction.iloc[i-1] == 1:
                if self.df['close'].iloc[i] < prev_st:
                    supertrend.iloc[i] = final_upper.iloc[i]
                    direction.iloc[i] = -1
                else:
                    supertrend.iloc[i] = max(final_lower.iloc[i], prev_st)
                    direction.iloc[i] = 1
            else:
                if self.df['close'].iloc[i] > prev_st:
                    supertrend.iloc[i] = final_lower.iloc[i]
                    direction.iloc[i] = 1
                else:
                    supertrend.iloc[i] = min(final_upper.iloc[i], prev_st)
                    direction.iloc[i] = -1
        
        self.df['Supertrend'] = supertrend
        self.df['Supertrend_direction'] = direction
        self.logger.debug(f"✅ Supertrend calculated. Latest: {supertrend.iloc[-1]:.4f}")
        return supertrend, direction



    def calculate_pivot_points(self, timeframe: str = 'daily') -> Dict[str, pd.Series]:
        """Calculate Classic Pivot Points."""
        if timeframe not in ['daily', 'bar']:
            raise ValueError("timeframe must be 'daily' or 'bar'")
        
        self.logger.info(f"📊 Calculating Pivot Points (timeframe={timeframe})")
        
        if timeframe == 'daily':
            # Aggregate previous day's OHLC
            daily = self.df.groupby(self.df['timestamp'].dt.date).agg({
                'high': 'max', 'low': 'min', 'close': 'last'
            }).shift(1)
            
            # Map the previous day's OHLC to each row
            daily_index = self.df['timestamp'].dt.date
            prev_high_dict = daily['high'].to_dict()
            prev_low_dict = daily['low'].to_dict()
            prev_close_dict = daily['close'].to_dict()

            prev_high = daily_index.map(prev_high_dict)
            prev_low = daily_index.map(prev_low_dict)
            prev_close = daily_index.map(prev_close_dict)
        else:
            prev_high = self.df['high'].shift(1)
            prev_low = self.df['low'].shift(1)
            prev_close = self.df['close'].shift(1)
        
        pp = (prev_high + prev_low + prev_close) / 3
        r1 = 2 * pp - prev_low
        s1 = 2 * pp - prev_high
        r2 = pp + (prev_high - prev_low)
        s2 = pp - (prev_high - prev_low)
        r3 = prev_high + 2 * (pp - prev_low)
        s3 = prev_low - 2 * (prev_high - pp)
        
        self.df['PP'] = pp
        self.df['R1'] = r1
        self.df['R2'] = r2
        self.df['R3'] = r3
        self.df['S1'] = s1
        self.df['S2'] = s2
        self.df['S3'] = s3
        
        self.logger.debug("✅ Pivot Points calculated")
        
        return {
            'PP': pp, 'R1': r1, 'R2': r2, 'R3': r3,
            'S1': s1, 'S2': s2, 'S3': s3
        }

    def calculate_pandas_ta(self, indicator: str, **kwargs) -> pd.Series:
        """Calculate indicator using pandas-ta for validation (Task 3.1)."""
        try:
            import pandas_ta_classic as ta
        except ImportError:
            self.logger.error("❌ pandas-ta not installed. Run `pip install pandas-ta`")
            raise
        
        self.logger.info(f"📊 Calculating {indicator} with pandas-ta")
        if indicator.lower() == 'sma':
            return ta.sma(self.df['close'], length=kwargs.get('period', 20))
        elif indicator.lower() == 'ema':
            return ta.ema(self.df['close'], length=kwargs.get('period', 20))
        elif indicator.lower() == 'rsi':
            return ta.rsi(self.df['close'], length=kwargs.get('period', 14))
        else:
            raise ValueError(f"Unsupported pandas-ta indicator: {indicator}")

    def calculate_all_indicators(self, indicators: Optional[List[str]] = None, 
                                sma_periods: List[int] = [20, 50, 200],
                                ema_periods: List[int] = [12, 26, 50],
                                rsi_period: int = 14,
                                macd_params: Tuple[int, int, int] = (12, 26, 9),
                                bb_params: Tuple[int, float] = (20, 2),
                                atr_period: int = 14,
                                adx_period: int = 14,
                                aroon_period: int = 25,
                                include_advanced: bool = True,
                                include_volume: bool = True,
                                include_volatility: bool = True) -> pd.DataFrame:
        """
        Calculate all technical indicators (Task 3.8).
        """
        if self.df is None:
            raise ValueError("No DataFrame loaded. Use fetch_klines or provide a DataFrame")
        
        self.logger.info("="*80)
        self.logger.info("🚀 CALCULATING ALL TECHNICAL INDICATORS")
        self.logger.info("="*80)
        
        try:
            if indicators is None:
                indicators = ['sma', 'ema', 'wma', 'vwap', 'rsi', 'macd', 'stochastic', 'roc',
                             'williams_r', 'cci', 'aroon', 'bollinger', 'atr', 'std_dev', 'z_score',
                             'obv', 'mfi', 'cmf', 'force_index', 'parabolic_sar', 'ichimoku',
                             'adx', 'keltner', 'donchian', 'supertrend', 'pivot_points']
            
            for ind in indicators:
                ind = ind.lower()
                if ind == 'sma':
                    for period in sma_periods:
                        self.calculate_sma(period)
                elif ind == 'ema':
                    for period in ema_periods:
                        self.calculate_ema(period)
                elif ind == 'wma':
                    self.calculate_wma(20)
                elif ind == 'vwap':
                    self.calculate_vwap()
                elif ind == 'rsi':
                    self.calculate_rsi(rsi_period)
                elif ind == 'macd':
                    self.calculate_macd(*macd_params)
                elif ind == 'stochastic':
                    self.calculate_stochastic()
                elif ind == 'roc':
                    self.calculate_roc(12)
                    self.calculate_roc(25)
                elif ind == 'williams_r':
                    self.calculate_williams_r()
                elif ind == 'cci':
                    self.calculate_cci()
                elif ind == 'aroon':
                    self.calculate_aroon(aroon_period)
                elif ind == 'bollinger' and include_volatility:
                    self.calculate_bollinger_bands(*bb_params)
                elif ind == 'atr' and include_volatility:
                    self.calculate_atr(atr_period)
                elif ind == 'std_dev' and include_volatility:
                    self.calculate_standard_deviation()
                elif ind == 'z_score' and include_volatility:
                    self.calculate_z_score()
                elif ind == 'obv' and include_volume:
                    self.calculate_obv()
                elif ind == 'mfi' and include_volume:
                    self.calculate_mfi()
                elif ind == 'cmf' and include_volume:
                    self.calculate_cmf()
                elif ind == 'force_index' and include_volume:
                    self.calculate_force_index()
                elif ind == 'parabolic_sar' and include_advanced:
                    self.calculate_parabolic_sar()
                elif ind == 'ichimoku' and include_advanced:
                    self.calculate_ichimoku()
                elif ind == 'adx' and include_advanced:
                    self.calculate_adx(adx_period)
                elif ind == 'keltner' and include_volatility:
                    self.calculate_keltner_channels()
                elif ind == 'donchian' and include_volatility:
                    self.calculate_donchian_channels()
                elif ind == 'supertrend' and include_advanced:
                    self.calculate_supertrend()
                elif ind == 'pivot_points' and include_advanced:
                    self.calculate_pivot_points()
            
            self.logger.info(f"✅ ALL INDICATORS CALCULATED. DataFrame shape: {self.df.shape}")
            return self.df
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating indicators: {e}", exc_info=True)
            raise

    def get_dataframe(self) -> pd.DataFrame:
        """Return DataFrame with all indicators."""
        if self.df is None:
            raise ValueError("No DataFrame loaded")
        return self.df.copy()

    def get_indicator_summary(self) -> pd.DataFrame:
        """Get summary statistics for all indicators."""
        if self.df is None:
            raise ValueError("No DataFrame loaded")
        
        indicator_cols = [col for col in self.df.columns 
                         if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
        if not indicator_cols:
            self.logger.warning("⚠️ No indicators found")
            return pd.DataFrame()
        
        summary_data = []
        for col in indicator_cols:
            latest_value = self.df[col].iloc[-1] if not self.df[col].isna().all() else np.nan
            summary_data.append({
                'Indicator': col,
                'Latest': latest_value,
                'Mean': self.df[col].mean(),
                'Std_Dev': self.df[col].std(),
                'Min': self.df[col].min(),
                'Max': self.df[col].max(),
                'Valid_Count': self.df[col].notna().sum(),
                'Coverage_%': (self.df[col].notna().sum() / len(self.df)) * 100
            })
        return pd.DataFrame(summary_data)

    def get_latest_signals(self) -> Dict[str, any]:
        """Get latest indicator values and trading signals."""
        if self.df is None:
            raise ValueError("No DataFrame loaded")
        
        latest = self.df.iloc[-1]
        signals = {
            'price': latest['close'],
            'timestamp': latest['timestamp']
        }
        
        if 'SMA_20' in self.df.columns and 'SMA_50' in self.df.columns:
            signals['sma_trend'] = 'BULLISH' if latest['SMA_20'] > latest['SMA_50'] else 'BEARISH'
        if 'EMA_12' in self.df.columns and 'EMA_26' in self.df.columns:
            signals['ema_trend'] = 'BULLISH' if latest['EMA_12'] > latest['EMA_26'] else 'BEARISH'
        if 'RSI_14' in self.df.columns:
            rsi = latest['RSI_14']
            signals['rsi'] = rsi
            signals['rsi_signal'] = 'OVERBOUGHT' if rsi > 70 else 'OVERSOLD' if rsi < 30 else 'NEUTRAL'
        if 'MACD_histogram' in self.df.columns:
            signals['macd_histogram'] = latest['MACD_histogram']
            signals['macd_signal'] = 'BULLISH' if latest['MACD_histogram'] > 0 else 'BEARISH'
        if 'BB_%B' in self.df.columns:
            bb_b = latest['BB_%B']
            signals['bb_percent_b'] = bb_b
            signals['bb_signal'] = 'ABOVE_UPPER' if bb_b > 1 else 'BELOW_LOWER' if bb_b < 0 else 'WITHIN_BANDS'
        if 'ADX_14' in self.df.columns:
            signals['adx'] = latest['ADX_14']
            signals['trend_strength'] = 'STRONG' if latest['ADX_14'] > 25 else 'WEAK'
        if 'Aroon_Up_25' in self.df.columns:
            signals['aroon_signal'] = 'BULLISH' if latest['Aroon_Up_25'] > latest['Aroon_Down_25'] else 'BEARISH'
        
        return signals

    def export_to_csv(self, filename: str) -> None:
        """Export DataFrame to CSV."""
        if self.df is None:
            raise ValueError("No DataFrame loaded")
        self.df.to_csv(filename, index=False)
        self.logger.info(f"💾 Data exported to {filename}")

    def set_timestamp_index(self) -> None:
        """Set timestamp as DataFrame index (Task 3.4)."""
        if self.df is None:
            raise ValueError("No DataFrame loaded")
        self.df.set_index('timestamp', inplace=True)
        self.logger.info("✅ Set timestamp as DataFrame index")

if __name__ == "__main__":
    logger.info("="*80)
    logger.info("🧪 TESTING TECHNICAL INDICATORS MODULE")
    logger.info("="*80)
    
    try:
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=250, freq='D')
        base_price = 100
        returns = np.random.randn(250) * 0.02
        trend = np.linspace(0, 20, 250)
        close_prices = base_price + np.cumsum(returns) + trend
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': close_prices + np.random.randn(250) * 0.5,
            'high': close_prices + np.abs(np.random.randn(250) * 1.5),
            'low': close_prices - np.abs(np.abs(np.random.randn(250) * 1.5)),
            'close': close_prices,
            'volume': np.random.randint(100000, 1000000, 250)
        })
        df['high'] = df[['high', 'close', 'open']].max(axis=1)
        df['low'] = df[['low', 'close', 'open']].min(axis=1)
        
        # Initialize and test
        ti = TechnicalIndicators(df, log_level='INFO')
        df_with_indicators = ti.calculate_all_indicators()
        
        # Display results
        print("\n📊 Indicator Summary:")
        summary = ti.get_indicator_summary()
        print(f"Total Indicators: {len(summary)}")
        print(summary[['Indicator', 'Latest', 'Coverage_%']].to_string(index=False))
        
        print("\n🚦 Latest Signals:")
        signals = ti.get_latest_signals()
        for key, value in signals.items():
            print(f"{key}: {value}")
        
        logger.info("✅ Testing completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        raise