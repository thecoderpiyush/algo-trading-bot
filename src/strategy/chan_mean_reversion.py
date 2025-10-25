"""
Chan Mean Reversion Strategy - SOLID Principles Implementation
Based on: Ernest Chan "Algorithmic Trading" Chapter 2

File Location: src/strategy/chan_mean_reversion.py

Design Patterns Used:
- Strategy Pattern: BaseStrategy inheritance
- Template Method: Indicator calculation steps
- Single Responsibility: Each class does ONE thing

Author: Piyush | Date: Oct 19, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging
from sklearn.linear_model import LinearRegression

try:
    from .base_strategy import BaseStrategy
except ImportError:
    from base_strategy import BaseStrategy


# ============================================================================
# 1. BOLLINGER BAND CALCULATOR (Single Responsibility)
# ============================================================================
class BollingerBandCalculator:
    """
    Calculates Bollinger Bands
    
    Single Responsibility: Band calculation only
    """
    
    def __init__(self, period: int = 20, std_multiplier: float = 2.0):
        self.period = period
        self.std_multiplier = std_multiplier
    
    def calculate(self, prices: pd.Series) -> pd.DataFrame:
        """
        Calculate Bollinger Bands
        
        Formula (Chan Ch. 2, p. 55):
            Middle Band = SMA(period)
            Upper Band = Middle + (std_multiplier × σ)
            Lower Band = Middle - (std_multiplier × σ)
        
        Returns:
            DataFrame with bb_middle, bb_upper, bb_lower
        """
        middle = prices.rolling(window=self.period).mean()
        std = prices.rolling(window=self.period).std()
        
        return pd.DataFrame({
            'bb_middle': middle,
            'bb_upper': middle + (self.std_multiplier * std),
            'bb_lower': middle - (self.std_multiplier * std),
            'bb_width': (2 * self.std_multiplier * std) / middle
        })


# ============================================================================
# 2. Z-SCORE CALCULATOR (Single Responsibility)
# ============================================================================
class ZScoreCalculator:
    """
    Calculates rolling Z-score
    
    Single Responsibility: Z-score computation
    """
    
    def __init__(self, lookback: int = 30):
        self.lookback = lookback
    
    def calculate(self, prices: pd.Series) -> pd.Series:
        """
        Calculate Z-score
        
        Formula (Chan Ch. 2, Eq. 2.10):
            Z = (Price - μ) / σ
        
        Returns:
            Z-score series
        """
        rolling_mean = prices.rolling(window=self.lookback).mean()
        rolling_std = prices.rolling(window=self.lookback).std()
        
        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)
        
        zscore = (prices - rolling_mean) / rolling_std
        return zscore


# ============================================================================
# 3. HALF-LIFE ESTIMATOR (Single Responsibility)
# ============================================================================
class HalfLifeEstimator:
    """
    Estimates half-life of mean reversion
    
    Single Responsibility: Ornstein-Uhlenbeck half-life calculation
    """
    
    def calculate_rolling(self, prices: pd.Series, window: int = 100) -> pd.Series:
        """
        Calculate rolling half-life
        
        Ornstein-Uhlenbeck Process (Chan Ch. 2, p. 58, Eq. 2.13):
            dP = θ(μ - P)dt + σdW
            Half-life = -ln(2) / θ
        
        Method: Regress ΔP_t on P_{t-1} to estimate θ
        
        Returns:
            Half-life series (in periods)
        """
        half_lives = []
        
        for i in range(window, len(prices)):
            price_window = prices.iloc[i-window:i].values
            
            # Price changes and lagged prices
            lag_prices = price_window[:-1]
            price_changes = np.diff(price_window)
            
            if len(lag_prices) > 10:
                try:
                    # Regression: ΔP = θ * P_{t-1} + α
                    model = LinearRegression()
                    X = lag_prices.reshape(-1, 1)
                    y = price_changes
                    model.fit(X, y)
                    
                    theta = model.coef_[0]
                    
                    # Calculate half-life (θ should be negative for mean reversion)
                    if theta < 0:
                        half_life = -np.log(2) / theta
                        half_lives.append(half_life)
                    else:
                        half_lives.append(np.inf)  # Not mean reverting
                
                except Exception:
                    half_lives.append(np.nan)
            else:
                half_lives.append(np.nan)
        
        # Pad with NaN for initial window
        result = [np.nan] * window + half_lives
        return pd.Series(result, index=prices.index)


# ============================================================================
# 4. HURST EXPONENT CALCULATOR (Single Responsibility)
# ============================================================================
class HurstExponentCalculator:
    """
    Calculates Hurst exponent for mean reversion detection
    
    Single Responsibility: Hurst calculation via R/S analysis
    
    Interpretation:
        H < 0.5: Mean reverting (use this strategy!)
        H = 0.5: Random walk
        H > 0.5: Trending
    """
    
    def calculate_rolling(self, prices: pd.Series, window: int = 100) -> pd.Series:
        """Calculate rolling Hurst exponent"""
        hurst_values = []
        
        for i in range(window, len(prices)):
            price_window = prices.iloc[i-window:i].values
            hurst = self._calculate_hurst(price_window)
            hurst_values.append(hurst)
        
        result = [np.nan] * window + hurst_values
        return pd.Series(result, index=prices.index)
    
    def _calculate_hurst(self, prices: np.ndarray) -> float:
        """
        Calculate Hurst exponent using R/S analysis
        
        Returns:
            Hurst exponent (0-1)
        """
        lags = range(2, min(20, len(prices) // 2))
        tau = []
        
        for lag in lags:
            std_dev = np.std(np.subtract(prices[lag:], prices[:-lag]))
            tau.append(std_dev)
        
        if len(tau) > 0:
            # Linear regression on log-log plot
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0]
        
        return 0.5  # Default to random walk


# ============================================================================
# 5. MEAN REVERSION SIGNAL GENERATOR (Single Responsibility)
# ============================================================================
class MeanReversionSignalGenerator:
    """
    Generates BUY/SELL signals based on mean reversion
    
    Single Responsibility: Signal generation logic
    """
    
    def __init__(self, entry_threshold: float = 2.0, exit_threshold: float = 0.5):
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
    
    def generate(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals
        
        Logic (Chan Ch. 2, p. 56):
        - BUY: Z-score < -entry_threshold AND price < BB_lower AND mean_reverting
        - SELL: Z-score > entry_threshold AND price > BB_upper AND mean_reverting
        - EXIT: |Z-score| < exit_threshold
        
        Returns:
            Signal series (1=BUY, -1=SELL, 0=HOLD)
        """
        signals = pd.Series(0, index=df.index)
        
        # Mean reverting regime filter
        mean_reverting = (df['hurst'] < 0.5) & (df['half_life'] < 30)
        
        # Entry conditions
        buy_condition = (
            (df['zscore'] < -self.entry_threshold) &
            (df['close'] < df['bb_lower']) &
            mean_reverting
        )
        
        sell_condition = (
            (df['zscore'] > self.entry_threshold) &
            (df['close'] > df['bb_upper']) &
            mean_reverting
        )
        
        # Apply signals
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        
        return signals


# ============================================================================
# 6. POSITION MANAGER (Single Responsibility)
# ============================================================================
class PositionManager:
    """
    Manages position state and transitions
    
    Single Responsibility: Position tracking and exit logic
    """
    
    def __init__(self, exit_threshold: float = 0.5):
        self.exit_threshold = exit_threshold
        self.current_position = 0
    
    def update_positions(self, df: pd.DataFrame) -> pd.Series:
        """
        Update positions based on signals and exit conditions
        
        Returns:
            Position series (1=long, -1=short, 0=flat)
        """
        positions = pd.Series(0, index=df.index)
        
        for i in range(len(df)):
            signal = df['signal'].iloc[i]
            zscore = df['zscore'].iloc[i]
            
            # Entry signals
            if signal == 1 and self.current_position == 0:
                self.current_position = 1
            elif signal == -1 and self.current_position == 0:
                self.current_position = -1
            
            # Exit logic (Z-score returns to mean)
            elif abs(zscore) < self.exit_threshold:
                self.current_position = 0
            
            positions.iloc[i] = self.current_position
        
        return positions
    
    def reset(self):
        """Reset position state"""
        self.current_position = 0


# ============================================================================
# 7. MAIN STRATEGY CLASS (Coordinates all components)
# ============================================================================
class ChanMeanReversion(BaseStrategy):
    """
    Chan Mean Reversion Strategy - Coordinator
    
    Follows Open/Closed Principle: Open for extension, closed for modification
    Uses Dependency Injection: Components injected or created
    
    Strategy for RANGING markets using Bollinger Bands
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config or {})
        
        # Configuration
        bb_period = self.config.get('bb_period', 20)
        bb_std = self.config.get('bb_std', 2.0)
        zscore_lookback = self.config.get('zscore_lookback', 30)
        entry_threshold = self.config.get('entry_threshold', 2.0)
        exit_threshold = self.config.get('exit_threshold', 0.5)
        
        # Dependency Injection: Create components
        self.bb_calculator = BollingerBandCalculator(bb_period, bb_std)
        self.zscore_calculator = ZScoreCalculator(zscore_lookback)
        self.halflife_estimator = HalfLifeEstimator()
        self.hurst_calculator = HurstExponentCalculator()
        self.signal_generator = MeanReversionSignalGenerator(entry_threshold, exit_threshold)
        self.position_manager = PositionManager(exit_threshold)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initialized with BB({bb_period},{bb_std}), Z-entry={entry_threshold}")
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all indicators using injected components
        
        Template Method Pattern: Define skeleton, components fill details
        """
        df = df.copy()
        
        # 1. Bollinger Bands
        bb_data = self.bb_calculator.calculate(df['close'])
        df = pd.concat([df, bb_data], axis=1)
        
        # 2. Z-Score
        df['zscore'] = self.zscore_calculator.calculate(df['close'])
        
        # 3. Half-Life
        df['half_life'] = self.halflife_estimator.calculate_rolling(df['close'])
        
        # 4. Hurst Exponent
        df['hurst'] = self.hurst_calculator.calculate_rolling(df['close'])
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals
        
        Coordinates all components following Single Responsibility
        """
        # Validate input
        if not self.validate_data(df):
            raise ValueError("Invalid input data")
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Generate signals
        df['signal'] = self.signal_generator.generate(df)
        
        # Manage positions
        self.position_manager.reset()
        df['position'] = self.position_manager.update_positions(df)
        
        # Calculate performance
        self.calculate_performance(df)
        
        self.logger.info(f"Generated {self.metrics['total_signals']} mean reversion signals")
        
        return df


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Configuration
    config = {
        'bb_period': 20,
        'bb_std': 2.0,
        'zscore_lookback': 30,
        'entry_threshold': 2.0,
        'exit_threshold': 0.5
    }
    
    # Create strategy
    strategy = ChanMeanReversion(config)
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=500, freq='1h')
    prices = 50000 + np.cumsum(np.random.randn(500) * 200)
    
    df = pd.DataFrame({
        'open': prices * 0.999,
        'high': prices * 1.002,
        'low': prices * 0.998,
        'close': prices,
        'volume': np.random.randint(100, 1000, 500)
    }, index=dates)
    
    # Generate signals
    result = strategy.generate_signals(df)
    
    print("\n" + "="*60)
    print("CHAN MEAN REVERSION - SIGNAL GENERATION")
    print("="*60)
    print(f"Total Signals: {strategy.metrics['total_signals']}")
    print(f"Buy Signals:   {strategy.metrics['buy_signals']}")
    print(f"Sell Signals:  {strategy.metrics['sell_signals']}")
    print("="*60)
    
    # Show latest signals
    print("\nLatest 5 Signals:")
    print(result[['close', 'zscore', 'bb_lower', 'bb_upper', 'signal', 'position']].tail())