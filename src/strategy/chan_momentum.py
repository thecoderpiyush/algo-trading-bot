"""
Chan Momentum Strategy - SOLID Principles Implementation
Based on: Ernest Chan "Algorithmic Trading" Chapter 4
Time-Series Momentum for Crypto (adapted from cross-sectional)

File Location: src/strategy/chan_momentum.py

Design Patterns Used:
- Strategy Pattern: BaseStrategy inheritance
- Single Responsibility: Each class does ONE thing
- Dependency Injection: Components are injected
- Template Method: Indicator calculation steps

Author: Piyush | Date: Oct 19, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging

try:
    from .base_strategy import BaseStrategy
except ImportError:
    from base_strategy import BaseStrategy


# ============================================================================
# 1. MOMENTUM CALCULATOR (Single Responsibility)
# ============================================================================
class MomentumCalculator:
    """
    Calculates momentum returns across multiple timeframes
    
    Single Responsibility: Momentum return calculation
    """
    
    def __init__(self, short: int = 20, medium: int = 60, long: int = 120):
        self.short = short
        self.medium = medium
        self.long = long
    
    def calculate(self, prices: pd.Series) -> pd.DataFrame:
        """
        Calculate momentum returns
        
        Formula (Chan Ch. 4, p. 129, Eq. 4.1):
            Return_n = (P_t - P_{t-n}) / P_{t-n}
        
        Returns:
            DataFrame with return_short, return_medium, return_long
        """
        return pd.DataFrame({
            'return_short': prices.pct_change(self.short),
            'return_medium': prices.pct_change(self.medium),
            'return_long': prices.pct_change(self.long)
        })


# ============================================================================
# 2. MOVING AVERAGE TREND CALCULATOR (Single Responsibility)
# ============================================================================
class TrendCalculator:
    """
    Calculates moving averages for trend confirmation
    
    Single Responsibility: Trend direction identification
    """
    
    def __init__(self, short: int = 20, medium: int = 60, long: int = 120):
        self.short = short
        self.medium = medium
        self.long = long
    
    def calculate(self, prices: pd.Series) -> pd.DataFrame:
        """
        Calculate trend MAs
        
        Returns:
            DataFrame with ma_short, ma_medium, ma_long
        """
        return pd.DataFrame({
            'ma_short': prices.rolling(window=self.short).mean(),
            'ma_medium': prices.rolling(window=self.medium).mean(),
            'ma_long': prices.rolling(window=self.long).mean()
        })


# ============================================================================
# 3. RSI CALCULATOR (Single Responsibility)
# ============================================================================
class RSICalculator:
    """
    Calculates Relative Strength Index
    
    Single Responsibility: RSI calculation
    """
    
    def __init__(self, period: int = 14):
        self.period = period
    
    def calculate(self, prices: pd.Series) -> pd.Series:
        """
        Calculate RSI
        
        Formula:
            RSI = 100 - (100 / (1 + RS))
            where RS = Average Gain / Average Loss
        
        Returns:
            RSI series (0-100)
        """
        delta = prices.diff()
        
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()
        
        avg_loss = avg_loss.replace(0, np.nan)
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi


# ============================================================================
# 4. MOMENTUM SCORE CALCULATOR (Single Responsibility)
# ============================================================================
class MomentumScoreCalculator:
    """
    Calculates composite momentum score
    
    Single Responsibility: Multi-timeframe momentum scoring
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        # Default weights favor shorter timeframes for crypto
        self.weights = weights or {
            'short': 0.5,
            'medium': 0.3,
            'long': 0.2
        }
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate composite momentum score
        
        Formula:
            Score = w_s × R_short + w_m × R_medium + w_l × R_long
        
        Returns:
            Momentum score series
        """
        score = (
            self.weights['short'] * df['return_short'] +
            self.weights['medium'] * df['return_medium'] +
            self.weights['long'] * df['return_long']
        )
        
        return score


# ============================================================================
# 5. MOMENTUM RANK CALCULATOR (Single Responsibility)
# ============================================================================
class MomentumRankCalculator:
    """
    Calculates momentum percentile rank
    
    Single Responsibility: Momentum strength ranking
    """
    
    def __init__(self, lookback: int = 60):
        self.lookback = lookback
    
    def calculate(self, momentum_score: pd.Series) -> pd.Series:
        """
        Calculate rolling percentile rank
        
        Returns:
            Percentile rank (0-100)
        """
        def percentile_rank(x):
            if len(x) < 2:
                return 50.0
            current = x.iloc[-1]
            min_val = x.min()
            max_val = x.max()
            
            if max_val == min_val:
                return 50.0
            
            return ((current - min_val) / (max_val - min_val)) * 100
        
        rank = momentum_score.rolling(
            window=self.lookback
        ).apply(percentile_rank, raw=False)
        
        return rank


# ============================================================================
# 6. MOMENTUM SIGNAL GENERATOR (Single Responsibility)
# ============================================================================
class MomentumSignalGenerator:
    """
    Generates BUY/SELL signals based on momentum
    
    Single Responsibility: Momentum-based signal logic
    """
    
    def __init__(self, 
                 momentum_threshold: float = 0.05,
                 rsi_overbought: float = 70,
                 rsi_oversold: float = 30):
        self.momentum_threshold = momentum_threshold
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
    
    def generate(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate momentum signals
        
        Logic (Chan Ch. 4, adapted):
        
        BUY CONDITIONS (Strong Upward Momentum):
        - return_short > threshold (e.g., 5%)
        - return_medium > 0
        - Price > MA_short > MA_medium (trend alignment)
        - RSI < overbought (not overextended)
        - momentum_score > 0
        
        SELL CONDITIONS (Strong Downward Momentum):
        - return_short < -threshold
        - return_medium < 0
        - Price < MA_short < MA_medium
        - RSI > oversold
        - momentum_score < 0
        
        Returns:
            Signal series (1=BUY, -1=SELL, 0=HOLD)
        """
        signals = pd.Series(0, index=df.index)
        
        # Strong upward momentum
        buy_conditions = (
            (df['return_short'] > self.momentum_threshold) &
            (df['return_medium'] > 0) &
            (df['close'] > df['ma_short']) &
            (df['ma_short'] > df['ma_medium']) &
            (df['rsi'] < self.rsi_overbought) &
            (df['momentum_score'] > 0)
        )
        
        # Strong downward momentum
        sell_conditions = (
            (df['return_short'] < -self.momentum_threshold) &
            (df['return_medium'] < 0) &
            (df['close'] < df['ma_short']) &
            (df['ma_short'] < df['ma_medium']) &
            (df['rsi'] > self.rsi_oversold) &
            (df['momentum_score'] < 0)
        )
        
        signals[buy_conditions] = 1
        signals[sell_conditions] = -1
        
        return signals


# ============================================================================
# 7. MOMENTUM EXIT MANAGER (Single Responsibility)
# ============================================================================
class MomentumExitManager:
    """
    Manages exit conditions for momentum trades
    
    Single Responsibility: Exit logic only
    """
    
    def __init__(self, rsi_overbought: float = 70, rsi_oversold: float = 30):
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
    
    def should_exit_long(self, df: pd.DataFrame, index: int) -> bool:
        """
        Check if should exit long position
        
        Exit Conditions:
        - Momentum weakens (return_short < 0)
        - OR RSI > overbought (overextended)
        
        Returns:
            True if should exit
        """
        return (
            df['return_short'].iloc[index] < 0 or
            df['rsi'].iloc[index] > self.rsi_overbought
        )
    
    def should_exit_short(self, df: pd.DataFrame, index: int) -> bool:
        """
        Check if should exit short position
        
        Exit Conditions:
        - Momentum reverses (return_short > 0)
        - OR RSI < oversold
        
        Returns:
            True if should exit
        """
        return (
            df['return_short'].iloc[index] > 0 or
            df['rsi'].iloc[index] < self.rsi_oversold
        )


# ============================================================================
# 8. POSITION MANAGER (Single Responsibility)
# ============================================================================
class MomentumPositionManager:
    """
    Manages position state with exit logic
    
    Single Responsibility: Position tracking with exits
    """
    
    def __init__(self, exit_manager: MomentumExitManager):
        self.exit_manager = exit_manager
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
            
            # Entry signals
            if signal == 1 and self.current_position == 0:
                self.current_position = 1
            elif signal == -1 and self.current_position == 0:
                self.current_position = -1
            
            # Exit logic
            elif self.current_position == 1:
                if self.exit_manager.should_exit_long(df, i):
                    self.current_position = 0
            
            elif self.current_position == -1:
                if self.exit_manager.should_exit_short(df, i):
                    self.current_position = 0
            
            positions.iloc[i] = self.current_position
        
        return positions
    
    def reset(self):
        """Reset position state"""
        self.current_position = 0


# ============================================================================
# 9. MAIN STRATEGY CLASS (Coordinator)
# ============================================================================
class ChanMomentum(BaseStrategy):
    """
    Chan Momentum Strategy - Coordinator
    
    Follows SOLID Principles:
    - Single Responsibility: Coordinates components
    - Open/Closed: Extend via components
    - Liskov Substitution: Replaces BaseStrategy
    - Interface Segregation: Clean component interfaces
    - Dependency Inversion: Depends on abstractions
    
    Strategy for STRONG TRENDING markets
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config or {})
        
        # Configuration
        lookback_short = self.config.get('lookback_short', 20)
        lookback_medium = self.config.get('lookback_medium', 60)
        lookback_long = self.config.get('lookback_long', 120)
        momentum_threshold = self.config.get('momentum_threshold', 0.05)
        rsi_period = self.config.get('rsi_period', 14)
        rsi_overbought = self.config.get('rsi_overbought', 70)
        rsi_oversold = self.config.get('rsi_oversold', 30)
        
        # Dependency Injection: Create components
        self.momentum_calculator = MomentumCalculator(
            lookback_short, lookback_medium, lookback_long
        )
        self.trend_calculator = TrendCalculator(
            lookback_short, lookback_medium, lookback_long
        )
        self.rsi_calculator = RSICalculator(rsi_period)
        self.score_calculator = MomentumScoreCalculator()
        self.rank_calculator = MomentumRankCalculator(lookback=60)
        self.signal_generator = MomentumSignalGenerator(
            momentum_threshold, rsi_overbought, rsi_oversold
        )
        self.exit_manager = MomentumExitManager(rsi_overbought, rsi_oversold)
        self.position_manager = MomentumPositionManager(self.exit_manager)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initialized Momentum({lookback_short}/{lookback_medium}/{lookback_long})")
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all indicators using components
        
        Template Method: Define skeleton, components fill details
        """
        df = df.copy()
        
        # 1. Momentum Returns
        momentum_data = self.momentum_calculator.calculate(df['close'])
        df = pd.concat([df, momentum_data], axis=1)
        
        # 2. Trend MAs
        trend_data = self.trend_calculator.calculate(df['close'])
        df = pd.concat([df, trend_data], axis=1)
        
        # 3. RSI
        df['rsi'] = self.rsi_calculator.calculate(df['close'])
        
        # 4. Momentum Score
        df['momentum_score'] = self.score_calculator.calculate(df)
        
        # 5. Momentum Rank
        df['momentum_rank'] = self.rank_calculator.calculate(df['momentum_score'])
        
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
        
        # Manage positions with exits
        self.position_manager.reset()
        df['position'] = self.position_manager.update_positions(df)
        
        # Calculate performance
        self.calculate_performance(df)
        
        self.logger.info(f"Generated {self.metrics['total_signals']} momentum signals")
        
        return df
    
    def get_current_signal(self, df: pd.DataFrame) -> Tuple[int, Dict]:
        """
        Get the most recent trading signal
        
        Returns:
            Tuple[signal, details]
        """
        df = self.generate_signals(df)
        
        latest = df.iloc[-1]
        signal = int(latest['signal'])
        
        signal_details = {
            'timestamp': df.index[-1],
            'close': float(latest['close']),
            'return_short': float(latest['return_short']),
            'return_medium': float(latest['return_medium']),
            'momentum_score': float(latest['momentum_score']),
            'momentum_rank': float(latest['momentum_rank']),
            'rsi': float(latest['rsi']),
            'position': int(latest['position']),
            'signal_type': 'BUY' if signal == 1 else 'SELL' if signal == -1 else 'HOLD'
        }
        
        return signal, signal_details


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Configuration
    config = {
        'lookback_short': 20,
        'lookback_medium': 60,
        'lookback_long': 120,
        'momentum_threshold': 0.05,
        'rsi_period': 14,
        'rsi_overbought': 70,
        'rsi_oversold': 30
    }
    
    # Create strategy
    strategy = ChanMomentum(config)
    
    # Generate sample data with strong trend
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=500, freq='1h')
    
    # Create strong uptrend
    trend = np.linspace(50000, 60000, 500)
    noise = np.random.randn(500) * 400
    prices = trend + noise
    
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
    print("CHAN MOMENTUM - SIGNAL GENERATION")
    print("="*60)
    print(f"Total Signals: {strategy.metrics['total_signals']}")
    print(f"Buy Signals:   {strategy.metrics['buy_signals']}")
    print(f"Sell Signals:  {strategy.metrics['sell_signals']}")
    print(f"Win Rate:      {strategy.metrics['win_rate']:.2%}")
    print(f"Sharpe Ratio:  {strategy.metrics['sharpe_ratio']:.2f}")
    print("="*60)
    
    # Show latest signals
    print("\nLatest 5 Signals:")
    cols = ['close', 'return_short', 'momentum_score', 'rsi', 'signal', 'position']
    print(result[cols].tail())
    
    # Get current signal
    signal, details = strategy.get_current_signal(df)
    print(f"\n🚦 Current Signal: {details['signal_type']}")
    print(f"   Price: ${details['close']:.2f}")
    print(f"   Momentum Score: {details['momentum_score']:.4f}")
    print(f"   Momentum Rank: {details['momentum_rank']:.1f}%")
    print(f"   RSI: {details['rsi']:.2f}")