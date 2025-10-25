"""
Comprehensive Test Suite for Strategy Validator
Tests all validators with various scenarios

File Location: tests/test_strategy_validator.py

Author: Piyush | Date: Oct 24, 2025
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

from src.strategy.strategy_validator import (
    StrategyValidator,
    ValidationChain,
    SignalValueValidator,
    IndicatorQualityValidator,
    PositionConsistencyValidator,
    DataSufficiencyValidator,
    SignalDistributionValidator,
    LookaheadBiasValidator,
    validate_strategy_output
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_valid_dataframe(n_points=300):
    """Create a valid sample DataFrame"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=n_points, freq='1h')
    
    # Generate realistic price data
    base_price = 50000
    returns = np.random.randn(n_points) * 0.02  # 2% volatility
    close_prices = base_price * (1 + returns).cumprod()
    
    df = pd.DataFrame({
        'open': close_prices * (1 + np.random.randn(n_points) * 0.001),
        'high': close_prices * (1 + np.abs(np.random.randn(n_points)) * 0.002),
        'low': close_prices * (1 - np.abs(np.random.randn(n_points)) * 0.002),
        'close': close_prices,
        'volume': np.random.randint(100, 1000, n_points),
    }, index=dates)
    
    # Add indicators with proper warm-up
    df['fast_ma'] = df['close'].rolling(20, min_periods=20).mean()
    df['slow_ma'] = df['close'].rolling(50, min_periods=50).mean()
    
    # Add RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Generate signals (sparse - 5% of time)
    df['signal'] = 0
    signal_mask = np.random.choice([True, False], n_points, p=[0.05, 0.95])
    df.loc[signal_mask, 'signal'] = np.random.choice([-1, 1], signal_mask.sum())
    
    # Generate position (tracks signals)
    df['position'] = df['signal'].replace(0, np.nan).ffill().fillna(0).astype(int)
    
    return df


def print_test_header(test_name):
    """Print formatted test header"""
    print("\n" + "=" * 70)
    print(f"🧪 {test_name}")
    print("=" * 70)


def print_test_result(passed, message=""):
    """Print test result"""
    if passed:
        print(f"✅ PASSED {message}")
    else:
        print(f"❌ FAILED {message}")
    print("-" * 70)


# ============================================================================
# TEST CASES
# ============================================================================

def test_1_valid_dataframe():
    """Test 1: Validate a properly formatted DataFrame"""
    print_test_header("Test 1: Valid DataFrame")
    
    df = create_valid_dataframe()
    
    print("\nDataFrame Info:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Signal distribution: {df['signal'].value_counts().to_dict()}")
    
    result = validate_strategy_output(df, verbose=True)
    print_test_result(result, "- Valid DataFrame should pass")
    
    return result


def test_2_invalid_signal_values():
    """Test 2: Invalid signal values"""
    print_test_header("Test 2: Invalid Signal Values")
    
    df = create_valid_dataframe()
    
    # Inject invalid signals
    df.loc[df.index[10], 'signal'] = 5
    df.loc[df.index[20], 'signal'] = -2
    
    print("\nInjected invalid signals: 5, -2")
    
    validator = StrategyValidator()
    result = validator.validate(df, verbose=True)
    
    print_test_result(not result, "- Should FAIL with invalid signals")
    
    return not result  # Test passes if validation fails


def test_3_missing_signal_column():
    """Test 3: Missing signal column"""
    print_test_header("Test 3: Missing Signal Column")
    
    df = create_valid_dataframe()
    df_no_signal = df.drop('signal', axis=1)
    
    print("\nRemoved 'signal' column")
    
    validator = StrategyValidator()
    result = validator.validate(df_no_signal, verbose=True)
    
    print_test_result(not result, "- Should FAIL with missing column")
    
    return not result


def test_4_insufficient_data():
    """Test 4: Insufficient data points"""
    print_test_header("Test 4: Insufficient Data")
    
    df = create_valid_dataframe(n_points=100)  # Less than minimum 250
    
    print(f"\nDataFrame has only {len(df)} points (minimum: 250)")
    
    validator = StrategyValidator()
    result = validator.validate(df, verbose=True)
    
    print_test_result(not result, "- Should FAIL with insufficient data")
    
    return not result


def test_5_no_signals_generated():
    """Test 5: No signals generated"""
    print_test_header("Test 5: No Signals Generated")
    
    df = create_valid_dataframe()
    df['signal'] = 0  # All signals are 0
    
    print("\nAll signals set to 0 (no BUY/SELL)")
    
    validator = StrategyValidator()
    result = validator.validate(df, verbose=True)
    
    print_test_result(not result, "- Should FAIL with no signals")
    
    return not result


def test_6_excessive_signals():
    """Test 6: Over-trading detection"""
    print_test_header("Test 6: Excessive Signals (Over-trading)")
    
    df = create_valid_dataframe()
    
    # Generate signals on 20% of candles (default max is 10%)
    signal_mask = np.random.choice([True, False], len(df), p=[0.2, 0.8])
    df.loc[signal_mask, 'signal'] = np.random.choice([-1, 1], signal_mask.sum())
    
    signal_rate = (df['signal'] != 0).sum() / len(df)
    print(f"\nSignal rate: {signal_rate:.2%} (max allowed: 10%)")
    
    validator = StrategyValidator()
    result = validator.validate(df, verbose=True)
    
    print_test_result(not result, "- Should FAIL with excessive signals")
    
    return not result


def test_7_nan_in_indicators():
    """Test 7: NaN values in indicators"""
    print_test_header("Test 7: NaN Values in Indicators")
    
    df = create_valid_dataframe()
    
    # Inject NaN in recent data
    df.loc[df.index[-10:], 'fast_ma'] = np.nan
    
    print("\nInjected NaN in 'fast_ma' (last 10 rows)")
    
    validator = StrategyValidator()
    validator.add_indicator_validator(['fast_ma', 'slow_ma', 'rsi'])
    result = validator.validate(df, verbose=True)
    
    print_test_result(not result, "- Should FAIL with NaN in indicators")
    
    return not result


def test_8_infinite_values():
    """Test 8: Infinite values in indicators"""
    print_test_header("Test 8: Infinite Values in Indicators")
    
    df = create_valid_dataframe()
    
    # Inject infinite values
    df.loc[df.index[-5], 'rsi'] = np.inf
    df.loc[df.index[-3], 'rsi'] = -np.inf
    
    print("\nInjected infinite values in 'rsi'")
    
    validator = StrategyValidator()
    validator.add_indicator_validator(['fast_ma', 'slow_ma', 'rsi'])
    result = validator.validate(df, verbose=True)
    
    print_test_result(not result, "- Should FAIL with infinite values")
    
    return not result


def test_9_position_consistency():
    """Test 9: Invalid position transitions"""
    print_test_header("Test 9: Position Consistency")
    
    df = create_valid_dataframe()
    
    # Create invalid position jump (-1 to 1 without 0)
    df['position'] = 0
    df.loc[df.index[100], 'position'] = -1
    df.loc[df.index[101], 'position'] = 1  # Invalid jump
    
    print("\nCreated invalid position jump: -1 → 1 (should go through 0)")
    
    validator = StrategyValidator()
    result = validator.validate(df, verbose=True)
    
    print_test_result(not result, "- Should FAIL with invalid position jump")
    
    return not result


def test_10_custom_validator_chain():
    """Test 10: Custom validation chain"""
    print_test_header("Test 10: Custom Validation Chain")
    
    df = create_valid_dataframe()
    
    print("\nBuilding custom validation chain...")
    
    # Build custom chain
    chain = ValidationChain()
    chain.add_validator(SignalValueValidator())
    chain.add_validator(DataSufficiencyValidator(min_data_points=200))
    chain.add_validator(SignalDistributionValidator(max_signal_rate=0.15))
    
    print(f"Added {len(chain.validators)} validators")
    
    results = chain.validate_all(df)
    all_passed = all(r.is_valid for r in results.values())
    
    print("\nValidation Results:")
    for name, result in results.items():
        status = "✅" if result.is_valid else "❌"
        print(f"  {status} {name}: {result.message}")
    
    print_test_result(all_passed, "- Custom chain should pass")
    
    return all_passed


def test_11_with_real_strategy_data():
    """Test 11: Integration with actual strategy output"""
    print_test_header("Test 11: Real Strategy Integration")
    
    # Simulate real strategy output
    df = create_valid_dataframe(n_points=500)
    
    # Calculate MA crossover signals
    df['signal'] = 0
    df['position'] = 0
    
    # Generate crossover signals
    df['ma_cross'] = df['fast_ma'] - df['slow_ma']
    
    for i in range(100, len(df)):
        if df['ma_cross'].iloc[i] > 0 and df['ma_cross'].iloc[i-1] <= 0:
            df.loc[df.index[i], 'signal'] = 1  # Golden cross
        elif df['ma_cross'].iloc[i] < 0 and df['ma_cross'].iloc[i-1] >= 0:
            df.loc[df.index[i], 'signal'] = -1  # Death cross
    
    # Track position
    df['position'] = df['signal'].replace(0, np.nan).ffill().fillna(0).astype(int)
    
    print(f"\nGenerated {(df['signal'] != 0).sum()} signals from MA crossover")
    print(f"Signal distribution: {df['signal'].value_counts().to_dict()}")
    
    # Validate with indicator checking
    validator = StrategyValidator()
    validator.add_indicator_validator(['fast_ma', 'slow_ma', 'rsi'])
    result = validator.validate(df, verbose=True)
    
    print_test_result(result, "- Real strategy should pass")
    
    return result


def test_12_failed_validators_report():
    """Test 12: Failed validators report"""
    print_test_header("Test 12: Failed Validators Report")
    
    df = create_valid_dataframe(n_points=100)  # Will fail sufficiency
    df.loc[df.index[0], 'signal'] = 999  # Will fail signal value
    
    validator = StrategyValidator()
    validator.validate(df, verbose=False)
    
    failed = validator.get_failed_validators()
    
    print(f"\nFailed validators: {failed}")
    print("\nGenerated report:")
    print(validator.generate_report())
    
    print_test_result(len(failed) > 0, "- Should detect failed validators")
    
    return len(failed) > 0


# ============================================================================
# RUN ALL TESTS
# ============================================================================

def run_all_tests():
    """Run all test cases"""
    print("\n" + "=" * 70)
    print("🚀 STRATEGY VALIDATOR TEST SUITE")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Valid DataFrame", test_1_valid_dataframe),
        ("Invalid Signal Values", test_2_invalid_signal_values),
        ("Missing Signal Column", test_3_missing_signal_column),
        ("Insufficient Data", test_4_insufficient_data),
        ("No Signals Generated", test_5_no_signals_generated),
        ("Excessive Signals", test_6_excessive_signals),
        ("NaN in Indicators", test_7_nan_in_indicators),
        ("Infinite Values", test_8_infinite_values),
        ("Position Consistency", test_9_position_consistency),
        ("Custom Validator Chain", test_10_custom_validator_chain),
        ("Real Strategy Integration", test_11_with_real_strategy_data),
        ("Failed Validators Report", test_12_failed_validators_report),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"\n❌ Test crashed: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("\n" + "=" * 70)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("=" * 70)
    
    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    results = run_all_tests()
    
    # Exit with proper code
    all_passed = all(results.values())
    exit(0 if all_passed else 1)