"""
Test script for the enhanced Technical Indicators module
Tests integration with Binance API and all indicators
"""

import sys
import os
from datetime import datetime

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from api.binance_client import BinanceClient
from utils.indicators import TechnicalIndicators


def test_with_live_data():
    """Test indicators with live Binance data"""
    print("=" * 80)
    print("🧪 TESTING ENHANCED INDICATORS WITH LIVE DATA")
    print("=" * 80 + "\n")
    
    try:
        # Initialize Binance client
        print("📡 Connecting to Binance Testnet...")
        binance_client = BinanceClient()
        
        # Initialize Technical Indicators with Binance client
        ti = TechnicalIndicators(client=binance_client.client, log_level='INFO')
        
        # Fetch K-line data
        print("\n📊 Fetching historical data...")
        df = ti.fetch_klines(
            symbol='BTCUSDT',
            interval='1h',  # Use binance.client.Client.KLINE_INTERVAL_1HOUR in production
            start_str='30 days ago UTC'
        )
        
        print(f"\n✓ Loaded {len(df)} candles")
        print(f"✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Calculate all indicators
        print("\n" + "=" * 80)
        print("📈 CALCULATING ALL INDICATORS")
        print("=" * 80)
        
        df_with_indicators = ti.calculate_all_indicators(
            sma_periods=[20, 50, 200],
            ema_periods=[12, 26, 50],
            rsi_period=14,
            macd_params=(12, 26, 9),
            bb_params=(20, 2),
            include_advanced=True,
            include_volume=True,
            include_volatility=True
        )
        
        # Display indicator summary
        print("\n" + "=" * 80)
        print("📊 INDICATOR SUMMARY")
        print("=" * 80)
        
        summary = ti.get_indicator_summary()
        print(f"\n✓ Total indicators calculated: {len(summary)}")
        print(f"✓ DataFrame shape: {df_with_indicators.shape}")
        print(f"✓ Total columns: {len(df_with_indicators.columns)}")
        
        # Show top indicators
        print("\n📋 Sample Indicators (with latest values):")
        important_indicators = [
            'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 
            'RSI_14', 'MACD', 'BB_upper', 'BB_lower', 'ADX_14'
        ]
        
        for ind in important_indicators:
            if ind in summary['Indicator'].values:
                row = summary[summary['Indicator'] == ind].iloc[0]
                print(f"   {ind:15s}: {row['Latest']:>12.2f} (Coverage: {row['Coverage_%']:.1f}%)")
        
        # Get trading signals
        print("\n" + "=" * 80)
        print("🚦 LATEST TRADING SIGNALS")
        print("=" * 80)
        
        signals = ti.get_latest_signals()
        print(f"\n💵 Current Price: ${signals['price']:,.2f}")
        print(f"⏰ Timestamp: {signals['timestamp']}")
        
        if 'sma_trend' in signals:
            emoji = "📈" if signals['sma_trend'] == 'BULLISH' else "📉"
            print(f"\n{emoji} SMA Trend: {signals['sma_trend']}")
        
        if 'ema_trend' in signals:
            emoji = "📈" if signals['ema_trend'] == 'BULLISH' else "📉"
            print(f"{emoji} EMA Trend: {signals['ema_trend']}")
        
        if 'rsi' in signals:
            print(f"\n📊 RSI: {signals['rsi']:.2f} - {signals['rsi_signal']}")
        
        if 'macd_signal' in signals:
            emoji = "📈" if signals['macd_signal'] == 'BULLISH' else "📉"
            print(f"{emoji} MACD: {signals['macd_signal']} (Histogram: {signals['macd_histogram']:.4f})")
        
        if 'trend_strength' in signals:
            print(f"\n💪 Trend Strength (ADX): {signals['trend_strength']} ({signals['adx']:.2f})")
        
        if 'bb_signal' in signals:
            print(f"🎯 Bollinger Bands: {signals['bb_signal']} (%B: {signals['bb_percent_b']:.2f})")
        
        # Display last 5 rows with key indicators
        print("\n" + "=" * 80)
        print("📈 LATEST DATA (Last 5 candles)")
        print("=" * 80)
        
        display_cols = ['close', 'SMA_20', 'SMA_50', 'RSI_14', 'MACD', 'BB_upper', 'BB_lower']
        available_cols = [col for col in display_cols if col in df_with_indicators.columns]
        
        print(df_with_indicators[available_cols].tail())
        
        # Export to CSV
        print("\n" + "=" * 80)
        print("💾 EXPORTING DATA")
        print("=" * 80)
        
        filename = f"data/indicators_BTCUSDT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        os.makedirs('data', exist_ok=True)
        ti.export_to_csv(filename)
        print(f"✓ Data exported to: {filename}")
        
        # Overall assessment
        print("\n" + "=" * 80)
        print("🎯 TRADING ASSESSMENT")
        print("=" * 80)
        
        bullish_signals = 0
        bearish_signals = 0
        
        if signals.get('sma_trend') == 'BULLISH':
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        if signals.get('ema_trend') == 'BULLISH':
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        if signals.get('rsi_signal') == 'OVERSOLD':
            bullish_signals += 1
        elif signals.get('rsi_signal') == 'OVERBOUGHT':
            bearish_signals += 1
        
        if signals.get('macd_signal') == 'BULLISH':
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        print(f"\n📊 Signal Summary:")
        print(f"   Bullish signals: {bullish_signals}")
        print(f"   Bearish signals: {bearish_signals}")
        
        if bullish_signals > bearish_signals:
            print(f"\n✅ Overall Sentiment: BULLISH 📈")
        elif bearish_signals > bullish_signals:
            print(f"\n⚠️  Overall Sentiment: BEARISH 📉")
        else:
            print(f"\n➡️  Overall Sentiment: NEUTRAL ↔️")
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED SUCCESSFULLY!")
        print("=" * 80)
        
        return df_with_indicators, ti
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Run the test
    df, ti_instance = test_with_live_data()
    
    if df is not None:
        print("\n💡 TIP: The indicator data is now available in the 'df' variable")
        print("💡 You can use it for backtesting and strategy development!")