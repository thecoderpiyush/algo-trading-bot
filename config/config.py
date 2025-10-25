"""
Configuration settings for the trading bot
"""

import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')

# Trading Configuration
TRADING_SYMBOL = os.getenv('TRADING_SYMBOL', 'BTCUSDT')
TRADING_INTERVAL = os.getenv('TRADING_INTERVAL', '1h')
RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', '0.02'))  # 2% risk per trade

# Strategy Parameters
SMA_SHORT_PERIOD = 20  # Short-term moving average
SMA_LONG_PERIOD = 50   # Long-term moving average
EMA_SHORT_PERIOD = 12
EMA_LONG_PERIOD = 26
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Risk Management
MAX_POSITION_SIZE = 0.95  # Use max 95% of available balance
STOP_LOSS_PERCENTAGE = 0.02  # 2% stop loss
TAKE_PROFIT_PERCENTAGE = 0.04  # 4% take profit

# Backtesting
BACKTEST_START_BALANCE = 10000  # Starting balance for backtesting
TRADING_FEE = 0.001  # 0.1% trading fee

# Logging
LOG_LEVEL = 'INFO'
LOG_FILE = 'logs/trading_bot.log'

# Telegram (Optional)
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')