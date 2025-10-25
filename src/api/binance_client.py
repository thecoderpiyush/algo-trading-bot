"""
Binance API Client Module
Handles connection to Binance Testnet and data fetching
"""
from dotenv import load_dotenv

import os
import sys
from binance.client import Client
from binance.exceptions import BinanceAPIException
import time
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

# Load environment variables
load_dotenv()

# Initialize logger
logger = get_logger('BinanceClient', 'DEBUG')


class BinanceClient:
    """Wrapper for Binance API with error handling and retry logic"""
    
    def __init__(self, testnet=True):
        """
        Initialize Binance client
        
        Args:
            testnet (bool): Use testnet if True, live trading if False
        """
        logger.info("Initializing Binance API Client...")
        
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_SECRET_KEY')
        
        if not self.api_key or not self.api_secret:
            logger.error("API keys not found in environment variables")
            raise ValueError("API keys not found. Please check your .env file")
        
        logger.debug(f"API Key loaded: {self.api_key[:8]}...")
        logger.debug(f"Secret Key loaded: {self.api_secret[:8]}...")
        
        try:
            # Initialize client
            self.client = Client(self.api_key, self.api_secret, testnet=testnet)
            self.testnet = testnet
            
            if testnet:
                # Set testnet URLs
                self.client.API_URL = 'https://testnet.binance.vision/api'
                logger.info("✅ Connected to Binance TESTNET")
                print("✅ Connected to Binance TESTNET")
            else:
                logger.warning("⚠️  Connected to Binance LIVE trading")
                print("⚠️  Connected to Binance LIVE trading")
                
        except Exception as e:
            logger.error(f"Failed to initialize Binance client: {e}", exc_info=True)
            raise
    
    def test_connection(self):
        """Test API connection and fetch account info"""
        logger.info("Testing connection to Binance API...")
        
        try:
            # Test connectivity
            status = self.client.get_system_status()
            logger.info(f"System Status: {status}")
            print(f"System Status: {status}")
            
            # Get account info
            account = self.client.get_account()
            logger.info(f"Account Type: {account['accountType']}")
            logger.info(f"Can Trade: {account['canTrade']}")
            logger.info(f"Can Withdraw: {account['canWithithdraw']}")
            
            print(f"\n📊 Account Status: {account['accountType']}")
            print(f"Can Trade: {account['canTrade']}")
            print(f"Can Withdraw: {account['canWithithdraw']}")
            
            # Show balances
            print("\n💰 Account Balances:")
            balances = [b for b in account['balances'] if float(b['free']) > 0 or float(b['locked']) > 0]
            
            if balances:
                for balance in balances:
                    logger.debug(f"Balance - {balance['asset']}: {balance['free']} (locked: {balance['locked']})")
                    print(f"  {balance['asset']}: {balance['free']} (locked: {balance['locked']})")
            else:
                logger.warning("No balances found in account")
                print("  No balances found")
            
            logger.info("✅ Connection test successful")
            return True
            
        except BinanceAPIException as e:
            logger.error(f"Binance API Error: {e}", exc_info=True)
            print(f"❌ API Error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during connection test: {e}", exc_info=True)
            print(f"❌ Error: {e}")
            return False
    
    def get_account_balance(self, asset='USDT'):
        """
        Get balance for a specific asset
        
        Args:
            asset (str): Asset symbol (e.g., 'USDT', 'BTC')
            
        Returns:
            dict: Balance information
        """
        logger.info(f"Fetching balance for {asset}...")
        
        try:
            balance = self.client.get_asset_balance(asset=asset)
            balance_info = {
                'asset': balance['asset'],
                'free': float(balance['free']),
                'locked': float(balance['locked']),
                'total': float(balance['free']) + float(balance['locked'])
            }
            logger.info(f"{asset} Balance: {balance_info['total']} (free: {balance_info['free']}, locked: {balance_info['locked']})")
            return balance_info
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error fetching balance for {asset}: {e}")
            print(f"❌ Error fetching balance: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching balance for {asset}: {e}", exc_info=True)
            print(f"❌ Error fetching balance: {e}")
            return None
    
    def get_historical_klines(self, symbol='BTCUSDT', interval='1h', lookback='100'):
        """
        Fetch historical candlestick data
        
        Args:
            symbol (str): Trading pair (e.g., 'BTCUSDT')
            interval (str): Candlestick interval ('1m', '5m', '15m', '1h', '4h', '1d')
            lookback (str): Number of candles to fetch
            
        Returns:
            pd.DataFrame: Historical price data
        """
        logger.info(f"Fetching {lookback} {interval} candles for {symbol}...")
        
        try:
            # Fetch klines
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=int(lookback)
            )
            
            logger.debug(f"Received {len(klines)} klines from API")
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Select and convert essential columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            logger.info(f"✅ Successfully fetched {len(df)} candles for {symbol} ({interval})")
            logger.debug(f"Price range: Low={df['low'].min():.2f}, High={df['high'].max():.2f}")
            print(f"✅ Fetched {len(df)} candles for {symbol} ({interval})")
            
            return df
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error fetching klines for {symbol}: {e}")
            print(f"❌ Binance API Error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}", exc_info=True)
            print(f"❌ Error fetching historical data: {e}")
            return None
    
    def get_current_price(self, symbol='BTCUSDT'):
        """
        Get current market price for a symbol
        
        Args:
            symbol (str): Trading pair
            
        Returns:
            float: Current price
        """
        logger.info(f"Fetching current price for {symbol}...")
        
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            price = float(ticker['price'])
            logger.info(f"{symbol} Current Price: ${price:,.2f}")
            print(f"💵 {symbol} Current Price: ${price:,.2f}")
            return price
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error fetching price for {symbol}: {e}")
            print(f"❌ Error fetching price: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}", exc_info=True)
            print(f"❌ Error fetching price: {e}")
            return None
    
    def get_order_book(self, symbol='BTCUSDT', limit=10):
        """
        Get order book depth
        
        Args:
            symbol (str): Trading pair
            limit (int): Number of orders to fetch
            
        Returns:
            dict: Order book data
        """
        logger.info(f"Fetching order book for {symbol} (limit: {limit})...")
        
        try:
            depth = self.client.get_order_book(symbol=symbol, limit=limit)
            order_book = {
                'bids': depth['bids'][:limit],
                'asks': depth['asks'][:limit]
            }
            logger.info(f"✅ Fetched order book: {len(order_book['bids'])} bids, {len(order_book['asks'])} asks")
            return order_book
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error fetching order book for {symbol}: {e}")
            print(f"❌ Error fetching order book: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}", exc_info=True)
            print(f"❌ Error fetching order book: {e}")
            return None
    
    def retry_on_failure(self, func, max_retries=3, delay=2, *args, **kwargs):
        """
        Retry a function on failure with exponential backoff
        
        Args:
            func: Function to execute
            max_retries (int): Maximum retry attempts
            delay (int): Initial delay in seconds
            
        Returns:
            Result of function or None
        """
        logger.debug(f"Executing {func.__name__} with retry logic (max_retries={max_retries})")
        
        for attempt in range(max_retries):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    logger.info(f"✅ {func.__name__} succeeded on attempt {attempt + 1}")
                return result
                
            except BinanceAPIException as e:
                if e.code == -1021:  # Timestamp error
                    logger.warning("Timestamp synchronization error, retrying...")
                    print("⚠️  Timestamp error, syncing...")
                    time.sleep(1)
                elif attempt < max_retries - 1:
                    wait_time = delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {wait_time}s...")
                    print(f"⚠️  Attempt {attempt + 1} failed. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {max_retries} retry attempts failed for {func.__name__}: {e}")
                    print(f"❌ All retry attempts failed: {e}")
                    return None
                    
            except Exception as e:
                logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
                print(f"❌ Unexpected error: {e}")
                return None


# Test the client
if __name__ == "__main__":
    logger.info("="*60)
    logger.info("Starting Binance API Client Test")
    logger.info("="*60)
    
    print("🚀 Testing Binance API Client\n")
    
    try:
        # Initialize client
        logger.info("Step 1: Initializing client...")
        client = BinanceClient(testnet=True)
        
        # Test connection
        print("\n1️⃣ Testing Connection...")
        logger.info("Step 2: Testing connection...")
        client.test_connection()
        
        # Get current price
        print("\n2️⃣ Fetching Current Price...")
        logger.info("Step 3: Fetching current price...")
        client.get_current_price('BTCUSDT')
        
        # Get historical data
        print("\n3️⃣ Fetching Historical Data...")
        logger.info("Step 4: Fetching historical data...")
        df = client.get_historical_klines('BTCUSDT', '1h', '50')
        
        if df is not None:
            print(f"\n📈 Latest candles:")
            print(df.tail())
            logger.info(f"Sample data:\n{df.tail().to_string()}")
        
        print("\n✅ All tests completed!")
        logger.info("="*60)
        logger.info("All tests completed successfully")
        logger.info("="*60)
        
    except Exception as e:
        logger.critical(f"Fatal error during testing: {e}", exc_info=True)
        print(f"\n❌ Fatal Error: {e}")
        print("Check logs/trading_bot_*.log for details")