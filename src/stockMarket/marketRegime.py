import datetime
import time
import numpy as np
import pandas as pd
import upstox_client
from upstox_client.rest import ApiException

# ========== CONFIG ==========
API_KEY = "da9d7a7b-cd8d-4c2c-84a6-1bb69407148e"
ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiIzVUNWS0wiLCJqdGkiOiI2OGZiNTA3MTExYTYxNDNjNzZjZGYwOGIiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6ZmFsc2UsImlhdCI6MTc2MTMwMDU5MywiaXNzIjoidWRhcGktZ2F0ZXdheS1zZXJ2aWNlIiwiZXhwIjoxNzYxMzQzMjAwfQ.4p_QbEDMSRlX-GCNEhw1T40w8Tj6Gos98e5IV10EcZk"
# Example: instrument_key for Nifty-50 (you will need the correct token for NSE index)
INSTRUMENT_KEY = "NSE_INDEX|IND_NIFTY50"
INTERVAL = "5minute"   # or "day", "15minute" etc
PERIOD = 50            # moving average period
# ============================

def init_api():
    configuration = upstox_client.Configuration()
    configuration.access_token = ACCESS_TOKEN
    api_client = upstox_client.ApiClient(configuration)
    history_api = upstox_client.HistoryApi(api_client)
    return history_api

def fetch_historical(history_api, instrument_key, interval, from_date, to_date):
    """
    Fetch historical OHLC data using Upstox API (v2).
    Returns a pandas DataFrame with time, open, high, low, close.
    """
    try:
        resp = history_api.get_historical_candle_data(
            api_version='2.0',  # REQUIRED argument
            instrument_key=instrument_key,
            interval=interval,
            from_date=from_date.strftime("%Y-%m-%d"),
            to_date=to_date.strftime("%Y-%m-%d")
        )
    except ApiException as e:
        print("Exception when calling get_historical_candle_data: %s\n" % e)
        return None

    if not hasattr(resp, "data") or resp.data is None or not hasattr(resp.data, "candles"):
        print("No candle data returned.")
        return None

    candles = resp.data.candles
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df

    # Convert response to DataFrame
    # In the latest SDK, resp.data.candles is a list of lists:
    # [ [timestamp, open, high, low, close, volume], ... ]
    if not hasattr(resp, "data") or resp.data is None or not hasattr(resp.data, "candles"):
        print("No candle data returned.")
        return None

    candles = resp.data.candles
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df


def compute_regime(df, period=PERIOD):
    """
    Compute the market regime based on recent data.
    Returns a string: 'Bullish', 'Bearish', or 'Sideways'
    """
    df["ma"] = df["close"].rolling(window=period).mean()
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    price = latest["close"]
    ma = latest["ma"]
    prev_price = prev["close"]
    prev_ma = df["ma"].iloc[-2]

    # Basic logic:
    if (price > ma) and (prev_price < prev_ma or price > prev_price):
        return "Bullish"
    elif (price < ma) and (prev_price > prev_ma or price < prev_price):
        return "Bearish"
    else:
        return "Sideways"

def main():
    history_api = init_api()
    to_date = datetime.datetime.now()
    from_date = to_date - datetime.timedelta(days=7)  # for example last 7 days

    df = fetch_historical(history_api, INSTRUMENT_KEY, INTERVAL, from_date, to_date)
    if df is None or df.empty:
        print("No data returned.")
        return

    regime = compute_regime(df)
    print(f"Current market regime: {regime}")

    # Optionally print last few rows
    print(df.tail())

if __name__ == "__main__":
    main()
