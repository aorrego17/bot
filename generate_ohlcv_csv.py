import os
import pandas as pd

def select_option(prompt, options, default=None):
    print(f"{prompt}")
    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")
    while True:
        inp = input(f"Selecciona opción [1-{len(options)}] (default: {default or 1}): ").strip()
        if not inp and default is not None:
            return options[default-1]
        if inp.isdigit() and 1 <= int(inp) <= len(options):
            return options[int(inp)-1]
        print("Opción inválida, intenta de nuevo.")

def ask_param(prompt, default=None, example=None):
    ex = f" (ej: {example})" if example else ""
    inp = input(f"{prompt}{ex}{' [' + str(default) + ']' if default else ''}: ").strip()
    return inp or default

def fetch_binance_ohlcv(symbol, interval, lookback):
    from binance.client import Client
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    client = Client(api_key, api_secret)
    klines = client.get_historical_klines(symbol, interval, lookback)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    num_cols = ['open', 'high', 'low', 'close', 'volume']
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
    df.set_index('timestamp', inplace=True)
    return df[num_cols]

def fetch_ccxt_ohlcv(exchange_name, symbol, interval, since, limit):
    import ccxt
    import time
    # Map intervals to minutes
    int_map = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
    exchange = getattr(ccxt, exchange_name)()
    df_total = pd.DataFrame()
    tframe = int_map[interval]
    ms_start = since  # epoch ms
    for i in range(limit):
        candles = exchange.fetch_ohlcv(symbol, interval, since=ms_start)
        if not candles:
            break
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df_total = pd.concat([df_total, df])
        ms_start = int(df.index[-1].timestamp() * 1000) + tframe * 60 * 1000  # trick: next bar
        if len(df) < 500:
            break
        time.sleep(exchange.rateLimit / 1000)  # respetar limit API
    return df_total

def fetch_yfinance_ohlcv(symbol, interval, start, end):
    import yfinance as yf
    df = yf.download(symbol, interval=interval, start=start, end=end)
    df.index.name = "timestamp"
    return df[["Open", "High", "Low", "Close", "Volume"]].rename(
        columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    )

def main():
    print("=== Generador CSV OHLCV multiplataforma ===")
    print("Plataformas disponibles: binance / ccxt / yfinance / csv")
    platform = select_option("¿Plataforma?", ["binance", "ccxt", "yfinance", "csv"], default=1)

    if platform == "binance":
        symbol = ask_param("Símbolo", "BTCUSDT", example="BTCUSDT, ETHUSDT, XRPUSDT")
        interval = select_option("Intervalo", ["1m", "5m", "15m", "1h", "4h", "1d"], default=4)
        lookback = ask_param("Lookback", "1000 hours ago UTC", example="500 days ago UTC, 1000 hours ago UTC")
        df = fetch_binance_ohlcv(symbol, interval, lookback)

    elif platform == "ccxt":
        exchanges = ["binance", "bybit", "kraken", "kucoin", "bitfinex"]
        exchange_name = select_option("Exchange", exchanges, default=1)
        symbol = ask_param("Símbolo", "BTC/USDT", example="BTC/USDT, ETH/USDT")
        interval = select_option("Intervalo", ["1m", "5m", "15m", "1h", "4h", "1d"], default=4)
        from datetime import datetime, timedelta
        start_str = ask_param("Fecha inicio", "2022-01-01", example="YYYY-MM-DD")
        since = int(pd.Timestamp(start_str).timestamp() * 1000)
        limit = int(ask_param("Cantidad de ciclos (500 barras cada uno)", "2"))
        df = fetch_ccxt_ohlcv(exchange_name, symbol, interval, since, limit)

    elif platform == "yfinance":
        symbol = ask_param("Ticker", "AAPL", example="AAPL, MSFT, TSLA, SPY")
        interval = select_option("Intervalo", ["1d", "1h", "5m"], default=2)
        start = ask_param("Fecha inicio", "2022-01-01", example="YYYY-MM-DD")
        end = ask_param("Fecha fin", "2023-01-01", example="YYYY-MM-DD")
        df = fetch_yfinance_ohlcv(symbol, interval, start, end)

    elif platform == "csv":
        src = ask_param("Ruta del CSV existente", example="data/BTCUSDT_1h.csv")
        df = pd.read_csv(src, index_col=0, parse_dates=True)
    else:
        print("Plataforma no soportada.")
        return

    filename = ask_param("Nombre archivo CSV destino", f"data/{symbol.replace('/','_')}_{interval}.csv")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    required_cols = ["open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Faltan columnas en los datos: {missing_cols}. No se guardó el archivo.")
        return

    df.to_csv(filename)
    print(f"¡Guardado: {filename} ({len(df)} filas)!")

if __name__ == "__main__":
    main()

