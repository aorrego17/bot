import os
import math
import pandas as pd
import numpy as np
import joblib
import requests
import json
import time
import threading
from datetime import datetime
from binance.client import Client
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from sklearn.ensemble import RandomForestClassifier
from decimal import Decimal
from typing import Optional
from sklearn.metrics import classification_report
from ta.volatility import BollingerBands, AverageTrueRange
from dotenv import load_dotenv

load_dotenv() # Esto carga las variables del .env en entorno

# Diccionario global para locks por símbolo
symbol_locks = {}
def get_symbol_lock(symbol):
    """Devuelve un lock para ese símbolo, creándolo si hace falta (thread-safe)."""
    global symbol_locks
    if symbol not in symbol_locks:
        symbol_locks[symbol] = threading.Lock()
    return symbol_locks[symbol]

def load_config(config_path="config.json"):
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        error_msg = f"❌ Error al cargar la configuración: {e}"
        print(error_msg)
        log_event(error_msg)
        send_telegram("Error crítico: No se pudo cargar config.json, bot detenido.", alert_type="ERROR")
        raise  # Para detener el bot en caso de error crítico de configuración

def retry_on_exception(max_attempts=3, delay=2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for i in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i < max_attempts - 1:
                        time.sleep(delay)
                    else:
                        raise e
        return wrapper
    return decorator

def get_stop_loss_file(symbol):
    # Retorna el path del archivo de stop loss específico de un símbolo
    stop_loss_dir = os.path.join(BASE_DIR, "stop_loss")
    if not os.path.exists(stop_loss_dir):
        os.makedirs(stop_loss_dir)
    return os.path.join(stop_loss_dir, f"stop_loss_{symbol}.json")

# === Ruta del directorio actual del script trading_bot_final.py ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "log.txt")
MODEL_PATH = os.path.join(BASE_DIR, "modelo_trading.pkl")
STATUS_FILE = os.path.join(BASE_DIR, "bot_status.json")
LAST_SIGNAL_FILE = os.path.join(BASE_DIR, "last_signal.json")
TRADES_LOG_FILE = os.path.join(BASE_DIR, "trades_log.json")

# === Cargar variables de entorno ===
USE_TESTNET = True  # Cambia a False para cuenta real

# === Tokens de API ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    print("⚠️ Advertencia: Telegram no esta completamente configurado.")

# === Claves de Binance ===
if USE_TESTNET:
    api_key = os.getenv("BINANCE_TESTNET_API_KEY")
    api_secret = os.getenv("BINANCE_TESTNET_API_SECRET")
else:
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

if not api_key or not api_secret:
    raise ValueError("❌ API KEY y SECRET no están definidos en variables de entorno.")

client = Client(api_key, api_secret)
if USE_TESTNET:
    client.API_URL = 'https://testnet.binance.vision/api'

# === Configuración de logs ===
def log_event(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    separator = "\n" + "=" * 60 + f"\n🕒 {timestamp}\n" + "=" * 60 + "\n"
    with open(LOG_FILE, "a") as f:
        f.write(separator)
        f.write(message.strip() + "\n")

# === Limpiar el archivo log cuando supere las 5mb
def cleanup_logs(max_size_mb=5):
    """
    Borra el archivo de log si supera cierto tamaño.
    Por defecto: 5 MB.
    """
    if os.path.exists(LOG_FILE):
        size_mb = os.path.getsize(LOG_FILE) / (1024 * 1024)
        if size_mb > max_size_mb:
            with open(LOG_FILE, "w") as f:
                f.write(f"🧹 Log limpiado automáticamente. Tamaño excedía {max_size_mb}MB.\n")
            log_event(f"🧼 Log reiniciado automáticamente. Tamaño anterior: {size_mb:.2f} MB.")

# === Notificación por Telegram ===
def send_telegram(message, alert_type="INFO"):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return "⚠️ Telegram no configurado. No se envió el mensaje."
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

    prefix = {
        "INFO": "ℹ️",
        "WARNING": "⚠️",
        "ERROR": "🚨"
    }.get(alert_type, "")

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": f"{prefix} {message}" # Incluye un prefijo en el mensaje si es necesario
    }

    try:
        response = requests.post(url, data=payload)
        if response.ok:
            return "📨 Mensaje enviado a Telegram correctamente."
        else:
            return f"❌ Error enviando mensaje a Telegram: {response.text}"
    except Exception as e:
        return f"❌ Excepción al enviar mensaje a Telegram: {e}"

def validate_strategy_types(strategy_names, config):
    tipos = set()
    for s in strategy_names:
        tipo = config.get("strategies_meta", {}).get(s, {}).get("type")
        if not tipo:
            raise Exception(f"Estrategia {s} no tiene tipo definido en config.")
        tipos.add(tipo)
    if len(tipos) != 1:
        raise Exception(f"Error: estrategias de tipos distintos en consenso: {tipos}. No se permite operar consenso.")
    return list(tipos)[0]

# === Datos históricos ===
@retry_on_exception(max_attempts=3)
def get_price_data(symbol, interval="1h", lookback="30 days ago UTC"):
    klines = client.get_historical_klines(symbol, interval, lookback)
    if not klines:
        raise ValueError(f"❌ No se obtuvieron datos de Binance para {symbol} ({interval})")
    
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
    ])

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Convertir todas las columnas apropiadas a numérico
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'taker_buy_base_volume', 'taker_buy_quote_volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Manejo de NaN después de las conversiones
    df.dropna(subset=['close', 'high', 'low'], inplace=True)

    df.set_index('timestamp', inplace=True)
    return df[['close', 'high', 'low']].dropna()

def add_indicators(df):
    # RSI y EMA
    df['RSI'] = RSIIndicator(close=df['close'], window=14).rsi()
    df['EMA'] = EMAIndicator(close=df['close'], window=20).ema_indicator()

    # MACD
    macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd() # Línea MACD
    df['MACD_signal'] = macd.macd_signal() # Línea de señal MACD

    # Bollinger Bands
    bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['BB_high'] = bb.bollinger_hband()
    df['BB_low'] = bb.bollinger_lband()

    # ATR
    atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['ATR'] = atr.average_true_range()

    df.dropna(inplace=True)
    df['signal'] = np.select(
        [
            (df['RSI'] < 30) & (df['close'] < df['EMA']) & (df['MACD'] > df['MACD_signal']),
            (df['RSI'] > 70) & (df['close'] > df['EMA']) & (df['MACD'] > df['MACD_signal'])
        ],
        [1, -1],
        default=0
    )
    return df

# === Entrenar modelo si no existe ===
def train_and_save_model(df):
    X = df[['close', 'RSI', 'EMA']]
    y = df['signal']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Guardar metricas de entrenamiento
    report = classification_report(y, model.predict(X), output_dict=True)
    metrics_path = os.path.join(BASE_DIR, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=4)

    joblib.dump(model, MODEL_PATH)

    # Copia de seguridad con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(BASE_DIR, "backups")
    os.makedirs(backup_dir, exist_ok=True)
    backup_path = os.path.join(backup_dir, f"modelo_backup_{timestamp}.pkl")
    joblib.dump(model, backup_path)
    return model

def load_or_train_model(df):
    if not os.path.exists(MODEL_PATH):
        print("🧠 Entrenando modelo por primera vez...")
        return train_and_save_model(df)
    try:
        model = joblib.load(MODEL_PATH)
        if not hasattr(model, "predict"):
            raise ValueError("El modelo cargado no es valido.")
        return model
    except Exception as e:
        log_event(f"❌ Error al cargar el modelo, se reentrenará: {e}")
        send_telegram("❌ Modelo corrupto. Se reentrenara automaticamente.")
        return train_and_save_model(df)

# === Validaciones ===
def get_lot_size_filters(symbol):
    exchange_info = client.get_symbol_info(symbol)
    for f in exchange_info['filters']:
        if f['filterType'] == 'LOT_SIZE':
            return { 'minQty': float(f['minQty']), 'stepSize': float(f['stepSize'])}
    raise Exception("No se encontró LOT_SIZE")

def adjust_quantity_to_step(quantity: float, step_size: float) -> float:
    precision = abs(Decimal(str(step_size)).as_tuple().exponent)
    return float(round(math.floor(quantity / step_size) * step_size, precision))

def get_assets_from_symbol(symbol: str) -> tuple[str, str]:
    info = client.get_symbol_info(symbol)
    return info['baseAsset'], info['quoteAsset']

def has_sufficient_balance(symbol: str, quantity: float, side: str) -> bool:
    base_asset, quote_asset = get_assets_from_symbol(symbol)
    price = float(client.get_symbol_ticker(symbol=symbol)['price'])
    if side == "BUY":
        required = quantity * price * 1.01
        balance = float(client.get_asset_balance(asset=quote_asset)['free'])
        return balance >= required
    else:
        balance = float(client.get_asset_balance(asset=base_asset)['free'])
        return balance >= quantity

# Logger de operaciones
def log_trade(action, price, quantity, reason):
    trade_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "price": round(price, 2),
        "quantity": quantity,
        "reason": reason
    }
    trades_log_path = TRADES_LOG_FILE
    if os.path.exists(trades_log_path):
        with open(trades_log_path, "r") as f:
            trades = json.load(f)
    else:
        trades = []
    trades.append(trade_data)
    with open(trades_log_path, "w") as f:
        json.dump(trades, f, indent=4)

# === Ejecutar orden ===
@retry_on_exception(max_attempts=3)
def place_order(symbol, signal, quantity, max_quantity, reason="MODEL_SIGNAL"):
    filters = get_lot_size_filters(symbol)
    qty = adjust_quantity_to_step(quantity, filters['stepSize'])
    if qty > max_quantity:
        return f"❌ Cantidad de orden {qty} excede el limite maximo permitido de {max_quantity}."

    if qty < filters['minQty']:
        return f"⚠️ Cantidad mínima no alcanzada: {qty}"
    if signal == 1:
        if has_sufficient_balance(symbol, qty, "BUY"):
            order = client.order_market_buy(symbol=symbol, quantity=qty)

            if order['status'] != 'FILLED':
                log_event(f"⚠️ [{symbol}] Orden de COMPRA no completamente FILLED: {order}")
                send_telegram("⚠️ Atencion: Orden de COMPRA no fue completamente FILLED.")
                return f"⚠️ Orden no FILLED completamente: {order['status']}"
            fills = order.get("fills", [])
            if not fills:
                raise ValueError("❌ La orden fue ejecutada pero no contiene informacion de fills.")
            price = float(fills[0]['price'])
            log_trade("BUY", price, qty, reason)
            return f"✅ COMPRA ejecutada: {order}"
        else:
            return "❌ Fondos insuficientes para COMPRAR."
    elif signal == -1:
        if has_sufficient_balance(symbol, qty, "SELL"):
            order = client.order_market_sell(symbol=symbol, quantity=qty)

            if order['status'] != 'FILLED':
                log_event(f"⚠️ [{symbol}] Orden de VENTA no completamente FILLED: {order}")
                send_telegram("⚠️ Atencion: Orden de VENTA no fue completamente FILLED.")
                return f"⚠️ Orden no FILLED completamente: {order['status']}"
            fills = order.get("fills", [])
            if not fills:
                raise ValueError("❌ La orden fue ejecutada pero no contiene informacion de fills.")
            price = float(fills[0]['price'])
            log_trade("SELL", price, qty, reason)
            return f"✅ VENTA ejecutada: {order}"
        else:
            return "❌ Fondos insuficientes para VENDER."
    else:
        return "ℹ️ HOLD: No se ejecutó ninguna orden."

# === Función para obtener el balance ===
def get_balances() -> tuple[float, float]:
    usdt = client.get_asset_balance(asset='USDT')
    btc = client.get_asset_balance(asset='BTC')
    return float(usdt['free']), float(btc['free'])

def get_bot_status():
    try:
        with open(STATUS_FILE, "r") as f:
            status = json.load(f)
        return status.get("active", True)
    except:
        return True # Si no existe, el bot esta activo por defecto

def set_bot_status(active: bool):
    with open(STATUS_FILE, "w") as f:
        json.dump({"active": active}, f)

# Funcion que guarda el precio de compra en un archivo stop_loss.json cada vez que el bot compra
def save_stop_loss_price(price, stop_loss_pct, take_profit_pct, symbol, timestamp=None):  
    take_profit_price = price * (1 + take_profit_pct / 100)
    stop_loss_price = price * (1 - stop_loss_pct / 100)
    data = {
        "symbol": symbol,
        "buy_price": price,
        "highest_price": price, # Inicial, se ira actualizando
        "take_profit_price": take_profit_price,
        "stop_loss_price": stop_loss_price,
        "timestamp": timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    file_path = get_stop_loss_file(symbol)
    lock = get_symbol_lock(symbol)
    with lock:
        with open(file_path, "w") as f:
            json.dump(data, f)
    log_event(f"📌 [{symbol}] Compra registrada: {price:.2f} | SL: {stop_loss_price:.2f} TP: {take_profit_price:.2f}")

# Funcion que mantiene highest_price actualizado si el precio sigue subiendo despues de la compra
def update_highest_price(current_price, symbol):
    file_path = get_stop_loss_file(symbol)
    lock = get_symbol_lock(symbol)
    if not os.path.exists(file_path):
        return
    try:
        with lock:
            with open(file_path, "r") as f:
                data = json.load(f)
            
            highest = data.get("highest_price", 0)
            if current_price > highest:
                data["highest_price"] = current_price
                with open(file_path, "w") as f:
                    json.dump(data, f)
                log_event(f"📈 [{symbol}] Nuevo maximo registrado para trailing stop: {current_price:.2f}")
    except Exception as e:
        log_event(f"❌ [{symbol}] Error al actualizar highest_price: {e}")

# Funcion que compara el precio actual contra el precio de compra registrado, aplicando el % de perdida maxima
def check_stop_loss(current_price, stop_loss_pct, trailing_stop_min_drop, symbol) -> tuple[bool, Optional[dict]]:
    file_path = get_stop_loss_file(symbol)
    lock = get_symbol_lock(symbol)
    try:
        if not os.path.exists(file_path):
            return False, None
        with lock:
            data = load_validated_stop_loss(symbol)
        if data is None:
            return False, None

        buy_price = float(data.get("buy_price", 0))
        highest_price = float(data.get("highest_price", buy_price))

        if buy_price <= 0:
            log_event(f"⚠️ [{symbol}] Precio de compra invalido en stop_loss.json.")
            return False, None
        # Trailing stop logic
        threshold = highest_price * (1 - stop_loss_pct / 100)
        min_absolute_drop = highest_price * (trailing_stop_min_drop / 100)
        if current_price <= threshold and (highest_price - current_price) > min_absolute_drop:
            log_event(f"🚨 [{symbol}] Trailing Stop activado: Precio actual {current_price:.2f} <= {threshold:.2f} (desde maximo {highest_price:.2f}).")
            return True, data
        return False, None
    except Exception as e:
        log_event(f"❌ [{symbol}] Error al verificar Trailing Stop: {e}")
        return False, None

# Funcion para chequear el take profit
def check_take_profit(current_price, symbol):
    file_path = get_stop_loss_file(symbol)
    lock = get_symbol_lock(symbol)
    try:
        if not os.path.exists(file_path):
            return False
        with lock:
            data = load_validated_stop_loss(symbol)
        if data is None:
            return False
        take_profit_price = float(data.get("take_profit_price", 0))
        if take_profit_price <= 0:
            return False
        if current_price >= take_profit_price:
            mensaje = (
                f"🎯 Take Profit alcanzado:\n"
                f"📈 Precio actual: {current_price:.2f}\n"
                f"🎯 TP configurado: {take_profit_price:.2f}\n"
                f"⏳ No se vende automáticamente, sigue activa la lógica del Trailing Stop."
            )
            log_event(mensaje)
            send_telegram(mensaje)

            # ✅ Marcar que se alcanzó el TP
            data["take_profit_reached"] = True
            with open(file_path, "w") as f:
                json.dump(data, f)

            return True
        return False
    except Exception as e:
        log_event(f"❌ [{symbol}] Error al verificar Take Profit: {e}")
        return False

# Funcion para borrar el archivo de stop_loss.json despues de una venta para evitar usarlo incorrectamente despues
def clear_stop_loss(symbol):
    file_path = get_stop_loss_file(symbol)
    lock = get_symbol_lock(symbol)
    with lock:
        if os.path.exists(file_path):
            os.remove(file_path)
            log_event(f"🧹 [{symbol}] Archivo stop_loss.json eliminado tras venta.")

# Funcion para manejar last signal para evitar que el bot compre mas de una vez seguido sin necesidad o venda lo que ya no tiene
def get_last_signal() -> Optional[str]:
    if not os.path.exists(LAST_SIGNAL_FILE):
        return None
    try:
        with open(LAST_SIGNAL_FILE, "r") as f:
            data = json.load(f)
            return data.get("signal")
    except Exception as e:
        log_event(f"❌ Error leyendo last_signal.json: {e}")
        return None

def save_last_signal(signal_str):
    if signal_str not in ["BUY", "SELL", "HOLD"]:
        log_event(f"⚠️ Señal invalida al guardar: {signal_str}")
        return
    try:
        with open(LAST_SIGNAL_FILE, "w") as f:
            json.dump({"signal": signal_str}, f)
    except Exception as e:
        log_event(f"❌ Error guardando last_signal.json: {e}")


# Funcion para verificar que hay dentro del archivo durante las pruebas
def inspect_stop_loss_file(symbol):
    file_path = get_stop_loss_file(symbol)
    if not os.path.exists(file_path):
        print("❌ No existe stop_loss.json")
        return
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        print(f"📄 [{symbol}] Contenido actual de stop_loss.json: " + json.dumps(data, indent=4))
        print(json.dumps(data, indent=4))
    except Exception as e:
        print(f"❌ Error al leer stop_loss para {symbol}: {e}")


def load_validated_stop_loss(symbol) -> Optional[dict]:
    file_path = get_stop_loss_file(symbol)
    lock = get_symbol_lock(symbol)
    if not os.path.exists(file_path):
        return None
    try:
        with lock:
            with open(file_path, "r") as f:
                data = json.load(f)
        # Validaciones estrictas
        buy_price = float(data.get("buy_price", 0))
        highest_price = float(data.get("highest_price", 0))
        take_profit_price = float(data.get("take_profit_price", 0))

        if buy_price <= 0 or highest_price <= 0 or take_profit_price <= 0:
            raise ValueError("Valores inválidos en stop_loss.json.")

        return data
    except Exception as e:
        log_event(f"⚠️ [{symbol}] stop_loss.json inválido. Se eliminará. Motivo: {e}")
        os.remove(file_path)
        return None


# === Estrategias ===
# Estrategia 1: RSI + MACD (ejemplo para comprar cuando RSI<30 y MACD>Signal)
def strategy_rsi_macd(df):
    try:
        latest = df.iloc[-1]
        if latest['RSI'] < 30 and latest['MACD'] > latest['MACD_signal']:
            log_event("Estrategia_rsi_macd: Señal de COMPRA")
            return 1 # Compra
        elif latest['RSI'] > 70 and latest['MACD'] < latest['MACD_signal']:
            log_event("Estrategia_rsi_macd: Señal de VENTA")
            return -1 # Venta
        else:
            log_event("Estrategia_rsi_macd: Señal NEUTRAL")
            return 0 # Neutral
    except Exception as e:
        log_event(f"❌ Error en la estrategia RSI_MACD: {e}")
        return 0

# Estrategia 2: EMA + Bollinger Bands (ejemplo compra por rebote en banda baja + precio > EMA)
def strategy_ema_bb(df):
    try:
        latest = df.iloc[-1]
        if latest['close'] < latest['BB_low'] and latest['close'] > latest['EMA']:
            log_event("Estrategia_ema_bb: Señal de COMPRA")
            return 1
        elif latest['close'] > latest['BB_high'] and latest['close'] < latest['EMA']:
            log_event("Estrategia_ema_bb: Señal de VENTA")
            return -1
        else:
            log_event("Estrategia_ema_bb: Señal NEUTRAL")
            return 0
    except Exception as e:
        log_event(f"❌ Error en la estrategia EMA_BB: {e}")
        return 0

# Estrategia 3: ATR + RSI (ejemplo volatilidad alta y RSI extramos)
def strategy_atr_rsi(df):
    try:
        latest = df.iloc[-1]
        if latest['ATR'] > df['ATR'].rolling(10).mean().iloc[-1] and latest['RSI'] < 35:
            log_event("Estrategia_atr_rsi: Señal de COMPRA")
            return 1
        elif latest['ATR'] > df['ATR'].rolling(10).mean().iloc[-1] and latest['RSI'] > 65:
            log_event("Estrategia_atr_rsi: Señal de VENTA")
            return -1
        else:
            log_event("Estrategia_atr_rsi: Señal NEUTRAL")
            return 0
    except Exception as e:
        log_event(f"❌ Error en la estrategia ATR_RSI: {e}")
        return 0

# Estrategia 4: Señal del modelo ML actual (opcional, si se quiere mantener)
def strategy_ml(df, model):
    try:
        latest = df.iloc[-1]
        features = latest[['close', 'RSI', 'EMA']]
        if features.isnull().any():
            log_event("Estrategia_ml: Hay NaN en features, se retorna NEUTRAL")
            return 0
        pred = model.predict(features.to_frame().T)[0]
        log_event(f"Estrategia_ml: Señal del modelo: {pred}")
        return int(pred)
    except Exception as e:
        log_event(f"❌ Error en Estrategia_ml: {e}")
        return 0

active_strategies = [
    ("RSI_MACD", strategy_rsi_macd),
    ("EMA_BB", strategy_ema_bb),
    ("ATR_RSI", strategy_atr_rsi)
]



# === Lógica principal. Ejecucion programada: Este script esta diseñado para ejecutarse periodicamente (e.g. cada hora con cron) ===
def main():
    cleanup_logs() # Limpieza automatica si el log es muy grande
    config = load_config() # Lee dinámicamente config.json

    if not get_bot_status():
        print("⏸️ Bot desactivado por el usuario.")
        return
    
    # Limita el número de operaciones abiertas
    operaciones_abiertas = 0

    try:
        for market in config.get("markets", []):
            symbol = market.get("symbol")
            interval = market.get("interval", "1h")
            risk_pct = market.get("risk_pct", 1)
            stop_loss_pct = market.get("stop_loss_pct", 5)
            take_profit_pct = market.get("take_profit_pct", 10)
            trailing_stop_min_drop = float(market.get("trailing_stop_min_drop", 0.2))
            strategies_names = market.get("strategies", ["RSI_MACD", "EMA_BB", "ATR_RSI", "ML"])
            max_quantity = float(market.get("max_quantity", config.get("limits", {}).get("max_quantity", 0.01)))
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"📥 === Procesando {symbol} {interval} [{now}] ===")

            # Descargar datos históricos y calcular indicadores
            try:
                df = get_price_data(symbol, interval=interval)
                if df.empty:
                    raise ValueError(f"❌ DataFrame vacio: No se obtuvieron datos para {symbol}.")
                df = add_indicators(df)
            except Exception as e:
                log_event(f"❌ Error al descargar datos {symbol}: {e}")
                if config.get("telegram_notifications", {}).get("errors", True):
                    send_telegram(f"Error al cargar datos de {symbol}: {e}", alert_type="ERROR")
                continue

            # PIPELINE TRAILING STOP
            price = df.iloc[-1]["close"]
            update_highest_price(price, symbol)
            stop_triggered, stop_data = check_stop_loss(price, stop_loss_pct, trailing_stop_min_drop, symbol)
            if stop_triggered:
                # Venta inmediata por trailing (independiente del consenso)
                order_size_usdt = get_balances()[0] * (risk_pct / 100.0)
                filters = get_lot_size_filters(symbol)
                theoretical_qty = order_size_usdt / price
                qty = adjust_quantity_to_step(theoretical_qty, filters['stepSize'])
                order_result = place_order(symbol, -1, qty, max_quantity, reason="TRAILING_STOP")
                clear_stop_loss(symbol)
                log_event(f"[{symbol}] ❗ Trailing stop activado. Venta ejecutada. {order_result}")
                send_telegram(f"[{symbol}] 🚨 Trailing stop ejecutado a {price:.2f}", alert_type="WARNING")
                save_last_signal("SELL")
                continue  # Salta a siguiente mercado

            check_take_profit(price, symbol)  # (esto solo actualiza el flag y notifica si hace falta)

            try:
                tipo = validate_strategy_types(strategies_names, config)
                print(f"Estrategias {strategies_names} configuradas para {symbol}, tipo: {tipo}")
            except Exception as e:
                log_event(f"❌ [VALIDACION] {e}")
                send_telegram(f"🚨 [VALIDACION] {e}")
                continue

            # PIPELINE CONSENSO DE ESTRATEGIAS
            try:
                model = load_or_train_model(df)
            except Exception as e:
                log_event(f"❌ Error al cargar/entrenar modelo para {symbol}: {e}")
                send_telegram(f"Error al cargar/entrenar modelo para {symbol}: {e}", alert_type="ERROR")
                continue

            # Configura lista de estrategias activas para este mercado. Mapea los nombres del JSON a las funciones reales
            strategy_map = {
                "RSI_MACD": strategy_rsi_macd,
                "EMA_BB": strategy_ema_bb,
                "ATR_RSI": strategy_atr_rsi,
                "ML": lambda df: strategy_ml(df, model)
            }
            current_strategies = [(name, strategy_map[name]) for name in strategies_names if name in strategy_map]

            # Ejecutar consenso de estrategias
            results = []
            for name, func in current_strategies:
                try:
                    result = func(df)
                except Exception as e:
                    log_event(f"❌ Error en la estrategia {name} para {symbol}: {e}")
                    result = 0
                results.append((name, result))
            
            total_strat = len(results)
            buy_signals = sum(1 for _, r in results if r == 1)
            sell_signals = sum(1 for _, r in results if r == -1)
            neutral_signals = sum(1 for _, r in results if r == 0)

            buy_prob = buy_signals / total_strat
            sell_prob = sell_signals / total_strat
            prob_threshold = config.get("telegram_notifications", {}).get("probability_threshold", 0.8)
            if buy_prob >= prob_threshold:
                final_signal = 1
                final_text = "BUY"
                final_prob = buy_prob
            elif sell_prob >= prob_threshold:
                final_signal = -1
                final_text = "SELL"
                final_prob = sell_prob
            else:
                final_signal = 0
                final_text = "HOLD"
                final_prob = max(buy_prob, sell_prob)
            
            # Logging y reporte completo
            log_event(f"[{symbol}] [VOTACIÓN] Resultados de estrategias: {results}")
            log_event(
                f"[{symbol}] [VOTACIÓN] COMPRA: {buy_signals}/{total_strat} ({buy_prob*100:.1f}%) | "
                f"VENTA: {sell_signals}/{total_strat} ({sell_prob*100:.1f}%) | "
                f"NEUTRAL: {neutral_signals}/{total_strat} | "
                f"Señal Final: {final_text} ({final_prob*100:.1f}%)"
            )

            details_txt = "\n".join([f" - {n}: {'COMPRA' if r==1 else 'VENTA' if r==-1 else 'NEUTRAL'}" for n, r in results])

            # Evitar duplicidad de señales
            last_signal = get_last_signal()
            if final_text == last_signal:
                log_event(f"[{symbol}] {now}\nSeñal duplicada detectada: {final_text}. No se ejecuta orden.")
                continue

            usdt_balance, btc_balance = get_balances()
            order_desc = "COMPRA ✅" if final_signal == 1 else "VENTA ✅" if final_signal == -1 else "HOLD ⏸️"
            message = (
                f"🔎 Señal final: {order_desc}\n"
                f"📊 Probabilidad: {final_prob*100:.1f}% (umbral >= 80%)\n"
                f"➡️ Estrategias activas: {total_strat} | COMPRA={buy_signals} VENTA={sell_signals} NEUTRAL={neutral_signals}\n"
                f"🏷️ Detalle por estrategia:\n{details_txt}\n"
                f"💰 BTC: {btc_balance:.4f} | USDT: {usdt_balance:.2f}\n"
                f"🗓️ Fecha: {now}"
            )
            # Notificación por Telegram según configuración
            if config.get("telegram_notifications", {}).get("executions", True):
                send_telegram(message)
            log_event(f"[TELEGRAM]: {message}")

            # Solo ejecutar si hay consenso
            if final_signal != 0:
                # Tamaño de lote dinámico, 1% del capital disponible
                order_size_usdt = usdt_balance * (risk_pct / 100.0)
                filters = get_lot_size_filters(symbol)
                price = df.iloc[-1]['close']
                theoretical_qty = order_size_usdt / price
                qty = adjust_quantity_to_step(theoretical_qty, filters['stepSize'])
                # Verificamos límite de operación, SL y TP
                order_result = place_order(symbol, final_signal, qty, max_quantity, reason=f"CONSENSO_{final_prob*100:.1f}")
                
                # Aplica el SL/TP según la config y guarda en archivo para trailing/reporte post
                if final_signal == 1:
                    # Entrando en compra, guarda precios de SL/TP configurados
                    save_stop_loss_price(price, stop_loss_pct, take_profit_pct, symbol, now)
                elif final_signal == -1:
                    clear_stop_loss(symbol)
                save_last_signal(final_text)
                log_event(order_result)
                operaciones_abiertas += 1

                # Límite de operaciones abiertas
                max_trades = config.get("limits", {}).get("max_open_trades", 2)
                if operaciones_abiertas >= max_trades:
                    log_event(f"⚠️ [{symbol}] Límite de operaciones abiertas alcanzado: {operaciones_abiertas}/{max_trades}")
                    break
            
            # Espera entre mercados
            sleep_seconds = config.get("limits", {}).get("sleep_seconds", 60)
            print(f"⏳ Esperando {sleep_seconds} segundos antes de next symbol...")
            time.sleep(sleep_seconds)
    except Exception as e:
        error_message = f"❌ Error durante el ciclo multi-mercado: {str(e)}"
        log_event(error_message)
        send_telegram(error_message)
        print(error_message)

if __name__ == "__main__":
    main()