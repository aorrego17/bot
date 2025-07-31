import os
import math
import pandas as pd
import numpy as np
import joblib
import requests
import json
import time
from datetime import datetime
from binance.client import Client
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from sklearn.ensemble import RandomForestClassifier
from decimal import Decimal
from typing import Optional
from sklearn.metrics import classification_report
from ta.volatility import BollingerBands
from ta.volatility import AverageTrueRange
from ta.trend import MACD
from dotenv import load_dotenv
load_dotenv() # Esto carga las variables del .env en entorno

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

# === Ruta del directorio actual del script trading_bot_final.py ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "log.txt")
MODEL_PATH = os.path.join(BASE_DIR, "modelo_trading.pkl")
STATUS_FILE = os.path.join(BASE_DIR, "bot_status.json")
STOP_LOSS_FILE = os.path.join(BASE_DIR, "stop_loss.json")
LAST_SIGNAL_FILE = os.path.join(BASE_DIR, "last_signal.json")
TRADES_LOG_FILE = os.path.join(BASE_DIR, "trades_log.json")

# === Cargar variables de entorno ===
USE_TESTNET = True  # Cambia a False para cuenta real
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
DEFAULT_QUANTITY = float(os.getenv("DEFAULT_QUANTITY", 0.001))
MAX_QUANTITY = float(os.getenv("MAX_QUANTITY", 0.01))
TRAILING_STOP_MIN_DROP_PERCENT = float(os.getenv("TRAILING_STOP_MIN_DROP_PERCENT", 0.2))
STOP_LOSS_PERCENTAGE = 5 # % de perdida maxima permitida
TAKE_PROFIT_PERCENTAGE = 10 # % de ganancia objetivo

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

def send_telegram_with_buttons():
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return "⚠️ Telegram no configurado. No se envió el mensaje."
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    message = "🧠 ¿Que deseas hacer con el bot?"
    keyboard = {
        "inline_keyboard": [
            [
                {"text": "✅ Reiniciar", "callback_data": "REINICIAR"},
                {"text": "❌ No hacer nada", "callback_data": "DETENER"}
            ]
        ]
    }
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "reply_markup": json.dumps(keyboard)
    }

    try:
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            return "📨 Botones enviados correctamente."
        else:
            return f"❌ Error enviando botones: {response.text}"
    except Exception as e:
        return f"❌ Excepción en la solicitud Telegram: {e}"
        
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

    # Imprime las primeras filas y los nombres de las columnas
    print("Datos descargados de Binance:")
    print(df.head())
    print("Columnas disponibles:", df.columns)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Convertir todas las columnas apropiadas a numérico
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'taker_buy_base_volume', 'taker_buy_quote_volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    # Verificar la conversión
    print("Tipos de datos después de conversiones:")
    print(df.dtypes)

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
        log_event(f"❌ Error al cargar el modelo, se reentrenara: {e}")
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
def place_order(symbol, signal, quantity, reason="MODEL_SIGNAL"):
    filters = get_lot_size_filters(symbol)
    qty = adjust_quantity_to_step(quantity, filters['stepSize'])
    if qty > MAX_QUANTITY:
        return f"❌ Cantidad de orden {qty} excede el limite maximo permitido de {MAX_QUANTITY}."

    if qty < filters['minQty']:
        return f"⚠️ Cantidad mínima no alcanzada: {qty}"
    if signal == 1:
        if has_sufficient_balance(symbol, qty, "BUY"):
            order = client.order_market_buy(symbol=symbol, quantity=qty)

            if order['status'] != 'FILLED':
                log_event(f"⚠️ Orden de COMPRA no completamente FILLED: {order}")
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
                log_event(f"⚠️ Orden de VENTA no completamente FILLED: {order}")
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
def save_stop_loss_price(price):
    take_profit_price = price * (1 + TAKE_PROFIT_PERCENTAGE / 100)
    data = {
        "buy_price": price,
        "highest_price": price, # Inicial, se ira actualizando
        "take_profit_price": take_profit_price,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(STOP_LOSS_FILE, "w") as f:
        json.dump(data, f)
    log_event(f"📌 Compra registrada: {price:.2f} | TP: {take_profit_price:.2f}")

# Funcion que mantiene highest_price actualizado si el precio sigue subiendo despues de la compra
def update_highest_price(current_price):
    if not os.path.exists(STOP_LOSS_FILE):
        return
    try:
        with open(STOP_LOSS_FILE, "r") as f:
            data = json.load(f)
        
        highest = data.get("highest_price", 0)
        if current_price > highest:
            data["highest_price"] = current_price
            with open(STOP_LOSS_FILE, "w") as f:
                json.dump(data, f)
            log_event(f"📈 Nuevo maximo registrado para trailing stop: {current_price:.2f}")
    except Exception as e:
        log_event(f"❌ Error al actualizar highest_price: {e}")

# Funcion que compara el precio actual contra el precio de compra registrado, aplicando el % de perdida maxima
def check_stop_loss(current_price) -> tuple[bool, Optional[dict]]:
    try:
        if not os.path.exists(STOP_LOSS_FILE):
            return False, None
        data = load_validated_stop_loss()
        if data is None:
            return False, None

        buy_price = float(data.get("buy_price", 0))
        highest_price = float(data.get("highest_price", buy_price))

        if buy_price <= 0:
            log_event("⚠️ Precio de compra invalido en stop_loss.json.")
            return False, None

        # Trailing stop logic
        threshold = highest_price * (1 - STOP_LOSS_PERCENTAGE / 100)
        min_absolute_drop = highest_price * (TRAILING_STOP_MIN_DROP_PERCENT / 100)
        if current_price <= threshold and (highest_price - current_price) > min_absolute_drop:
            log_event(f"🚨 Trailing Stop activado: Precio actual {current_price:.2f} <= {threshold:.2f} (desde maximo {highest_price:.2f}).")
            return True, data
        return False, None
    except Exception as e:
        log_event(f"❌ Error al verificar Trailing Stop: {e}")
        return False, None

# Funcion para chequear el take profit
def check_take_profit(current_price):
    try:
        if not os.path.exists(STOP_LOSS_FILE):
            return False
        data = load_validated_stop_loss()
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
            with open(STOP_LOSS_FILE, "w") as f:
                json.dump(data, f)

            return True
        return False
    except Exception as e:
        log_event(f"❌ Error al verificar Take Profit: {e}")
        return False

# Funcion para borrar el archivo de stop_loss.json despues de una venta para evitar usarlo incorrectamente despues
def clear_stop_loss():
    if os.path.exists(STOP_LOSS_FILE):
        os.remove(STOP_LOSS_FILE)
        log_event("🧹 Archivo stop_loss.json eliminado tras venta.")

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
def inspect_stop_loss_file():
    if not os.path.exists(STOP_LOSS_FILE):
        print("❌ No existe stop_loss.json")
        return
    try:
        with open(STOP_LOSS_FILE, "r") as f:
            data = json.load(f)
        print("📄 Contenido actual de stop_loss.json:")
        print(json.dumps(data, indent=4))
    except Exception as e:
        print(f"❌ Error al leer stop_loss.json: {e}")


def load_validated_stop_loss() -> Optional[dict]:
    if not os.path.exists(STOP_LOSS_FILE):
        return None
    try:
        with open(STOP_LOSS_FILE, "r") as f:
            data = json.load(f)

        # Validaciones estrictas
        buy_price = float(data.get("buy_price", 0))
        highest_price = float(data.get("highest_price", 0))
        take_profit_price = float(data.get("take_profit_price", 0))

        if buy_price <= 0 or highest_price <= 0 or take_profit_price <= 0:
            raise ValueError("Valores inválidos en stop_loss.json.")

        return data
    except Exception as e:
        log_event(f"⚠️ stop_loss.json inválido. Se eliminará. Motivo: {e}")
        os.remove(STOP_LOSS_FILE)
        return None


# === Lógica principal. Ejecucion programada: Este script esta diseñado para ejecutarse periodicamente (e.g. cada hora con cron) ===
def main():
    cleanup_logs() # Limpieza automatica si el log es muy grande
    if not get_bot_status():
        print("⏸️ Bot desactivado por el usuario.")
        return
    try:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("📥 Descargando datos y generando señales...")
        df = get_price_data(SYMBOL)
        if df.empty:
            raise ValueError("❌ DataFrame vacio: No se obtuvieron datos historicos.")
        df = add_indicators(df)

        # Verifica el contenido del DataFrame después de agregar indicadores
        print("Indicadores añadidos a DataFrame:")
        print(df.tail()) # Imprime las últimas filas para asegurar que los indicadores fueron añadidos

        # Cargar modelo de forma robusta. Si no existe o esta corrupto, se entrena de nuevo
        model = load_or_train_model(df)
        
        latest = df.iloc[-1]
        price = latest['close']
        update_highest_price(price)
        rsi = latest['RSI']
        ema = latest['EMA']

        # Verificar si el precio activa el stop loss: solo una vez y ejecutar venta directa
        stop_triggered, stop_data = check_stop_loss(price)
        if stop_triggered:
            tp_reached = stop_data.get("take_profit_reached", False)
            buy_price = float(stop_data.get("buy_price", 0))
            real_profit_pct = None

            orden_resultado = place_order(SYMBOL, -1, DEFAULT_QUANTITY, reason="TRAILING_STOP")

            # Calcular ganancia real si corresponde
            if tp_reached and buy_price > 0:
                real_profit_pct = ((price - buy_price) / buy_price) * 100
                try:
                    if real_profit_pct > 0:
                        stop_data["real_profit_pct"] = round(real_profit_pct, 2)
                    with open(STOP_LOSS_FILE, "w") as f:
                        json.dump(stop_data, f, indent=4)
                except Exception as e:
                    log_event(f"⚠️ No se pudo guardar el % real de ganancia: {e}")

            # Limpiar archivo stop_loss
            clear_stop_loss()

            if tp_reached:
                message = (
                    f"💰 Ganancia protegida tras Take Profit:\n"
                    f"🚨 Venta ejecutada por Trailing Stop a {price:.2f}\n"
                    f"📊 Ganancia real: {real_profit_pct:.2f}%\n"
                    f"🧠 El bot aseguró beneficios incluso si el mercado se revirtió."
                )
            else:
                message = f"🚨 Stop Loss ejecutado a {price:.2f}\n{orden_resultado}"

            log_event(message)
            send_telegram(message)
            save_last_signal("SELL")
            return

        # Si no hay Stop Loss, continuamos con el modelo
        features = latest[['close', 'RSI', 'EMA']]

        # Verificar si se activa Take Profit (solo actualiza highest_price si aplica)
        check_take_profit(price) # No ejecuta venta, solo refuerza trailing

        # Predecir señal del modelo ML
        if features.isnull().any():
            raise ValueError("❌ Valores NaN detectados en las features antes de la prediccion.")
        # Si se quiere agregar un control de rango
        if not (0 < features['RSI'] < 100):
            log_event(f"⚠️ RSI fuera de rango: {features['RSI']}")
            return
        
        if pd.isna(rsi) or pd.isna(ema) or pd.isna(price):
            log_event("⚠️ Valores NaN detectados en los indicadores.")
            return
        
        #signal = 1
        signal = model.predict(features.to_frame().T)[0]
        signal_str = "BUY" if signal == 1 else "SELL" if signal == -1 else "HOLD"

        # Verificar si esta señal ya fue ejecutada
        last_signal = get_last_signal()

        # Si es la misma que la ultima señal, evitar ejecucion y envio doble
        if signal_str == last_signal:
            message = f"""
🕒 {now}
📊 Señal duplicada detectada: {signal_str}. No se ejecuta nueva orden.
💵 Precio actual: {price:.2f}
"""
            return
        
        # Como no es duplicada, ahora si obtener balances
        usdt_balance, btc_balance = get_balances()

        # Formatear mensaje final con estilo único
        orden = "COMPRA ✅" if signal == 1 else "VENTA ✅" if signal == -1 else "HOLD ⏸️"
        message = f"""📌 ORDEN EJECUTADA: {orden}
        📈 Precio de entrada: {price:,.2f}
        🧠 Señal del modelo: {signal_str}
        📊 RSI: {rsi:.1f} | EMA: {ema:,.0f}

        💰 Balance BTC: {btc_balance:.4f}
        💵 Balance USDT: {usdt_balance:.2f}

        🧾 Stop Loss: {STOP_LOSS_PERCENTAGE}% | Take Profit: {TAKE_PROFIT_PERCENTAGE}%
        📆 Fecha: {now}
        """

        send_telegram(message)

        # Ejecutar orden
        orden_resultado = place_order(SYMBOL, signal, DEFAULT_QUANTITY, reason="MODEL_SIGNAL")

        # Si fue una compra, guardar precio de compra para el stop loss
        if signal == 1:
            save_stop_loss_price(price)
        # Si fue una venta, limpiar archivo de stop loss
        elif signal == -1:
            clear_stop_loss()
        
        # Guardar nueva señal
        save_last_signal(signal_str)

        # Registrar en log.txt
        log_event(message)

        # Imprimir solo resumen para cron_output.log
        print(f"✅ {now} - Bot ejecutado correctamente.")
    except Exception as e:
        error_message = f"❌ Error en la ejecución: {str(e)}"
        log_event(error_message)
        send_telegram(error_message)
        print(error_message)

#def test_telegram():
#    print("📨 Probando envío de mensaje a Telegram...")
#    result = send_telegram("🚨 Prueba de conexión desde trading_bot_final.py")
#    print(result)

if __name__ == "__main__":
    main()