import os
import json
import subprocess
import joblib
import requests
import datetime
from dotenv import load_dotenv
from binance.client import Client

# === Cargar entorno ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# === Configuración ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
USE_TESTNET = True
MODEL_PATH = os.path.join(BASE_DIR, "modelo_trading.pkl")
STOP_LOSS_DIR = os.path.join(BASE_DIR, "stop_loss")
STATUS_FILE = os.path.join(BASE_DIR, "bot_status.json")
ENV_FILE = os.path.join(BASE_DIR, ".env")

API_KEY = os.getenv("BINANCE_TESTNET_API_KEY") if USE_TESTNET else os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_TESTNET_API_SECRET") if USE_TESTNET else os.getenv("BINANCE_API_SECRET")

client = Client(API_KEY, API_SECRET)
if USE_TESTNET:
    client.API_URL = 'https://testnet.binance.vision/api'

# === Enviar mensaje por Telegram ===
def send_telegram_alert(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": f"🚨 VERIFICACIÓN SISTEMA:\n{msg}"})

# === Funciones de verificación ===
def check_file_exists(path, name):
    if not os.path.exists(path):
        return f"❌ Archivo faltante: {name}"
    return None

def check_json_valid(path, name):
    try:
        with open(path, "r") as f:
            json.load(f)
    except:
        return f"❌ JSON malformado: {name}"
    return None

def check_model():
    try:
        model = joblib.load(MODEL_PATH)
        if not hasattr(model, "predict"):
            return "❌ Modelo inválido (sin método predict)"
    except:
        return "❌ Error al cargar el modelo .pkl"
    return None

def check_env():
    if not os.path.exists(ENV_FILE):
        return "❌ Falta el archivo .env"
    return None

def check_balance():
    try:
        btc = client.get_asset_balance(asset='BTC')
        usdt = client.get_asset_balance(asset='USDT')
        if not btc or not usdt:
            return "❌ Error accediendo al balance de Binance"
    except Exception as e:
        return f"❌ Binance API error: {e}"
    return None

def check_service():
    try:
        result = subprocess.run(["systemctl", "is-active", "--quiet", "telegram_listener.service"])
        if result.returncode != 0:
            return "❌ Servicio 'telegram_listener' está detenido"
    except:
        return "❌ Error al verificar el servicio 'telegram_listener'"
    return None

def check_cron():
    try:
        result = subprocess.run(["crontab", "-l"], stdout=subprocess.PIPE, text=True)
        if "trading_bot_final.py" not in result.stdout:
            return "⚠️ No se encontró la tarea cron para trading_bot_final.py"
    except:
        return "❌ Error al verificar cron"
    return None

def check_all_stop_loss_files():
    errores = []
    if not os.path.exists(STOP_LOSS_DIR):
        return None  # No hay operaciones activas, no es un error

    for fname in os.listdir(STOP_LOSS_DIR):
        if fname.endswith(".json"):
            path = os.path.join(STOP_LOSS_DIR, fname)
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                # 1. Chequeo de presencia de campos
                required = ["symbol", "buy_price", "highest_price", "take_profit_price", "stop_loss_price", "timestamp"]
                for clave in required:
                    if clave not in data:
                        send_telegram_paranoid_alert(fname, f"FALTA {clave}")
                        errores.append(f"❌ FALTA {clave} en {fname}")

                # 2. Chequeo de tipos y valores positivos
                for clave in ["buy_price", "highest_price", "take_profit_price", "stop_loss_price"]:
                    valor = data.get(clave, None)
                    if clave not in data or not isinstance(valor, (float, int)) or valor <= 0:
                        send_telegram_paranoid_alert(fname, f"{clave} inválido o no positivo ({valor})")
                        errores.append(f"❌ {clave} inválido o no positivo en {fname} (valor: {valor})")

                # 3. Congruencia de precios
                buy = data.get("buy_price", 0)
                highest = data.get("highest_price", 0)
                sl = data.get("stop_loss_price", 0)
                tp = data.get("take_profit_price", 0)
                if highest < buy:
                    send_telegram_paranoid_alert(fname, f"highest_price < buy_price ({highest} < {buy})")
                    errores.append(f"❌ highest_price < buy_price en {fname} ({highest} < {buy})")
                if not (sl < buy < tp):
                    send_telegram_paranoid_alert(fname, f"stop_loss < buy < take_profit no cumple: SL:{sl}, BUY:{buy}, TP:{tp}")
                    errores.append(f"❌ stop_loss < buy < take_profit no se cumple en {fname} (SL: {sl} | BUY: {buy} | TP: {tp})")

                # 4. Timestamp
                ts = data.get("timestamp", "")
                try:
                    dt = datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                    if dt > datetime.datetime.now() + datetime.timedelta(minutes=5):
                        send_telegram_paranoid_alert(fname, f"timestamp futuro ({ts})")
                        errores.append(f"❌ timestamp {ts} es futuro en {fname}")
                except Exception as e:
                    send_telegram_paranoid_alert(fname, f"timestamp ilegible: {e}")
                    errores.append(f"❌ timestamp ilegible ({ts}) en {fname}: {e}")

                # 5. Coincidencia symbol-archivo
                symbol_expect = fname.replace("stop_loss_", "").replace(".json", "")
                if data.get("symbol", "").upper() != symbol_expect.upper():
                    send_telegram_paranoid_alert(fname, f"symbol en archivo ('{data.get('symbol')}') != de filename ({symbol_expect})")
                    errores.append(f"❌ symbol ('{data.get('symbol')}') distinto al filename ({symbol_expect}) en {fname}")

                # 6. Campo take_profit_reached, si existe, debe ser bool
                tpr = data.get("take_profit_reached", None)
                if tpr is not None and not isinstance(tpr, bool):
                    send_telegram_paranoid_alert(fname, "take_profit_reached no booleano")
                    errores.append(f"❌ take_profit_reached no booleano en {fname}")

            except Exception as e:
                send_telegram_paranoid_alert(fname, f"Error leyendo/parsing: {e}")
                errores.append(f"❌ Error leyendo/parsing {fname}: {e}")

    if errores:
        return "\n".join(errores)
    return None

def send_telegram_paranoid_alert(filename, error_msg):
    recomendaciones = (
        "🦺 Recomendación:\n"
        "- Revise los logs del bot para detalles técnicos.\n"
        "- Si este archivo corresponde a una orden ya cerrada o si está corrupto, puede eliminarlo aquí abajo.\n"
        "- Si persiste el error, consulte con soporte."
    )
    mensaje = (
        f"⚠️ Archivo trailing stop defectuoso:\n"
        f"Archivo: {filename}\n"
        f"➡️ Detalle: {error_msg}\n\n"
        f"{recomendaciones}"
    )
    send_telegram_with_buttons(mensaje, filename)

def send_telegram_with_buttons(text, filename):
    import json as _json
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    keyboard = {
        "inline_keyboard": [
            [{"text": "🗑️ Eliminar archivo", "callback_data": f"DEL_{filename}"}],
            [{"text": "🟢 Ignorar alerta", "callback_data": f"IGNORE_{filename}"}],
            [{"text": "🔄 Reiniciar bot", "callback_data": "RESTART_BOT"}]
        ]
    }
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "reply_markup": _json.dumps(keyboard)
    }
    requests.post(url, data=payload)


# === Ejecutar chequeos ===
def main():
    errores = []

    for check in [
        lambda: check_env(),
        lambda: check_file_exists(MODEL_PATH, "modelo_trading.pkl"),
        lambda: check_model(),
        lambda: check_file_exists(STATUS_FILE, "bot_status.json"),
        lambda: check_json_valid(STATUS_FILE, "bot_status.json"),
        lambda: check_service(),
        lambda: check_cron(),
        lambda: check_balance(),
        lambda: check_all_stop_loss_files()
    ]:
        error = check()
        if error:
            errores.append(error)

    if errores:
        mensaje = "\n".join(errores)
        send_telegram_alert(mensaje)
        print("❌ Problemas detectados:\n", mensaje)
    else:
        print("✅ Todo correcto en la verificación del sistema.")

if __name__ == "__main__":
    main()