import os
import json
import subprocess
import joblib
import requests
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
STOP_LOSS_FILE = os.path.join(BASE_DIR, "stop_loss.json")
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
        lambda: check_file_exists(STOP_LOSS_FILE, "stop_loss.json") if os.path.exists(STOP_LOSS_FILE) else None,
        lambda: check_json_valid(STOP_LOSS_FILE, "stop_loss.json") if os.path.exists(STOP_LOSS_FILE) else None,
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