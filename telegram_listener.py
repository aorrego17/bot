import os
import time
import json
import requests
import logging
from dotenv import load_dotenv
load_dotenv() # Esto carga las variables del .env en entorno

logging.basicConfig(
    filename='bot_errors.log',
    level=logging.ERROR,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATUS_FILE = os.path.join(BASE_DIR, "bot_status.json")
LAST_UPDATE_FILE = os.path.join(BASE_DIR, "last_update_id.txt")
TRADES_FILE = os.path.join(BASE_DIR, "trades_log.json")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TELEGRAM_TOKEN or not CHAT_ID:
    raise RuntimeError("❌ TELEGRAM_TOKEN o TELEGRAM_CHAT_ID no estan definidos en las variables de entorno.")


def send_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text}
    print(f"📨 Enviando mensaje: {text}")
    try:
        safe_post(url, payload, context="send_message")
    except Exception as e:
        print(f"❌ Error al enviar mensaje: {e}")

def set_bot_status(active):
    try:
        with open(STATUS_FILE, "w") as f:
            json.dump({"active": active}, f)
    except Exception as e:
        print(f"❌ Error al guardar el estado del bot: {e}")

def get_bot_status():
    try:
        if os.path.exists(STATUS_FILE):
            with open(STATUS_FILE, "r") as f:
                status = json.load(f)
            return status.get("active", True)
    except Exception as e:
        print(f"❌ No se pudo leer el estado del bot: {e}")
    return True #Predeterminado

def get_last_update_id():
    try:
        if os.path.exists(LAST_UPDATE_FILE):
            with open(LAST_UPDATE_FILE, "r") as f:
                return int(f.read().strip())
    except Exception as e:
        print(f"❌ Error leyendo last_update_id: {e}")
    return None

def save_last_update_id(update_id):
    with open(LAST_UPDATE_FILE, "w") as f:
        f.write(str(update_id))

def safe_post(url, payload, context="", retries=2, delay=1):
    for attempt in range(retries + 1):
        try:
            response = requests.post(url, data=payload)
            if response.ok:
                return
            else:
                logging.error(f"[{context}] Intento {attempt + 1}/{retries + 1} - HTTP {response.status_code}: {response.text}")
        except Exception as e:
            logging.error(f"[{context}] Intento {attempt + 1}/{retries + 1} - Exception: {e}")
        time.sleep(delay)

def poll_telegram():
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
    last_id = get_last_update_id()
    params = {"timeout": 100}
    if last_id:
        params["offset"] = last_id + 1
    
    try:
        response = requests.get(url, params=params)
        if not response.ok:
            print(f"❌ Error en getUpdates: {response.status_code} - {response.text}")
            return
        updates = response.json().get("result", [])
    except Exception as e:
        print(f"❌ Excepcion en poll_telegram(): {e}")
        return

    for update in updates:
        try:
            update_id = update["update_id"]
        except Exception as e:
            logging.error(f"❌ Error procesando update: {e}")
            continue # Saltar si no se puede leer

        # Mensaje de texto (comando escrito)
        message = update.get("message")
        if message and str(message["chat"]["id"]) == CHAT_ID:
            text = message.get("text")
            if text:
                handle_command(text)

        # Botón (callback_query)
        callback = update.get("callback_query")
        if callback and str(callback["from"]["id"]) == CHAT_ID:
            data = callback["data"]
            if data.startswith("/"):
                handle_command(data)
            else:
                handle_callback(callback)
            send_reply(callback["id"], "✅ Comando recibido.")

        save_last_update_id(update_id)

def send_reply(callback_id, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery"
    payload = {
        "callback_query_id": callback_id,
        "text": text,
        "show_alert": False
    }
    safe_post(url, payload, context="send_reply")

def handle_callback(callback):
    data = callback["data"]
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    STOP_LOSS_DIR = os.path.join(BASE_DIR, "stop_loss")

    # Eliminar archivo stop_loss por símbolo
    if data.startswith("DEL_"):
        fname = data.replace("DEL_", "")
        file_path = os.path.join(STOP_LOSS_DIR, fname)
        if os.path.exists(file_path):
            os.remove(file_path)
            send_message(f"🗑️ Archivo {fname} eliminado exitosamente.")
        else:
            send_message(f"El archivo {fname} ya no existe.")

    # Ignorar alerta
    elif data.startswith("IGNORE_"):
        fname = data.replace("IGNORE_", "")
        send_message(f"✅ Alerta ignorada para {fname} (no se realizó ninguna acción).")

    # Reiniciar bot (simulado, aquí deberías integrar systemd o similar si quieres acción real)
    elif data == "RESTART_BOT":
        send_message("🔄 Solicitud de reinicio enviada (a implementar según tu infraestructura).")


def handle_command(command_text):
    print(f"📨 Comando recibido: {command_text}")
    if command_text == "/estado":
        try:
            with open(STATUS_FILE, "r") as f:
                status = json.load(f)
            active = status.get("active", True)
        except Exception as e:
            active = True
        status_text = "🟢 Activo" if active else "🔴 Pausado"
        send_message(f"🤖 Estado del bot: {status_text}")

    elif command_text == "/pausar":
        set_bot_status(False)
        send_message("⏸️ Bot pausado manualmente.")
    
    elif command_text == "/reanudar":
        set_bot_status(True)
        send_message("▶️ Bot reanudado manualmente.")
    
    elif command_text == "/ultima_operacion":
        try:
            if not os.path.exists(TRADES_FILE):
                raise FileNotFoundError("Archivo no encontrado.")
            with open(TRADES_FILE, "r") as f:
                trades = json.load(f)
            if not trades:
                raise ValueError("Archivo vacio.")
            last = trades[-1]
            message = (
                f"📈 Ultima operacion:\n"
                f"- Accion: {last['action']}\n"
                f"- Precio: {last['price']}\n"
                f"- Cantidad: {last['quantity']}\n"
                f"- Motivo: {last['reason']}\n"
                f"- Fecha: {last['timestamp']}"
            )
        except Exception as e:
            logging.error(f"Error en /ultima_operacion: {e}")
            message = "❌ No se pudo leer la ultima operacion."
        send_message(message)
    
    elif command_text == "/reporte":
        try:
            if not os.path.exists(TRADES_FILE):
                raise FileNotFoundError("Archivo no encontrado.")
            with open(TRADES_FILE, "r") as f:
                trades = json.load(f)
            if not trades:
                raise ValueError("Archivo vacio.")
            total = len(trades)
            buys = sum(1 for t in trades if t["action"] == "BUY")
            sells = sum(1 for t in trades if t["action"] == "SELL")
            message = (
                f"📊 Resumen de operaciones:\n"
                f"- Total: {total}\n"
                f"- Compras: {buys}\n"
                f"- Ventas: {sells}"
            )
        except Exception as e:
            logging.error(f"Error en /reporte: {e}")
            message = "❌ No se pudo generar el reporte."
        send_message(message)
    else:
        send_message("🤷‍♂️ Comando no reconocido. Usa /estado, /pausar, /reanudar, /ultima_operacion o /reporte.")

def send_telegram_with_buttons():
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    message = "🧠 ¿Qué deseas hacer con el bot?"
    keyboard = {
        "inline_keyboard": [
            [{"text": "📊 Estado", "callback_data": "/estado"}],
            [{"text": "⏸️ Pausar", "callback_data": "/pausar"}],
            [{"text": "▶️ Reanudar", "callback_data": "/reanudar"}],
            [{"text": "📈 Última operación", "callback_data": "/ultima_operacion"}],
            [{"text": "📋 Reporte", "callback_data": "/reporte"}]
        ]
    }
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "reply_markup": json.dumps(keyboard)
    }
    try:
        safe_post(url, payload, context="send_telegram_with_buttons")
    except Exception as e:
        print(f"❌ Error al enviar botones: {e}")



if __name__ == "__main__":
    send_telegram_with_buttons() # Opcional: al iniciar, envia los botones

    while True:
        try:
            poll_telegram()
        except Exception as e:
            logging.error(f"Error en bucle principal: {e}")
        time.sleep(5)