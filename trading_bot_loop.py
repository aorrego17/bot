import time
import subprocess
import logging

INTERVALO_MINUTOS = 5  # Cambia a 1, 10, 30 según la frecuencia deseada

logging.basicConfig(
    filename='trading_bot_loop.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

while True:
    logging.info("⏳ Ejecutando trading_bot_final.py")
    try:
        subprocess.run(
            ["python3", "/home/morgan/Documents/trading-bot/trading_bot_final.py"],
            check=True
        )
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Error al ejecutar el bot: {e}")
    time.sleep(INTERVALO_MINUTOS * 60)