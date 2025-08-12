import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from trading_bot_final import (
    load_config, add_indicators,
    strategy_rsi_macd, strategy_ema_bb, strategy_atr_rsi, strategy_ml, load_or_train_model
)

def interactive_parse_args(config):
    def ask_param(prompt, default=None, options=None, example=None):
        ex = f" (ej: {example})" if example else ""
        opt_txt = f" Opciones: {options}" if options else ""
        val = input(f"{prompt}{opt_txt}{ex}{' ['+str(default)+']' if default else ''}: ").strip()
        if not val and default is not None:
            return default
        return val

    symbol = ask_param("Símbolo", example="BTCUSDT, ETHUSDT")
    interval = ask_param("Intervalo", default="1h", options=["1m", "5m", "15m", "1h", "4h", "1d"])
    session = ask_param("Sesión", default="None", options=list(SESSION_MAP.keys()) + ["None"])
    balance = float(ask_param("Balance inicial", default="1000"))
    risk_pct = float(ask_param("Porcentaje riesgo por operación", default="1"))
    engine = ask_param("Engine", default="pandas", options=["pandas", "polars"])
    available_strategies = list(config.get("strategies_meta", {}).keys())
    print(f"Estrategias disponibles: {' '.join(available_strategies)}")
    strategies_input = ask_param("Estrategias (separadas por espacio, enter para default)", example=' '.join(available_strategies))
    strategies = strategies_input.split() if strategies_input.strip() else None
    start = ask_param("Fecha inicio (YYYY-MM-DD, enter para todo)", default=None, example="2022-01-01")
    end = ask_param("Fecha fin (YYYY-MM-DD, enter para todo)", default=None, example="2023-01-01")
    session = session if session != "None" else None

    if not strategies or strategies == ['']:
        strategies = None
    return symbol, interval, session, balance, risk_pct, engine, strategies, start, end

def read_list_from_file(prompt):
    filepath = input(f"{prompt} (deja vacío para no usar lista de archivo): ").strip()
    if not filepath or not os.path.exists(filepath):
        return None
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def validate_csv_structure(df):
    required_cols = ["open", "high", "low", "close", "volume"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"El archivo CSV está incompleto. Faltan columnas: {missing_cols}")

# Loader universal con pandas/polars.
def load_historical_data(symbol, interval="1h", data_dir="data", engine="pandas", start=None, end=None):
    filename = os.path.join(data_dir, f"{symbol}_{interval}.csv")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Archivo no encontrado: {filename}")
    if engine == "polars":
        import polars as pl
        df = pl.read_csv(filename).to_pandas()
    else:
        df = pd.read_csv(filename)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    if start:
        df = df[df.index >= pd.to_datetime(start)]
    if end:
        df = df[df.index <= pd.to_datetime(end)]
    return df

# Filtro de sesión
SESSION_MAP = {
    "europea": (7, 15),
    "americana": (13, 22),
    "asiatica": (0, 8)
}
def filter_by_session(df, session):
    if session in SESSION_MAP:
        ini, fin = SESSION_MAP[session]
        return df[(df.index.hour >= ini) & (df.index.hour < fin)]
    return df

# Validación de 'type' de estrategias (homogeneidad para consenso)
def validate_strategy_types(strategy_names, config):
    tipos = set()
    for s in strategy_names:
        tipo = config.get("strategies_meta", {}).get(s, {}).get("type")
        if not tipo:
            raise Exception(f"Estrategia {s} no tiene tipo definido en config.")
        tipos.add(tipo)
    if len(tipos) != 1:
        raise Exception(f"Error: Estrategias de tipos distintos: {tipos}. No se permite operar consenso.")
    return list(tipos)[0]

# Risk dynamic sizing
def calculate_position_size(balance, price, min_qty, risk_pct):
    position_size = (balance * risk_pct / 100) / price
    if position_size < min_qty:
        min_risk = (min_qty * price / balance) * 100
        print(f"Riesgo insuficiente, se ajusta a {min_risk:.2f}% para cumplir mínimo del broker.")
        position_size = min_qty
    return position_size

# Enviar alerta backtesting a Telegram
def send_telegram_backtesting(message):
    import requests
    from dotenv import load_dotenv
    load_dotenv()
    TOKEN = os.getenv("TELEGRAM_TOKEN")
    CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    msg = f"[BACKTESTING] {message}"
    payload = {"chat_id": CHAT_ID, "text": msg}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"❌ (Backtesting) Telegram error: {e}")

# Simulación pro — usa numpy en cálculos donde aplique.
def simulate_backtest(df, strategies, model, market_config, session=None, initial_balance=1000.0):
    if session:
        df = filter_by_session(df, session)
    balance = initial_balance
    trades = []
    in_position = False
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    highest_price = 0
    min_qty = market_config.get("min_qty", 0.0001)
    risk_pct = market_config.get("risk_pct", 1)
    stop_loss_pct = market_config.get("stop_loss_pct", 5)
    take_profit_pct = market_config.get("take_profit_pct", 10)
    max_quantity = market_config.get("max_quantity", 0.01)
    trailing_stop_min_drop = market_config.get("trailing_stop_min_drop", 0.2)

    for i in range(len(df)):
        row = df.iloc[i]
        results = []
        for name, func in strategies.items():
            try:
                res = func(df.iloc[:i+1])
            except Exception as e:
                res = 0
            results.append((name, res))

        total_strat = len(results)
        buy_signals = sum(1 for _, r in results if r == 1)
        sell_signals = sum(1 for _, r in results if r == -1)
        buy_prob = buy_signals / total_strat
        sell_prob = sell_signals / total_strat
        final_signal = 1 if buy_prob >= 0.8 else -1 if sell_prob >= 0.8 else 0

        close = row["close"]; high = row["high"]; low = row["low"]

        # Trailing/TP/SL 
        if in_position:
            highest_price = np.maximum(highest_price, high)
            trailing_sl = highest_price * (1 - stop_loss_pct/100)
            exit = False; reason = ""
            if low <= stop_loss or low <= trailing_sl:
                pnl = (stop_loss - entry_price) / entry_price
                balance += balance * pnl
                exit = True; reason = "SL/TRAIL"
                send_telegram_backtesting(f"SL/Trailing ejecutado en {close:.2f}")
            elif high >= take_profit:
                pnl = (take_profit - entry_price) / entry_price
                balance += balance * pnl
                exit = True; reason = "TP"
                send_telegram_backtesting(f"Take Profit ejecutado en {close:.2f}")
            if exit:
                resultado, explicacion = classify_trade({"tipo": reason})
                trades.append({
                    "fecha_cierre": row.name,
                    "tipo": reason,
                    "precio_entry": entry_price,
                    "precio_exit": close,
                    "balance": balance,
                    "resultado": resultado,
                    "explicacion": explicacion
                })
                in_position = False

        if not in_position and final_signal == 1:
            qty = min(calculate_position_size(balance, close, min_qty, risk_pct), max_quantity)
            entry_price = close
            stop_loss = close * (1 - stop_loss_pct/100)
            take_profit = close * (1 + take_profit_pct/100)
            highest_price = close
            in_position = True
            trades.append({
                "fecha_entrada": row.name,
                "tipo": "BUY",
                "precio_entry": entry_price,
                "balance_entry": balance,
                "resultado": "Apertura",
                "explicacion": "Condiciones del consenso cumplidas (prob >= 80%)"
            })
            send_telegram_backtesting(f"COMPRA simulada en {entry_price:.2f}, SL: {stop_loss:.2f}, TP: {take_profit:.2f}")

    return {"trades": trades, "balance_final": balance, "trades_count": len(trades)}

def summarize_observations(trades):
    total_trades = sum(1 for t in trades if t.get("tipo") == "BUY")
    positivos = sum(1 for t in trades if t.get("resultado") == "Positivo")
    negativos = sum(1 for t in trades if t.get("resultado") == "Negativo")
    winrate = (positivos / total_trades)*100 if total_trades > 0 else 0
    obs = []
    obs.append(f"Total Trades: {total_trades}")
    obs.append(f"Positivos (TP): {positivos}")
    obs.append(f"Negativos (SL/Trailing): {negativos}")
    obs.append(f"Winrate: {winrate:.2f} %")
    if winrate >= 70:
        obs.append("Excelente performance, la mayoría de operaciones se cierran en positivo.")
    elif winrate >= 50:
        obs.append("Performance aceptable pero puede optimizarse con ajuste de SL/TP o reglas de consenso.")
    else:
        obs.append("Estrategia de riesgo, revisar señales y condiciones de consenso/risk.")
    return "\n".join(obs)

# CLI pro
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--session", default=None, choices=list(SESSION_MAP.keys()) + [None])
    parser.add_argument("--strategies", nargs="+", default=None)
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--risk_pct", type=float, default=1.0)
    parser.add_argument("--engine", default="pandas", choices=["pandas", "polars"])
    return parser.parse_args()

def classify_trade(trade):
    """
    Analiza el resultado individual del trade y explica el motivo de éxito/fallo.
    """
    if 'tipo' in trade:
        if trade['tipo'] == "TP":
            return "Positivo", "Take Profit alcanzado"
        elif trade['tipo'] == "SL/TRAIL":
            return "Negativo", "Stop Loss o Trailing Stop ejecutado"
        elif trade['tipo'] == "SELL":
            # Si agregas lógica de cierre manual o por consenso en tu sim constructor, puedes refinar
            return "Negativo", "Cierre manual/estrategia"
        else:
            return "Neutro", "-Sin motivo específico-"
    else:
        return "Neutro", "-Sin motivo específico-"

def main():
    config = load_config()
    batch_mode = False
    if len(sys.argv) == 1:
        batch_in = input("¿Deseas modo batch para múltiple símbolo y estrategias? (s/n) [n]: ").strip().lower()
        batch_mode = batch_in == "s"

    if batch_mode:
        symbols = read_list_from_file("Ruta archivo de símbolos (uno por línea)")
        if not symbols:
            symbols = input("Símbolos separados por espacio (ej: BTCUSDT ETHUSDT): ").strip().split()
        strategy_combos = read_list_from_file("Ruta archivo combinaciones estrategias (una por línea, separadas por espacio)")
        if not strategy_combos:
            available_strategies = list(config.get("strategies_meta", {}).keys())
            print(f"Estrategias disponibles: {' '.join(available_strategies)}")
            default_str = ' '.join(available_strategies)
            user_input = input(f"Estrategias (o combos) para todas las pruebas (separa con espacio) [{default_str}]: ").strip() or default_str
            strategy_combos = [user_input]

        interval = input("Intervalo para todos los símbolos (ej: 1h, 5m): ").strip() or "1h"
        balance = float(input("Balance inicial para cada backtest (ej: 1000): ").strip() or "1000")
        risk_pct = float(input("Porcentaje riesgo por operación (ej: 1): ").strip() or "1")
        engine = input("Engine (pandas/polars): ").strip() or "pandas"
        session = input(f"Sesión (europea/americana/asiatica/None): ").strip() or None
        start = input("Fecha inicio (YYYY-MM-DD, enter para todo): ").strip() or None
        end = input("Fecha fin (YYYY-MM-DD, enter para todo): ").strip() or None

        for symbol in symbols:
            for combo in strategy_combos:
                strat_names = combo.strip().split()
                try:
                    market = next(m for m in config['markets'] if m['symbol'] == symbol and m['interval'] == interval)
                except StopIteration:
                    print(f"❌ No se encontró configuración para {symbol} {interval}.")
                    continue
                try:
                    tipo = validate_strategy_types(strat_names, config)
                except Exception as e:
                    print(f"❌ {e}")
                    send_telegram_backtesting(f"Error en validación de tipo: {e}")
                    continue
                print(f"\n### Backtest Batch: {symbol} | Estrategias: {strat_names} (tipo: {tipo}) | TF: {interval} ###")
                try:
                    df = load_historical_data(
                        symbol,
                        interval,
                        engine=engine,
                        start=start,
                        end=end
                    )
                    validate_csv_structure(df)
                    df = add_indicators(df)
                    if session:
                        df = filter_by_session(df, session)
                except Exception as e:
                    print(f"❌ Error cargando datos CSV: {e}")
                    continue

                strategy_map = {
                    "RSI_MACD": strategy_rsi_macd,
                    "EMA_BB": strategy_ema_bb,
                    "ATR_RSI": strategy_atr_rsi
                }
                if "ML" in strat_names:
                    model = load_or_train_model(df)
                    strategy_map["ML"] = lambda dfi: strategy_ml(dfi, model)
                else:
                    model = None 
                strategies = {name: strategy_map[name] for name in strat_names if name in strategy_map}

                try:
                    result = simulate_backtest(
                        df,
                        strategies,
                        None,  # o model si solo usas ML
                        market,
                        session=session,
                        initial_balance=balance
                    )
                except Exception as e:
                    print(f"❌ Error durante el backtest de {symbol} {combo}: {e}")
                    continue

                report_file = f"report_{symbol}_{'-'.join(strat_names)}_{interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                pd.DataFrame(result['trades']).to_csv(report_file, index=False)
                print(f"🗂️ Reporte batch guardado en: {report_file}")

                global_obs = summarize_observations(result['trades'])
                with open(report_file.replace(".csv", "_obs.txt"), "w") as f:
                    f.write(global_obs)
                print(f"📝 Observaciones globales en: {report_file.replace('.csv', '_obs.txt')}")
                if "Excelente performance" in global_obs:
                    print("  🟢 Resultado FAVORABLE")
                elif "Performance aceptable" in global_obs:
                    print("  🟡 Resultado ACEPTABLE")
                else:
                    print("  🔴 Resultado NO FAVORABLE")
        print("\nBatch completado.")
        return  # IMPORTANTE: no ejecutar el pipeline “single” abajo

    # Pipeline SINGLE NORMAL
    if len(sys.argv) == 1:
        symbol, interval, session, balance, risk_pct, engine, strategies, start, end = interactive_parse_args(config)
        class Args: pass
        args = Args()
        args.symbol = symbol
        args.interval = interval
        args.session = session
        args.balance = balance
        args.risk_pct = risk_pct
        args.engine = engine
        args.strategies = strategies
        args.start = start
        args.end = end
    else:
        args = parse_args()

    try:
        market = next(m for m in config['markets'] if m['symbol'] == args.symbol and m['interval'] == args.interval)
    except StopIteration:
        print(f"❌ No se encontró configuración de mercado para {args.symbol} {args.interval}")
        return

    strategy_names = args.strategies or market['strategies']

    try:
        tipo = validate_strategy_types(strategy_names, config)
    except Exception as e:
        print(f"❌ {e}")
        send_telegram_backtesting(f"Error en validación de tipo: {e}")
        return

    print(f"\n### Parámetros seleccionados para backtesting ###")
    print(f"   Símbolo:   {args.symbol}")
    print(f"   Intervalo: {args.interval}")
    if args.start or args.end:
        print(f"   Periodo:   {args.start or '--'} - {args.end or '--'}")
    print(f"   Sesión:    {args.session if args.session else 'Todas'}")
    print(f"   Estrategias: {strategy_names} (tipo: {tipo})")
    print(f"   Balance inicial: {args.balance}")
    print(f"   Risk pct:  {args.risk_pct}")
    print(f"   Engine:    {args.engine}\n")

    try:
        df = load_historical_data(
            args.symbol,
            args.interval,
            engine=args.engine,
            start=args.start,
            end=args.end
        )
        validate_csv_structure(df)
    except Exception as e:
        print(f"❌ CSV inválido: {e}")
        send_telegram_backtesting(f"CSV inválido: {e}")
        return

    df = add_indicators(df)
    if args.session:
        df = filter_by_session(df, args.session)

    strategy_map = {
        "RSI_MACD": strategy_rsi_macd,
        "EMA_BB": strategy_ema_bb,
        "ATR_RSI": strategy_atr_rsi
    }
    model = None
    if "ML" in strategy_names:
        model = load_or_train_model(df)
        strategy_map["ML"] = lambda dfi: strategy_ml(dfi, model)
    strategies = {name: strategy_map[name] for name in strategy_names if name in strategy_map}

    result = simulate_backtest(
        df,
        strategies,
        model,
        market,
        session=args.session,
        initial_balance=args.balance
    )
    print(f"--- Resultados: {args.symbol} {args.interval} ---")
    print(f"Balance final: {result['balance_final']:.2f} USDT, Trades: {result['trades_count']}")

    report_file = f"report_{args.symbol}_{args.interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame(result['trades']).to_csv(report_file, index=False)
    print(f"\nReporte detallado guardado en {report_file}")

    global_obs = summarize_observations(result['trades'])
    print("\n--- Observaciones globales ---")
    print(global_obs)

    with open(report_file.replace(".csv", "_obs.txt"), "w") as f:
        f.write(global_obs)
    print(f"Observaciones globales escritas en {report_file.replace('.csv', '_obs.txt')}")

    if "Excelente performance" in global_obs:
        print("\n*** 🟢 El resultado es FAVORABLE. Estrategia/mix con alto winrate. 🟢 ***")
    elif "Performance aceptable" in global_obs:
        print("\n*** 🟡 Resultado ACEPTABLE. Podrías optimizar o revisar reglas. 🟡 ***")
    else:
        print("\n*** 🔴 El resultado NO ES FAVORABLE. Revisa señales y consenso. 🔴 ***")

    #print(f"\nArchivos generados:\n - Trades: {os.path.abspath(report_file)}\n - Observaciones: {os.path.abspath(report_file.replace('.csv', '_obs.txt'))}")
    print(f"\nTodos los reportes batch están en la carpeta: {os.path.abspath('data/')}")

if __name__ == "__main__":
    main()

