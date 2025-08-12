import json
import os

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

def main():
    config_file = input("Ruta de config.json (default: config.json): ").strip() or "config.json"
    if not os.path.exists(config_file):
        print(f"No encontrado: {config_file}")
        return
    with open(config_file, "r") as f:
        config = json.load(f)

    sname = input("Nombre de la nueva estrategia (ej: ADX): ").strip()
    tipos = ["tendencia", "reversion", "scalping", "otro"]
    tipo = select_option("Tipo de estrategia", tipos, default=1)
    # Añadir o actualizar entrada
    if "strategies_meta" not in config:
        config["strategies_meta"] = {}
    config["strategies_meta"][sname] = {"type": tipo}
    with open(config_file, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Estrategia {sname} de tipo {tipo} añadida a {config_file}")

if __name__ == "__main__":
    main()
