import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ruta al archivo resumen de resultados mensuales de la optimización
PATH_RESUMEN = "Etapa5/resultados_optimizacion_mensual.csv"

# Patrón de archivos con las series originales y optimizadas
PATH_SERIES = "Etapa5/consumo_optimizado_*.csv"

# Carpeta de salida para esta etapa
SALIDA_DIR = "Etapa6"

# Cantidad de meses representativos a graficar
N_TOP_MESES = 3


def cargar_resumen(path: str) -> pd.DataFrame:
    """Carga el archivo resumen con resultados mensuales y agrega columna de fechas."""
    df = pd.read_csv(path)
    if "mes_dt" not in df.columns:
        df["mes_dt"] = pd.to_datetime(df["mes"])
    df = df.sort_values("mes_dt").reset_index(drop=True)
    return df


def stats_globales(df: pd.DataFrame) -> dict:
    """Calcula estadísticas globales a partir del resumen mensual."""
    m = {}
    m["alpha_prom"] = df["alpha_opt"].mean()
    m["peak_red_prom"] = df["max_peak_reduction_%"].mean()
    m["max_red_prom"] = df["max_reduction_%"].mean()
    m["energy_diff_prom_%"] = df["energy_diff_%"].mean()

    # Porcentaje de meses con mejora y con empeoramiento
    m["meses_con_mejora_pico_%"] = 100 * (df["max_peak_reduction_%"] > 0).mean()
    m["meses_con_empeoramiento_global_%"] = 100 * (df["max_reduction_%"] < 0).mean()

    # Matriz de correlaciones entre métricas
    m["corr_matrix"] = df[["alpha_opt", "max_peak_reduction_%", "max_reduction_%", "energy_diff_%"]].corr()
    return m


def imprimir_stats(stats: dict):
    """Imprime estadísticas globales y correlaciones."""
    print("\n========== RESUMEN GLOBAL ==========")
    print(f"α* promedio:                        {stats['alpha_prom']:.4f}")
    print(f"Reducción pico promedio (%):        {stats['peak_red_prom']:.2f}")
    print(f"Reducción máx. global prom. (%):    {stats['max_red_prom']:.2f}")
    print(f"Diferencia energía prom. (%):       {stats['energy_diff_prom_%']:.4f}")
    print(f"% meses con mejora en pico:         {stats['meses_con_mejora_pico_%']:.1f}%")
    print(f"% meses con peor máximo global:     {stats['meses_con_empeoramiento_global_%']:.1f}%")
    print("\n--- Correlaciones ---")
    print(stats["corr_matrix"])


def plot_alpha(df: pd.DataFrame, outdir: str):
    """Grafica la evolución del valor óptimo α* por mes."""
    plt.figure(figsize=(12, 4))
    plt.plot(df["mes_dt"], df["alpha_opt"], marker="o")
    plt.title("α* por mes")
    plt.xlabel("Mes")
    plt.ylabel("α*")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "alpha_por_mes.png"), dpi=150)
    plt.show()


def plot_peak_reduction(df: pd.DataFrame, outdir: str):
    """Grafica la reducción de pico porcentual por mes."""
    plt.figure(figsize=(12, 4))
    plt.plot(df["mes_dt"], df["max_peak_reduction_%"], marker="o")
    plt.title("Reducción de pico (%) por mes")
    plt.xlabel("Mes")
    plt.ylabel("Reducción pico (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "reduccion_pico_por_mes.png"), dpi=150)
    plt.show()


def plot_energy_totals(df: pd.DataFrame, outdir: str):
    """Grafica energía total mensual original vs optimizada."""
    if {"energy_before", "energy_after"}.issubset(df.columns):
        plt.figure(figsize=(12, 4))
        plt.plot(df["mes_dt"], df["energy_before"], label="Energía original", marker="o")
        plt.plot(df["mes_dt"], df["energy_after"], label="Energía optimizada", marker="o", linestyle="--")
        plt.title("Energía total mensual (antes vs después)")
        plt.xlabel("Mes")
        plt.ylabel("Energía (unidades del dataset)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "energia_total_mensual.png"), dpi=150)
        plt.show()


def plot_boxplots(df: pd.DataFrame, outdir: str):
    """Grafica diagramas de caja para las métricas clave."""
    cols = ["max_peak_reduction_%", "max_reduction_%", "energy_diff_%"]
    df[cols].plot(kind="box", figsize=(8, 4), grid=True)
    plt.title("Distribución de métricas clave")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "boxplots_metricas.png"), dpi=150)
    plt.show()


def plot_mes_representativo(series_dir_pattern: str, df_resumen: pd.DataFrame, outdir: str, n_top: int = 3):
    """Grafica las series originales y optimizadas de los meses con mayor reducción de pico."""
    top = df_resumen.sort_values("max_peak_reduction_%", ascending=False).head(n_top)["mes"].tolist()
    for mes in top:
        pattern = os.path.join(os.path.dirname(series_dir_pattern), f"consumo_optimizado_{mes}.csv")
        files = glob.glob(pattern)
        if not files:
            print(f"[AVISO] No encontré series para el mes {mes} en {pattern}")
            continue
        df_mes = pd.read_csv(files[0], parse_dates=True, index_col=0)
        df_mes.index = pd.to_datetime(df_mes.index)

        plt.figure(figsize=(12, 4))
        plt.plot(df_mes.index, df_mes["original"], label="Original")
        plt.plot(df_mes.index, df_mes["optimizado"], label="Optimizado", linestyle="--")
        plt.title(f"Mes representativo {mes} (Top reducción pico)")
        plt.xlabel("Tiempo")
        plt.ylabel("Consumo")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"mes_{mes}_comparacion.png"), dpi=150)
        plt.show()


# === EJECUCIÓN PRINCIPAL ===
if __name__ == "__main__":
    os.makedirs(SALIDA_DIR, exist_ok=True)

    # Cargar resumen de resultados mensuales
    df = cargar_resumen(PATH_RESUMEN)

    # Calcular estadísticas globales
    stats = stats_globales(df)
    imprimir_stats(stats)

    # Gráficas generales
    plot_alpha(df, SALIDA_DIR)
    plot_peak_reduction(df, SALIDA_DIR)
    plot_energy_totals(df, SALIDA_DIR)
    plot_boxplots(df, SALIDA_DIR)

    # Graficar meses más representativos (mayor reducción de picos)
    plot_mes_representativo(PATH_SERIES, df, SALIDA_DIR, n_top=N_TOP_MESES)

    # Guardar correlaciones y resumen
    correl_out = os.path.join(SALIDA_DIR, "correlaciones.csv")
    stats["corr_matrix"].to_csv(correl_out)

    resumen_out = os.path.join(SALIDA_DIR, "resumen_global.txt")
    with open(resumen_out, "w") as f:
        f.write("===== RESUMEN GLOBAL =====\n")
        f.write(f"alpha_prom               : {stats['alpha_prom']:.6f}\n")
        f.write(f"max_peak_reduction_% mean: {stats['peak_red_prom']:.6f}\n")
        f.write(f"max_reduction_% mean     : {stats['max_red_prom']:.6f}\n")
        f.write(f"energy_diff_% mean       : {stats['energy_diff_prom_%']:.6f}\n")
        f.write(f"% meses mejora pico      : {stats['meses_con_mejora_pico_%']:.2f}%\n")
        f.write(f"% meses peor max global  : {stats['meses_con_empeoramiento_global_%']:.2f}%\n")

    print("\nArchivos de la Etapa 6 guardados en:", SALIDA_DIR)
