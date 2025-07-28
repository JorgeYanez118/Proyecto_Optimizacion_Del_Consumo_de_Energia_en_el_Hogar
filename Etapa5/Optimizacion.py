import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

RANGO_PARTES = range(1, 21)
PATH_ARCHIVOS = "Datos/datos_limpios_parte_{i}.txt"

DATETIME_COL = "Datetime"
CONSUMO_COL  = "Global_active_power"
AGG   = "mean"   
FREQ  = "h"

N_PEAK = 5

LAMBDA_MIN, LAMBDA_MAX = 0.5, 5.0
MU_INIT = 1e-3
MU_MIN  = 1e-7

USE_NEWTON_RAPHSON = True
USE_FIXED_POINT    = True

GUARDAR_SERIES_X_MES = True
GRAFICAR_X_MES       = False
SALIDA_DIR = "Etapa5"

def cargar_partes(path_pattern: str, r_parts: range) -> pd.DataFrame:
    partes = []
    for i in r_parts:
        path = path_pattern.format(i=i)
        if os.path.exists(path):
            df_p = pd.read_csv(path, sep=';')
            partes.append(df_p)
        else:
            print(f"[AVISO] No existe: {path}")
    if not partes:
        raise FileNotFoundError("No se encontraron archivos de datos limpios.")
    return pd.concat(partes, ignore_index=True)


def preparar_series_horarias(df: pd.DataFrame,
                             datetime_col: str,
                             value_col: str,
                             agg: str = "mean",
                             freq: str = "H") -> pd.Series:
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.set_index(datetime_col).sort_index()
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce').interpolate('linear')

    if agg == "sum":
        x = df[value_col].resample(freq).sum()
    else:
        x = df[value_col].resample(freq).mean()

    return x.dropna()


def detectar_horas_pico(x_hourly: pd.Series, n_peak: int = 5) -> List[int]:
    consumo_hora = x_hourly.groupby(x_hourly.index.hour).mean()
    return consumo_hora.sort_values(ascending=False).head(n_peak).index.tolist()


def split_peak_offpeak(x: pd.Series, peak_hours: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    hours = x.index.hour
    mask_peak = np.isin(hours, peak_hours)
    return mask_peak, ~mask_peak


def estimar_lambda(x_hourly: pd.Series, peak_hours: List[int],
                   lam_min: float = 0.5, lam_max: float = 5.0) -> float:
    mask_peak, mask_off = split_peak_offpeak(x_hourly, peak_hours)
    mean_peak = x_hourly[mask_peak].mean()
    mean_off  = x_hourly[mask_off].mean()
    if mean_off <= 0 or np.isnan(mean_off):
        lam = 1.0
    else:
        lam = mean_peak / mean_off
    return float(np.clip(lam, lam_min, lam_max))


def objective(alpha: float, x: np.ndarray, P: np.ndarray, O: np.ndarray, lam: float = 1.0) -> float:
    xP = x[P]
    xO = x[O]
    S1_P = xP.sum()
    nO = xO.size
    return (((1 - alpha) * xP) ** 2).sum() + lam * (((xO + alpha * S1_P / nO) ** 2).sum())


def derivatives(alpha: float, x: np.ndarray, P: np.ndarray, O: np.ndarray, lam: float = 1.0) -> Tuple[float, float]:
    xP = x[P]
    xO = x[O]
    S2_P = np.sum(xP ** 2)
    S1_P = np.sum(xP)
    S1_O = np.sum(xO)
    nO = xO.size
    fprime  = -2 * (1 - alpha) * S2_P + 2 * lam * (S1_P * S1_O / nO + alpha * S1_P**2 / nO)
    fsecond =  2 * S2_P + 2 * lam * (S1_P**2 / nO)
    return fprime, fsecond


def newton_raphson_alpha(x: np.ndarray, P: np.ndarray, O: np.ndarray, lam: float = 1.0,
                         alpha0: float = 0.2, tol: float = 1e-8, max_iter: int = 50,
                         clip: Tuple[float, float] = (0.0, 1.0)):
    alpha = alpha0
    for k in range(max_iter):
        fprime, fsecond = derivatives(alpha, x, P, O, lam)
        step = fprime / (fsecond + 1e-12)
        new_alpha = np.clip(alpha - step, clip[0], clip[1])
        if abs(new_alpha - alpha) < tol:
            return new_alpha, k + 1, objective(new_alpha, x, P, O, lam)
        alpha = new_alpha
    return alpha, max_iter, objective(alpha, x, P, O, lam)


def fixed_point_alpha_adaptive(x: np.ndarray, P: np.ndarray, O: np.ndarray, lam: float = 1.0,
                               alpha0: float = 0.2, mu0: float = 1e-3, tol: float = 1e-8,
                               max_iter: int = 5000, clip: Tuple[float, float] = (0.0, 1.0),
                               mu_min: float = 1e-7, backtrack: float = 0.5):
    alpha = alpha0
    mu = mu0
    f_prev = objective(alpha, x, P, O, lam)

    for k in range(max_iter):
        fprime, _ = derivatives(alpha, x, P, O, lam)
        trial_alpha = np.clip(alpha - mu * fprime, clip[0], clip[1])
        f_trial = objective(trial_alpha, x, P, O, lam)

        if f_trial <= f_prev:  # mejora
            if abs(trial_alpha - alpha) < tol:
                return trial_alpha, k + 1, f_trial, mu
            alpha, f_prev = trial_alpha, f_trial
        else:
            # backtracking
            mu *= backtrack
            if mu < mu_min:
                return alpha, k + 1, f_prev, mu

    return alpha, max_iter, f_prev, mu


def apply_alpha(x: pd.Series, alpha: float, mask_peak: np.ndarray) -> pd.Series:
    x_arr = x.values.copy()
    xP = x_arr[mask_peak]
    xO_mask = ~mask_peak
    xO = x_arr[xO_mask]

    S1_P = xP.sum()
    nO = xO.size

    x_arr[mask_peak] = (1 - alpha) * xP
    x_arr[xO_mask] = xO + alpha * S1_P / nO

    return pd.Series(x_arr, index=x.index, name="optimizado")

def metrics_before_after(x: pd.Series, x_opt: pd.Series, mask_peak: np.ndarray) -> Dict[str, float]:
    max_before = x.max()
    max_after  = x_opt.max()

    max_peak_before = x[mask_peak].max() if mask_peak.any() else np.nan
    max_peak_after  = x_opt[mask_peak].max() if mask_peak.any() else np.nan

    energy_before = x.sum()
    energy_after  = x_opt.sum()

    return {
        "max_before": max_before,
        "max_after": max_after,
        "max_reduction_%": 100 * (max_before - max_after) / max_before if max_before > 0 else 0.0,
        "max_peak_before": max_peak_before,
        "max_peak_after": max_peak_after,
        "max_peak_reduction_%": (
            100 * (max_peak_before - max_peak_after) / max_peak_before
            if (pd.notna(max_peak_before) and max_peak_before > 0) else 0.0
        ),
        "energy_before": energy_before,
        "energy_after": energy_after,
        "energy_diff_%": 100 * (energy_after - energy_before) / energy_before if energy_before > 0 else 0.0
    }


def plot_before_after(x: pd.Series, x_opt: pd.Series, title_extra=""):
    plt.figure(figsize=(14, 4))
    plt.plot(x.index, x.values, label="Original")
    plt.plot(x_opt.index, x_opt.values, label="Optimizado", linestyle="--")
    plt.title(f"Consumo horario: original vs optimizado {title_extra}")
    plt.xlabel("Tiempo")
    plt.ylabel("Consumo")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    os.makedirs(SALIDA_DIR, exist_ok=True)

    df_raw = cargar_partes(PATH_ARCHIVOS, RANGO_PARTES)

  
    x_hourly = preparar_series_horarias(df_raw, DATETIME_COL, CONSUMO_COL, agg=AGG, freq=FREQ)


    PEAK_HOURS = detectar_horas_pico(x_hourly, N_PEAK)
    LAMBDA = estimar_lambda(x_hourly, PEAK_HOURS, LAMBDA_MIN, LAMBDA_MAX)
    print("Parámetros detectados:")
    print("PEAK_HOURS:", PEAK_HOURS)
    print("LAMBDA    :", LAMBDA)

    meses = x_hourly.index.to_period('M').unique()
    resultados = []

    for mes in meses:
        x_mes = x_hourly[x_hourly.index.to_period('M') == mes]
        if x_mes.empty:
            continue

        mask_peak, mask_off = split_peak_offpeak(x_mes, PEAK_HOURS)
        x_arr = x_mes.values.astype(float)

        res_dict = {}

        if USE_NEWTON_RAPHSON:
            alpha_nr, it_nr, f_nr = newton_raphson_alpha(x_arr, mask_peak, mask_off,
                                                         lam=LAMBDA, alpha0=0.2)
            res_dict["Newton-Raphson"] = (alpha_nr, it_nr, f_nr)

        if USE_FIXED_POINT:
            alpha_fp, it_fp, f_fp, mu_used = fixed_point_alpha_adaptive(
                x_arr, mask_peak, mask_off, lam=LAMBDA, alpha0=0.2,
                mu0=MU_INIT, mu_min=MU_MIN
            )
            res_dict["Punto Fijo"] = (alpha_fp, it_fp, f_fp, mu_used)

        # escoger mejor
        best_name = min(res_dict, key=lambda k: res_dict[k][2])
        best_alpha, best_iters, best_f = res_dict[best_name][:3]

        # aplicar
        x_opt = apply_alpha(x_mes, best_alpha, mask_peak)

        # métricas
        mets = metrics_before_after(x_mes, x_opt, mask_peak)

        # exportar series
        if GUARDAR_SERIES_X_MES:
            out_csv_mes = os.path.join(SALIDA_DIR, f"consumo_optimizado_{mes}.csv")
            pd.DataFrame({"original": x_mes, "optimizado": x_opt}).to_csv(out_csv_mes)

        if GRAFICAR_X_MES:
            plot_before_after(x_mes, x_opt, title_extra=f"{mes} (α*={best_alpha:.3f}, {best_name})")

        registro = {
            "mes": str(mes),
            "metodo": best_name,
            "alpha_opt": best_alpha,
            "iters": best_iters,
            "f_obj": best_f,
            "lambda": LAMBDA,
        }
        if USE_FIXED_POINT:
            registro["mu_usado"] = res_dict["Punto Fijo"][3] if "Punto Fijo" in res_dict else np.nan

        registro.update(mets)
        resultados.append(registro)

        print(f"[{mes}] método={best_name} α*={best_alpha:.5f} iters={best_iters} f={best_f:.4f} "
              f"max_reduc%={mets['max_reduction_%']:.2f} max_peak_reduc%={mets['max_peak_reduction_%']:.2f}")

 
    df_res = pd.DataFrame(resultados)
    out_csv = os.path.join(SALIDA_DIR, "resultados_optimizacion_mensual.csv")
    df_res.to_csv(out_csv, index=False)
    print("\nResumen mensual guardado en:", out_csv)

    try:
        df_res['mes_dt'] = pd.to_datetime(df_res['mes'])
        df_res = df_res.sort_values('mes_dt')

        plt.figure(figsize=(12, 4))
        plt.plot(df_res['mes_dt'], df_res['alpha_opt'], marker='o')
        plt.title('α* por mes')
        plt.xlabel('Mes')
        plt.ylabel('α*')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 4))
        plt.plot(df_res['mes_dt'], df_res['max_peak_reduction_%'], marker='o')
        plt.title('Reducción de pico (%) por mes')
        plt.xlabel('Mes')
        plt.ylabel('Reducción pico (%)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("No se pudieron generar los gráficos agregados:", e)
