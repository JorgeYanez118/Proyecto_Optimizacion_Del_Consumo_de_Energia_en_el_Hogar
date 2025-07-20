import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import os
from dateutil.rrule import rrule, MONTHLY
from datetime import datetime

# Unir los archivos de datos limpios en uno solo
partes = []
for i in range(1, 21):
    nombre_archivo = f"Datos/datos_limpios_parte_{i}.txt"
    if os.path.exists(nombre_archivo):
        df_parte = pd.read_csv(nombre_archivo, sep=';')
        partes.append(df_parte)
    else:
        print(f"Archivo no encontrado: {nombre_archivo}")
        
df = pd.concat(partes, ignore_index=True)

for col in ['Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    
# Calcular energía activa no submedida (Wh por minuto)
df['Energia_no_submedida'] = (df['Global_active_power'] * 1000 / 60) - df['Sub_metering_1'] - df['Sub_metering_2'] - df['Sub_metering_3']

df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)

rango_meses = list(rrule(freq=MONTHLY, dtstart=datetime(2006, 12, 16), until=datetime(2010, 11, 26)))

for fecha in rango_meses:
    mes_str = fecha.strftime('%Y-%m')
    print(f"Analizando mes: {mes_str}")
    
    try:
        df_mes = df.loc[mes_str].copy()
    except KeyError:
        print(f" → No hay datos para {mes_str}")
        continue

    df_mes = df_mes.dropna(subset=['Energia_no_submedida'])
    if df_mes.empty:
        print(f" → Datos vacíos para {mes_str}")
        continue

    # Calcular variable de tiempo desde inicio del mes
    df_mes['ts'] = (df_mes.index - df_mes.index[0]).total_seconds() / 3600  # en horas

    # Variables para la regresión
    X = df_mes[['ts']].values
    y = df_mes['Energia_no_submedida'].values

    # Ajuste polinómico
    grado = 3
    poly = PolynomialFeatures(degree=grado)
    X_poly = poly.fit_transform(X)
    modelo = LinearRegression()
    modelo.fit(X_poly, y)
    y_pred = modelo.predict(X_poly)

    # Graficar ajuste
    plt.figure(figsize=(10, 4))
    plt.scatter(X, y, s=5, label='Datos reales', alpha=0.5)
    plt.plot(X, y_pred, color='red', label='Ajuste polinómico')
    plt.title(f'{mes_str} - Energía no submedida')
    plt.xlabel('Horas desde el inicio del mes')
    plt.ylabel('Energía no submedida (Wh/min)')
    plt.grid(True)
    plt.legend()
    plt.show()