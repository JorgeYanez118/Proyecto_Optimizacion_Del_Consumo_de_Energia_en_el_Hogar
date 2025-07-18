import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import os

# Unir los archivos de datos limpios en uno solo
partes = []
for i in range(1, 11):
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

# Prueba de comportamiento mensual
mes = '2007-01'
df_mes = df.loc[mes].copy()
df_mes = df_mes.dropna(subset=['Energia_no_submedida'])

df_mes['ts'] = (df_mes.index - df_mes.index[0]).total_seconds()/3600

X = df_mes[['ts']].values
y = df_mes['Energia_no_submedida'].values

grado = 3
poly = PolynomialFeatures(degree=grado)
X_poly = poly.fit_transform(X)
modelo = LinearRegression()
modelo.fit(X_poly, y)
y_pred = modelo.predict(X_poly)

# Graficar resultados
plt.figure(figsize=(12, 5))
plt.scatter(X, y, label='Datos reales', s=5)
plt.plot(X, y_pred, color='red', label=f'Polinomio grado {grado}')
plt.xlabel('Horas desde el inicio del mes')
plt.ylabel('Energía no submedida (Wh/min)')
plt.title(f'Regresión Polinómica sobre la Energía no Submedida ({mes})')
plt.legend()
plt.grid(True)
plt.show()


