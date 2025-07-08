# ETAPA 2 – ANÁLISIS DEL CONSUMO ELÉCTRICO (USANDO ARCHIVOS DIVIDIDOS)
# --------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Cargar y unir los archivos .txt
partes = []
for i in range(1, 11):
    archivo = f"datos_limpios_parte_{i}.txt"
    df_parte = pd.read_csv(archivo, sep=';')
    partes.append(df_parte)

df = pd.concat(partes, ignore_index=True)

# 2. Procesar columna Datetime y convertir a índice
df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)

# 3. Convertir columnas a numérico (en caso de errores)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 4. Estadísticas Descriptivas
print("Estadísticas descriptivas:")
print(df.describe()[['Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']])

# 5. Visualización del Consumo Global en el tiempo
plt.figure(figsize=(15, 5))
df['Global_active_power'].plot()
plt.title('Consumo Global Activo (kW)')
plt.ylabel('kW')
plt.xlabel('Fecha y Hora')
plt.grid(True)
plt.show()




