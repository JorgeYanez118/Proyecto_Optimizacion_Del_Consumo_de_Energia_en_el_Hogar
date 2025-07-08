# Etapa 2: Análisis y Visualización del Consumo de Energía
# ---------------------------------------------------------
# Este análisis se basa en archivos .txt con datos limpios divididos por partes.
# El objetivo es explorar visualmente y estadísticamente el comportamiento del consumo eléctrico.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuración general para las gráficas
sns.set(style='whitegrid')
plt.rcParams.update({'figure.max_open_warning': 0})

# 1. Cargar todos los archivos de datos y unirlos en un solo DataFrame
partes = []
for i in range(1, 11):
    nombre_archivo = f"datos_limpios_parte_{i}.txt"
    if os.path.exists(nombre_archivo):
        df_parte = pd.read_csv(nombre_archivo, sep=';')
        partes.append(df_parte)
    else:
        print(f"Archivo no encontrado: {nombre_archivo}")

# Unimos todas las partes
df = pd.concat(partes, ignore_index=True)

# 2. Asegurarnos de que la columna 'Datetime' sea de tipo datetime y usarla como índice
df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)

# 3. Convertimos todas las columnas numéricas en caso de que haya errores por caracteres
for columna in df.columns:
    df[columna] = pd.to_numeric(df[columna], errors='coerce')

# 4. Estadísticas básicas del consumo
print("Resumen estadístico de los datos de consumo:")
print(df.describe()[['Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']])

# 5. Evolución del consumo global a lo largo del tiempo
plt.figure(figsize=(15, 4))
df['Global_active_power'].plot()
plt.title('Evolución del consumo global activo')
plt.ylabel('Consumo (kW)')
plt.xlabel('Fecha y hora')
plt.grid(True)

# 6. Histogramas de cada variable relevante
fig_hist, axs = plt.subplots(2, 2, figsize=(12, 8))
df['Global_active_power'].hist(ax=axs[0, 0], bins=50, color='skyblue')
axs[0, 0].set_title('Global Active Power')

df['Sub_metering_1'].hist(ax=axs[0, 1], bins=50, color='lightgreen')
axs[0, 1].set_title('Sub_metering_1')

df['Sub_metering_2'].hist(ax=axs[1, 0], bins=50, color='salmon')
axs[1, 0].set_title('Sub_metering_2')

df['Sub_metering_3'].hist(ax=axs[1, 1], bins=50, color='plum')
axs[1, 1].set_title('Sub_metering_3')

fig_hist.suptitle('Distribución de variables de consumo')
fig_hist.tight_layout()

# 7. Boxplots para ver valores atípicos y rangos
plt.figure(figsize=(10, 5))
df[['Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].plot(kind='box')
plt.title('Comparación de submedidores (boxplot)')
plt.grid(True)

# 8. Mapa de calor: consumo promedio por hora según el día de la semana
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek  # 0 = lunes, 6 = domingo

pivot_table = df.pivot_table(index='dayofweek', columns='hour', values='Global_active_power', aggfunc='mean')

plt.figure(figsize=(12, 6))
sns.heatmap(pivot_table, cmap='YlGnBu', linewidths=0.5)
plt.title("Mapa de calor: Consumo promedio por hora y día")
plt.xlabel("Hora del día")
plt.ylabel("Día de la semana (0 = Lunes)")

# 9. Matriz de correlación entre variables
plt.figure(figsize=(8, 6))
corr = df[['Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlación entre variables de consumo")

# 10. Top 10 momentos de mayor consumo promedio por hora
top_10 = df['Global_active_power'].resample('H').mean().sort_values(ascending=False).head(10)

print("\nTop 10 horas con mayor consumo promedio:")
print(top_10)

plt.figure(figsize=(10, 5))
top_10.plot(kind='bar', color='orange')
plt.title('Top 10 horas con mayor consumo promedio')
plt.ylabel('Consumo promedio (kW)')
plt.xlabel('Fecha y hora')
plt.grid(True)

# Mostrar todas las gráficas al final
plt.show()





