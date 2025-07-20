import pandas as pd
import matplotlib.pyplot as plt

df_resultados = pd.read_csv('Etapa3/resultados_modelos_mensuales.csv')

df_resultados['mes'] = pd.to_datetime(df_resultados['mes'])
df_resultados = df_resultados.sort_values('mes')

# Coeficientes
plt.figure(figsize=(12, 6))
plt.plot(df_resultados['mes'], df_resultados['coef_x1'], label='Coef. x1 (lineal)')
plt.plot(df_resultados['mes'], df_resultados['coef_x2'], label='Coef. x2 (cuadrático)')
plt.plot(df_resultados['mes'], df_resultados['coef_x3'], label='Coef. x3 (cúbico)')
plt.title('Evolución de los coeficientes del modelo polinómico por mes')
plt.xlabel('Mes')
plt.ylabel('Valor del coeficiente')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# MSE
plt.figure(figsize=(10, 4))
plt.plot(df_resultados['mes'], df_resultados['mse'], marker='o')
plt.title('Error cuadrático medio (MSE) por mes')
plt.xlabel('Mes')
plt.ylabel('MSE')
plt.grid(True)
plt.tight_layout()
plt.show()

# Picos
plt.figure(figsize=(10, 4))
plt.plot(df_resultados['mes'], df_resultados['num_picos_positivos'], label='Picos positivos', color='green')
plt.plot(df_resultados['mes'], df_resultados['num_picos_negativos'], label='Picos negativos', color='red')
plt.title('Cantidad de picos (derivada 1ª) por mes')
plt.xlabel('Mes')
plt.ylabel('Número de picos detectados')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
