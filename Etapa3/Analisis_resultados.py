# Etapa 3: Visualización de Resultados del Modelado Polinómico
# -------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import os

# Configuración de rutas 
PATH_RESULTADOS = 'Etapa3/resultados_modelos_mensuales.csv'
DIR_SALIDA = 'Etapa3/Figuras'
os.makedirs(DIR_SALIDA, exist_ok=True)

# Cargar resultados 
df = pd.read_csv(PATH_RESULTADOS)
df['mes'] = pd.to_datetime(df['mes'])
df = df.sort_values('mes')

# Gráfico 1: Evolución de coeficientes polinómicos
plt.figure(figsize=(12, 6))
plt.plot(df['mes'], df['coef_x1'], label='Coef. x1 (lineal)')
plt.plot(df['mes'], df['coef_x2'], label='Coef. x2 (cuadrático)')
plt.plot(df['mes'], df['coef_x3'], label='Coef. x3 (cúbico)')
plt.title('Evolución de los coeficientes del modelo polinómico por mes')
plt.xlabel('Mes')
plt.ylabel('Valor del coeficiente')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(DIR_SALIDA, 'coeficientes_polinomio.png'), dpi=150)
plt.show()

# Gráfico 2: Error cuadrático medio por mes
plt.figure(figsize=(10, 4))
plt.plot(df['mes'], df['mse'], marker='o', color='purple')
plt.title('Error cuadrático medio (MSE) por mes')
plt.xlabel('Mes')
plt.ylabel('MSE')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(DIR_SALIDA, 'mse_por_mes.png'), dpi=150)
plt.show()

# Gráfico 3: Cantidad de picos positivos y negativos por mes
plt.figure(figsize=(10, 4))
plt.plot(df['mes'], df['num_picos_positivos'], label='Picos positivos', color='green')
plt.plot(df['mes'], df['num_picos_negativos'], label='Picos negativos', color='red')
plt.title('Cantidad de picos (derivada primera) por mes')
plt.xlabel('Mes')
plt.ylabel('Número de picos detectados')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(DIR_SALIDA, 'picos_detectados.png'), dpi=150)
plt.show()
