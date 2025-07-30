import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns

# Cargar archivo con resultados mensuales de modelos de regresión
df_resultados = pd.read_csv('Etapa3/resultados_modelos_mensuales.csv')

# Columnas a usar para el clustering
columnas_clustering = [
    'coef_x1', 'coef_x2', 'coef_x3',
    'mse', 'num_picos_positivos', 'num_picos_negativos'
]

# Seleccionar datos y escalar
df_cluster = df_resultados[columnas_clustering].copy()
scaler = StandardScaler()
datos_escalados = scaler.fit_transform(df_cluster)

# Crear nuevo DataFrame con datos escalados
df_clustering = pd.DataFrame(datos_escalados, columns=columnas_clustering)
df_clustering['mes'] = df_resultados['mes']

# BÚSQUEDA DEL NÚMERO ÓPTIMO DE CLUSTERS (k) 

inertias = []       # Inercia para método del codo
silhouettes = []    # Coeficiente de silueta
Krange = range(2, 10)

for k in Krange:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    etiquetas = kmeans.fit_predict(df_clustering.drop('mes', axis=1))
    
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(df_clustering.drop('mes', axis=1), etiquetas))

# Graficar resultados
plt.figure(figsize=(10, 4))
plt.plot(Krange, inertias, marker='o')
plt.title('Inercia vs Número de Clusters')
plt.xlabel('Número de Clusters')
plt.ylabel('Inercia')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(Krange, silhouettes, marker='o', color='green')
plt.title('Coeficiente de silueta vs Número de Clusters')
plt.xlabel('Número de Clusters')
plt.ylabel('Coeficiente de Silueta')
plt.grid(True)
plt.show()

# AGRUPAMIENTO CON K-MEANS 
# k = 4

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(df_clustering.drop('mes', axis=1))
df_clustering['cluster'] = cluster_labels

# Promedio de características por cluster
cluster_patterns = df_clustering.groupby('cluster').mean(numeric_only=True)

# Gráfico de barras de los promedios
cluster_patterns.T.plot(kind='bar', figsize=(12, 6))
plt.title('Media de características por cluster')
plt.xlabel('Variable')
plt.ylabel('Valor promedio (escalado)')
plt.grid(True)
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()

# Mapa de calor de los promedios
plt.figure(figsize=(10, 6))
sns.heatmap(cluster_patterns, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Mapa de calor: características promedio por cluster')
plt.xlabel('Variables')
plt.ylabel('Cluster')
plt.tight_layout()
plt.show()

# ANÁLISIS DE DISTRIBUCIÓN

# Conteo de meses por cluster
conteo_clusters = df_clustering.groupby('cluster')['mes'].count()
print(conteo_clusters)

plt.figure(figsize=(8, 5))
conteo_clusters.plot(kind='bar', color='skyblue')
plt.title('Cantidad de meses en cada cluster')
plt.xlabel('Cluster')
plt.ylabel('Número de meses')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Mostrar asignación de meses a clusters
meses_por_cluster = df_clustering[['mes', 'cluster']].sort_values('cluster')
print(meses_por_cluster.to_string(index=False))
print(cluster_patterns)

# CÁLCULO DE POTENCIAL DE MEJORA

# Agregar cluster al DataFrame original
df_resultados = pd.read_csv('Etapa3/resultados_modelos_mensuales.csv')
df_resultados['cluster'] = df_clustering['cluster']

# Tomar cluster 1 como referencia de eficiencia
centroide_eficiente = cluster_patterns.loc[1]

# Inicializar columnas para deltas
df_resultados['delta_mse'] = 0.0
df_resultados['delta_picos_pos'] = 0.0
df_resultados['delta_picos_neg'] = 0.0

# Calcular diferencias si no pertenece al cluster eficiente
for idx, row in df_resultados.iterrows():
    if row['cluster'] != 1:
        df_resultados.at[idx, 'delta_mse'] = row['mse'] - centroide_eficiente['mse']
        df_resultados.at[idx, 'delta_picos_pos'] = row['num_picos_positivos'] - centroide_eficiente['num_picos_positivos']
        df_resultados.at[idx, 'delta_picos_neg'] = row['num_picos_negativos'] - centroide_eficiente['num_picos_negativos']

# Totales de mejora potencial
mejora_mse = df_resultados['delta_mse'].sum()
mejora_picos_pos = df_resultados['delta_picos_pos'].sum()
mejora_picos_neg = df_resultados['delta_picos_neg'].sum()

print(f"Mejora total en MSE: {mejora_mse: .4f}")
print(f"Mejora total en Picos Positivos: {mejora_picos_pos: .4f}")
print(f"Mejora total en Picos Negativos: {mejora_picos_neg: .4f}")

# Totales originales
total_mse_original = df_resultados['mse'].sum()
total_picos_pos_original = df_resultados['num_picos_positivos'].sum()
total_picos_neg_original = df_resultados['num_picos_negativos'].sum()

# Porcentaje de mejora
porcentaje_mejora_mse = (mejora_mse / total_mse_original) * 100
porcentaje_mejora_pos = (mejora_picos_pos / total_picos_pos_original) * 100
porcentaje_mejora_neg = (mejora_picos_neg / total_picos_neg_original) * 100

print("Valores originales:")
print(f"MSE total original: {total_mse_original:.2f}")
print(f"Picos positivos totales: {total_picos_pos_original:.2f}")
print(f"Picos negativos totales: {total_picos_neg_original:.2f}")

print("\nPorcentajes de mejora:")
print(f"MSE: {porcentaje_mejora_mse:.2f}%")
print(f"Picos positivos: {porcentaje_mejora_pos:.2f}%")
print(f"Picos negativos: {porcentaje_mejora_neg:.2f}%")

# VISUALIZACIÓN DE MEJORAS

labels = ['MSE', 'Picos Positivos', 'Picos Negativos']
totales = [total_mse_original, total_picos_pos_original, total_picos_neg_original]
mejoras = [mejora_mse, mejora_picos_pos, mejora_picos_neg]
x = range(len(labels))
ancho = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x, totales, width=ancho, label='Total Original')
plt.bar([i + ancho for i in x], mejoras, width=ancho, label='Mejora Potencial')
plt.xticks([i + ancho / 2 for i in x], labels)
plt.ylabel('Valor')
plt.title('Comparación entre Totales Originales y Mejoras Potenciales')
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# Guardar DataFrame con cluster actualizado
df_resultados.to_csv('Etapa3/resultados_modelos_mensuales.csv', index=False)
