import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns

df_resultados = pd.read_csv('Etapa3/resultados_modelos_mensuales.csv')

columnas_clustering = [
    'coef_x1', 'coef_x2', 'coef_x3',
    'mse', 'num_picos_positivos', 'num_picos_negativos'
]

df_cluster = df_resultados[columnas_clustering].copy()

scaler = StandardScaler()
datos_escalados = scaler.fit_transform(df_cluster)

df_clustering = pd.DataFrame(datos_escalados, columns=columnas_clustering)
df_clustering['mes'] = df_resultados['mes']

inertias = []
silhouettes = []
Krange = range(2,10)
for k in Krange:
    kmeans = KMeans(n_clusters =k, random_state=42,n_init='auto')
    etiquetas = kmeans.fit_predict(df_clustering.drop('mes', axis=1))
    
    inertia = kmeans.inertia_
    inertias.append(inertia)
    
    silhouette_avg = silhouette_score(df_clustering.drop('mes',axis=1), etiquetas)
    silhouettes.append(silhouette_avg)
    
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


# k = 4

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(df_clustering.drop('mes', axis=1))

df_clustering['cluster'] = cluster_labels

cluster_patterns = df_clustering.groupby('cluster').mean(numeric_only=True)

cluster_patterns_T = cluster_patterns.T

cluster_patterns_T.plot(kind='bar', figsize=(12, 6))  # Usa figsize aquí directamente
plt.title('Media de características por cluster')
plt.xlabel('Variable')
plt.ylabel('Valor promedio (escalado)')
plt.grid(True)
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(cluster_patterns, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Mapa de calor: características promedio por cluster')
plt.xlabel('Variables')
plt.ylabel('Cluster')
plt.tight_layout()
plt.show()


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

# Mostrar de qué mes cayó en qué cluster
meses_por_cluster = df_clustering[['mes', 'cluster']].sort_values('cluster')

print(meses_por_cluster.to_string(index=False))

print(cluster_patterns)