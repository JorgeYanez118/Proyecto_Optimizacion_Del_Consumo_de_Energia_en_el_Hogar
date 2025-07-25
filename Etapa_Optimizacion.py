import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Cargar los coeficientes desde el archivo CSV
ruta_csv = 'Etapa3/resultados_modelos_mensuales.csv'
df_resultados = pd.read_csv(ruta_csv)

mes_objetivo = '2006-12'
fila_mes = df_resultados[df_resultados['mes'] == mes_objetivo].iloc[0]

# 3. Obtener los coeficientes del modelo polinómico f(x) = a·x³ + b·x² + c·x + d
a = fila_mes['coef_x3']
b = fila_mes['coef_x2']
c = fila_mes['coef_x1']
d = fila_mes['intercepto']

print(f"Coeficientes para {mes_objetivo}:")
print(f"a = {a:.6e}, b = {b:.6e}, c = {c:.6e}, d = {d:.6f}")

# 4. Definir funciones
def f(x):
    return a*x**3 + b*x**2 + c*x + d

def f1(x):
    return 3*a*x**2 + 2*b*x + c

def f2(x):
    return 6*a*x + 2*b

# 5. Método de Newton-Raphson
def newton_raphson(f1, f2, x0, tol=1e-6, max_iter=100):
    x = x0
    for i in range(max_iter):
        fx = f1(x)
        dfx = f2(x)
        if abs(dfx) < 1e-10:
            print(f"Derivada cercana a cero en iteración {i}. Se detiene.")
            return None, i
        x_new = x - fx / dfx
        if abs(x_new - x) < tol:
            return x_new, i+1
        x = x_new
    print("No convergió tras máximo número de iteraciones.")
    return None, max_iter

# 6. Aplicar Newton-Raphson desde distintos puntos iniciales
puntos_iniciales = [1000, 800, 600, 400, 200]
resultados = []

print("\nResultados del método de Newton-Raphson:\n")
for x0 in puntos_iniciales:
    x_min, iters = newton_raphson(f1, f2, x0)
    if x_min is not None and 0 <= x_min <= 1440:
        fx_val = f(x_min)
        print(f"x₀ = {x0}, x óptimo ≈ {x_min:.2f}, f(x) ≈ {fx_val:.2f} Wh, iteraciones = {iters}")
        resultados.append((x0, x_min, fx_val, iters))
    else:
        print(f"x₀ = {x0}, sin convergencia válida o fuera de rango horario.")

# 7. (Opcional) Graficar f(x), f′(x) y marcar raíces encontradas
x_vals = np.linspace(0, 1440, 500)
fx_vals = f(x_vals)
f1_vals = f1(x_vals)

plt.figure(figsize=(12, 6))
plt.plot(x_vals, fx_vals, label='f(x): consumo estimado', color='blue')
plt.plot(x_vals, f1_vals, label="f'(x): derivada", color='orange', linestyle='--')
for _, x_crit, fx_crit, _ in resultados:
    plt.plot(x_crit, f(x_crit), 'ro')
    plt.text(x_crit, f(x_crit)+0.5, f'{x_crit:.0f}', color='red', fontsize=9, ha='center')
plt.title(f'Modelo polinómico y derivada - {mes_objetivo.capitalize()}')
plt.xlabel('Minuto del día')
plt.ylabel('Consumo (Wh)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
