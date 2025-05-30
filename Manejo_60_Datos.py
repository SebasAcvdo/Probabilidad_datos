# Sebastian Acevedo - 20222020095
# Universidad Distrital Francisco José de Caldas
# Probabilidad y Estadística

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# Paso 1: Ingreso de datos
data_str = """29.37,39.18,35.67,35.49 29.24,36.67,37.94,36.84 28.17,35.25,27.76,38.72 29.37,33.56,38.63,29.98,
22.04,35.67,31.04,32.06 33.56,27.88,24.19,24.27 20.72,21.34,22.91,30.08 31.61,24.84,36.84,35.49
39.79,28.88,36.67,35.67 26.92,28.25,30.08,32.06 24.84,24.19,30.08,28.25 34.93,38.72,38.63,22.04
28.45,38.18,28.17,29.37 25.91,32.06,35.67,35.67 32.77,21.82,21.69,36.41"""

# Limpiar y procesar los datos
data_str = data_str.replace('\n', ' ')  # Reemplazar saltos de línea por espacios
data_str = data_str.replace(' ', ',')   # Reemplazar espacios por comas
data_str = ','.join(filter(None, data_str.split(',')))  # Eliminar elementos vacíos
data = [float(x) for x in data_str.split(',')]  # Convertir a lista de floats

# Paso 2: Crear intervalos (modificar esta parte)
min_value = np.floor(min(data))  # Usar el mínimo real de los datos
max_value = np.ceil(max(data))   # Usar el máximo real de los datos
bin_width = 2
bins = np.arange(min_value, max_value + bin_width, bin_width)

# Paso 3: Calcular frecuencias
freq, edges = np.histogram(data, bins=bins)
intervals = [f"[{edges[i]:.2f} – {edges[i+1]:.2f})" for i in range(len(edges)-1)]

# Paso 4: Crear tabla de distribución
frequency_table = pd.DataFrame({
    "Intervalo": intervals,
    "Frecuencia absoluta": freq
})
frequency_table["Frecuencia acumulada"] = frequency_table["Frecuencia absoluta"].cumsum()
frequency_table["Frecuencia relativa"] = (frequency_table["Frecuencia absoluta"] / len(data)).round(4)
frequency_table["Frecuencia relativa acumulada"] = frequency_table["Frecuencia relativa"].cumsum().round(4)

# Paso 5: Estadísticos
media = np.mean(data)
mediana = np.median(data)
moda = stats.mode(data, keepdims=True).mode[0]
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)

# Paso 6: Crear figura con tabla y 4 histogramas
fig = plt.figure(figsize=(20, 14))
fig.suptitle("Análisis de Frecuencias con Estadísticos", fontsize=18)
gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1.2])

# Tabla de distribución
ax_table = plt.subplot(gs[0, :])
ax_table.axis("off")
table_text = frequency_table.to_string(index=False)
ax_table.text(0, 1, table_text, fontsize=10, fontfamily='monospace', va='top')

# Función auxiliar para dibujar líneas de estadísticos
def draw_stats_lines(ax):
    ax.axvline(media, color='red', linestyle='--', label='Media')
    ax.axvline(mediana, color='green', linestyle='--', label='Mediana')
    ax.axvline(moda, color='blue', linestyle='--', label='Moda')
    ax.axvline(q1, color='orange', linestyle='--', label='Q1')
    ax.axvline(q3, color='purple', linestyle='--', label='Q3')
    ax.legend()

data_np = np.array(data)

# Histograma 1: Frecuencia absoluta
ax1 = plt.subplot(gs[1, 0])
n1, bins1, patches1 = ax1.hist(data_np, bins=bins, color='skyblue', edgecolor='black')
ax1.set_title("Frecuencia Absoluta")
ax1.set_xlabel("Valores")
ax1.set_ylabel("Frecuencia")
draw_stats_lines(ax1)

# Histograma 2: Frecuencia acumulada
ax2 = plt.subplot(gs[1, 1])
n2, bins2, patches2 = ax2.hist(data_np, bins=bins, cumulative=True, color='orange', edgecolor='black')
ax2.set_title("Frecuencia Acumulada")
ax2.set_xlabel("Valores")
ax2.set_ylabel("Frecuencia Acumulada")
draw_stats_lines(ax2)

# Histograma 3: Frecuencia relativa
ax3 = plt.subplot(gs[2, 0])
n3, bins3, patches3 = ax3.hist(data_np, bins=bins, density=True, color='green', edgecolor='black')
ax3.set_title("Frecuencia Relativa")
ax3.set_xlabel("Valores")
ax3.set_ylabel("Frecuencia Relativa")
draw_stats_lines(ax3)

# Histograma 4: Frecuencia relativa acumulada
ax4 = plt.subplot(gs[2, 1])
n4, bins4, patches4 = ax4.hist(data_np, bins=bins, density=True, cumulative=True, color='purple', edgecolor='black')
ax4.set_title("Frecuencia Relativa Acumulada")
ax4.set_xlabel("Valores")
ax4.set_ylabel("Frecuencia Relativa Acumulada")
draw_stats_lines(ax4)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
