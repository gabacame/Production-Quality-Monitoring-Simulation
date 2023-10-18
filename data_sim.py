import sqlite3
import pandas as pd
import numpy as np

# Paso 1: Simulación de Datos con Tendencia

# Parámetros
dias = 30
horas_por_dia = 24

# Datos simulados
fechas = pd.date_range(start="2023-10-01", periods=dias*horas_por_dia, freq="H")

# Tendencia al alza
tendencia = np.linspace(1000, 5000, len(fechas))
produccion = (tendencia + np.random.normal(0, 300, len(fechas))).astype(int)
produccion = np.clip(produccion, 1000, 5000)

defectuosos = (produccion * np.random.uniform(0.01, 0.05)).astype(int)
eficiencia = np.clip(np.random.normal(0.8, 0.1, len(fechas)), 0.5, 1.0) * 100

# Inventario de materias primas
inventario_inicial = 50000
consumo = (produccion * np.random.uniform(0.01, 0.02)).astype(int)
inventario = [inventario_inicial]
for i in range(1, len(fechas)):
    inventario_actual = inventario[-1] - consumo[i]
    # Reabastecer el inventario aleatoriamente
    if inventario_actual < 10000 and np.random.random() < 0.1:
        inventario_actual += 30000
    inventario.append(inventario_actual)

df_tendencia = pd.DataFrame({
    "fecha": fechas,
    "produccion": produccion,
    "defectuosos": defectuosos,
    "eficiencia": eficiencia,
    "inventario": inventario
})

# Paso 2: Creación de una Base de Datos en SQL

# Crear una base de datos SQLite y guardar los datos con tendencia
db_path = "produccion.db"
conn = sqlite3.connect(db_path)
df_tendencia.to_sql("produccion", conn, if_exists="replace", index=False)

df_tendencia.head()
