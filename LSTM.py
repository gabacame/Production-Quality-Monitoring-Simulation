import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Simulación de Datos con Tendencia
dias = 30
horas_por_dia = 24
fechas = pd.date_range(start="2023-10-01", periods=dias*horas_por_dia, freq="H")
tendencia = np.linspace(1000, 5000, len(fechas))
produccion = (tendencia + np.random.normal(0, 300, len(fechas))).astype(int)
produccion = np.clip(produccion, 1000, 5000)
defectuosos = (produccion * np.random.uniform(0.01, 0.05)).astype(int)
eficiencia = np.clip(np.random.normal(0.8, 0.1, len(fechas)), 0.5, 1.0) * 100
inventario_inicial = 50000
consumo = (produccion * np.random.uniform(0.01, 0.02)).astype(int)
inventario = [inventario_inicial]
for i in range(1, len(fechas)):
    inventario_actual = inventario[-1] - consumo[i]
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

# Ajuste del modelo ARIMA
model = ARIMA(df_tendencia["produccion"], order=(1,1,1))
results = model.fit()

# Predicción para los próximos 3 días
forecast_steps = 72
forecast = results.get_forecast(steps=forecast_steps)
confidence_interval = forecast.conf_int()
forecast_dates = pd.date_range(df_tendencia["fecha"].iloc[-1] + pd.Timedelta(hours=1), periods=forecast_steps, freq="H")

# Visualización
plt.figure(figsize=(14, 7))
plt.plot(df_tendencia["fecha"], df_tendencia["produccion"], label="Producción observada", color="blue")
plt.plot(forecast_dates, forecast.predicted_mean, label="Predicción", color="red")
plt.fill_between(forecast_dates, confidence_interval.iloc[:, 0], confidence_interval.iloc[:, 1], color="pink", alpha=0.3)
plt.title("Producción y Predicción a lo largo del tiempo")
plt.xlabel("Fecha")
plt.ylabel("Producción")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
