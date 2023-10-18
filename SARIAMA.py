import pandas as pd
import numpy as np
import sqlite3
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Leer datos desde la base de datos SQLite
db_path = "produccion.db"
conn = sqlite3.connect(db_path)
query = "SELECT fecha, produccion FROM produccion"
df_tendencia = pd.read_sql(query, conn)
df_tendencia["fecha"] = pd.to_datetime(df_tendencia["fecha"])

# Dividir los datos en entrenamiento y prueba
train_size = int(0.8 * len(df_tendencia))
train = df_tendencia[:train_size]
test = df_tendencia[train_size:]

# Ajuste del modelo SARIMA en el conjunto de entrenamiento
model = SARIMAX(train["produccion"], order=(1,1,1), seasonal_order=(1,1,1,24))
results = model.fit()

# Predicción en el conjunto de prueba
forecast = results.get_forecast(steps=len(test))
predicted_values = forecast.predicted_mean

# Métricas de evaluación
mse = mean_squared_error(test["produccion"], predicted_values)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test["produccion"], predicted_values)
mape = np.mean(np.abs((test["produccion"] - predicted_values) / test["produccion"])) * 100

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}%")


