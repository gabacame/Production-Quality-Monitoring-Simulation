# Importar bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from statsmodels.tsa.arima.model import ARIMA

# Función principal
def main():
    # Conexión y lectura de datos
    db_path = "produccion.db"
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM produccion"
    df_db = pd.read_sql(query, conn)
    df_db["fecha"] = pd.to_datetime(df_db["fecha"])

    # Convertir el dataframe a una serie temporal
    series = df_db.set_index("fecha")["produccion"]

    # Visualizar la serie temporal
    series.plot()
    plt.title("Producción Original")
    plt.show()

    # Construir el modelo ARIMA
    model = ARIMA(series, order=(5,1,0))
    model_fit = model.fit()

    # Hacer predicciones para la próxima semana
    forecast = model_fit.forecast(steps=24*7)

    # Visualizar las predicciones
    plt.plot(series.index, series.values, label='Historial')
    forecast_index = pd.date_range(series.index[-1], periods=24*7+1, closed='right')
    plt.plot(forecast_index, forecast, color='red', label='Predicción')
    plt.title("Predicción de Producción para la Próxima Semana")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
