import dash
from dash import dcc, html
import plotly.graph_objs as go
import sqlite3
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output
import warnings
import traceback

# Suprimir advertencias
warnings.filterwarnings('ignore')

# Leer datos desde la base de datos SQLite
db_path = "produccion.db"
conn = sqlite3.connect(db_path)
query = "SELECT * FROM produccion"
df_db = pd.read_sql(query, conn)
df_db["fecha"] = pd.to_datetime(df_db["fecha"])

avg_defect_percentage = df_db["defectuosos"].sum() / df_db["produccion"].sum()

from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

# Preparación de datos
X = df_db['fecha']
y = df_db['produccion']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Entrenar el modelo ARIMA inicialmente con todos los datos disponibles
model = ARIMA(df_db['produccion'], order=(5,1,0))
model_fit = model.fit()

# Evaluar el modelo
y_pred_test = model_fit.forecast(steps=len(X_test))
initial_error = mean_absolute_error(y_test, y_pred_test)

# Supongamos que queremos que la tendencia alcance 6000 después de 30 días adicionales
slope = (6000 - df_db["produccion"].iloc[-1]) / (30 * 24)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Dashboard de Producción de Laboratorio"),
    dcc.Graph(id='produccion-tiempo'),
    dcc.Graph(id='defectuosos-tiempo'),
    dcc.Graph(id='eficiencia-tiempo'),
    dcc.Graph(id='inventario-tiempo'),
    dcc.Interval(
        id='interval-component',
        interval=1*1000,
        n_intervals=0
    )
])

# Variables globales para controlar el reentrenamiento
RETRAIN_INTERVAL = 24
counter = 0

def simulate_new_data(last_date, last_production, make_prediction=False):
    try:
        new_dates = [last_date + pd.Timedelta(hours=i) for i in range(1, 25)]
        new_productions, new_defects, new_efficiencies, new_inventories = [], [], [], []

        for i in range(24):
            current_production = last_production + slope + np.random.normal(0, 30)
            current_production = int(np.clip(current_production, 1000, 6000))

            current_defect = int(current_production * np.random.uniform(avg_defect_percentage - 0.01, avg_defect_percentage + 0.01))
            current_efficiency = np.random.uniform(0.5, 1.0) * 100
            consumo = int(current_production * np.random.uniform(0.01, 0.02))
            current_inventory = (new_inventories[-1] if new_inventories else df_db["inventario"].iloc[-1]) - consumo

            if current_inventory < 10000 and np.random.random() < 0.1:
                current_inventory += 30000

            new_productions.append(current_production)
            new_defects.append(current_defect)
            new_efficiencies.append(current_efficiency)
            new_inventories.append(current_inventory)

            last_production = current_production

        if make_prediction:
            # Reentrenar el modelo con todos los datos
            global model_fit
            model = ARIMA(df_db['produccion'], order=(5,1,0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=24)
            predicted_productions = forecast

        return pd.DataFrame({
            "fecha": new_dates,
            "produccion": new_productions,
            "defectuosos": new_defects,
            "eficiencia": new_efficiencies,
            "inventario": new_inventories
        })
    except Exception as e:
        print("Error en simulate_new_data: ", str(e))
        traceback.print_exc()

# Global variable to store accumulated error
accumulated_error = 0
SOME_THRESHOLD_VALUE = 500

@app.callback(
    [Output('produccion-tiempo', 'figure'),
     Output('defectuosos-tiempo', 'figure'),
     Output('eficiencia-tiempo', 'figure'),
     Output('inventario-tiempo', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_graphs(n_intervals):
    global df_db, counter

    # Simulate new data
    new_real_data = simulate_new_data(df_db["fecha"].iloc[-1], df_db["produccion"].iloc[-1])
    df_db = pd.concat([df_db, new_real_data])
    
    # Increment the counter
    counter += 1
    
    # Make a prediction and retrain if necessary
    if counter % RETRAIN_INTERVAL == 0:
        new_predicted_data = simulate_new_data(df_db["fecha"].iloc[-1], df_db["produccion"].iloc[-1], make_prediction=True)
        counter = 0
    else:
        # Make a prediction without retraining
        new_predicted_data = simulate_new_data(df_db["fecha"].iloc[-1], df_db["produccion"].iloc[-1], make_prediction=True)

    # Calculate error for all 24 points
    errors = mean_absolute_error(new_real_data['produccion'], new_predicted_data['produccion'])
    accumulated_error += errors.sum()

    # Re-train the model if accumulated error is too high
    if accumulated_error > SOME_THRESHOLD_VALUE:
        model = ARIMA(df_db['produccion'], order=(5,1,0))
        model_fit = model.fit(disp=0)
        accumulated_error = 0  # Reset accumulated error

        return [
            {
                'data': [
                    go.Scatter(x=df_db['fecha'], y=df_db['produccion'], mode='lines', name='Producción observada'),
                    go.Scatter(x=new_predicted_data['fecha'], y=new_predicted_data['produccion'], mode='lines', name='Predicción')
                ],
                'layout': go.Layout(title='Producción a lo largo del tiempo')
            },
            {
                'data': [go.Scatter(x=df_db['fecha'], y=df_db['defectuosos'], mode='lines', name='Defectuosos')],
                'layout': go.Layout(title='Medicamentos Defectuosos a lo largo del Tiempo')
            },
            {
                'data': [go.Bar(x=df_db['fecha'], y=df_db['eficiencia'], name='Eficiencia')],
                'layout': go.Layout(title='Eficiencia de la Línea de Producción por Día')
            },
            {
                'data': [go.Scatter(x=df_db['fecha'], y=df_db['inventario'], mode='lines', name='Inventario')],
                'layout': go.Layout(title='Nivel de Inventario de Materias Primas a lo largo del Tiempo')
            }
        ]

if __name__ == '__main__':
    app.run_server(debug=True)
