import dash
from dash import dcc, html
import plotly.graph_objs as go
import sqlite3
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output

#modelo de machine learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Leer datos desde la base de datos SQLite
db_path = "produccion.db"
conn = sqlite3.connect(db_path)
query = "SELECT * FROM produccion"
df_db = pd.read_sql(query, conn)
df_db["fecha"] = pd.to_datetime(df_db["fecha"])

# Preparación de datos
X = df_db.drop(columns=['produccion', 'fecha'])
y = df_db['produccion']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Evaluar el modelo
y_pred_test = model.predict(X_test)
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
        interval=1*1000,  # in milliseconds (1 second for this example)
        n_intervals=0
    )
])

# Establecemos una pendiente basada en la tendencia original
total_hours = 30 * 24
slope = (5000 - 1000) / total_hours

# 1. Calcula el porcentaje promedio de medicamentos defectuosos
avg_defect_percentage = (df_db["defectuosos"] / df_db["produccion"]).mean()

def simulate_new_data(last_date, last_production, make_prediction=False): 
    new_date = last_date + pd.Timedelta(hours=1)
    
    # Simulamos la tendencia al alza basada en la pendiente y una pequeña variación
    new_production = last_production + slope + np.random.normal(0, 300)
    new_production = int(np.clip(new_production, 5000, 10000))
    
    # 2. Usa el porcentaje promedio para calcular los defectuosos basados en la nueva producción
    new_defect = int(new_production * np.random.uniform(avg_defect_percentage - 0.01, avg_defect_percentage + 0.01))
    new_efficiency = np.random.uniform(0.5, 1.0) * 100
    
    # Simulamos el inventario
    consumo = int(new_production * np.random.uniform(0.01, 0.02))
    new_inventory = df_db["inventario"].iloc[-1] - consumo
    if new_inventory < 10000 and np.random.random() < 0.1:
        new_inventory += 30000

    if make_prediction:
        features = np.array([new_defect, new_efficiency, new_inventory]).reshape(1, -1)
        predicted_production = model.predict(features)[0]
        return pd.DataFrame({
            "fecha": [new_date],
            "produccion": [predicted_production],
            "defectuosos": [new_defect],
            "eficiencia": [new_efficiency],
            "inventario": [new_inventory]
        })
    return pd.DataFrame({
        "fecha": [new_date],
        "produccion": [new_production],
        "defectuosos": [new_defect],
        "eficiencia": [new_efficiency],
        "inventario": [new_inventory]
    })
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
    global df_db, accumulated_error

    # Simulate new data
    new_real_data = simulate_new_data(df_db["fecha"].iloc[-1], df_db["produccion"].iloc[-1])
    df_db = pd.concat([df_db, new_real_data])

    # Make a prediction
    new_predicted_data = simulate_new_data(df_db["fecha"].iloc[-1], df_db["produccion"].iloc[-1], make_prediction=True)

    # Calculate error
    error = mean_absolute_error([new_real_data['produccion'].iloc[0]], [new_predicted_data['produccion'].iloc[0]])
    accumulated_error += error

    # Re-train the model if accumulated error is too high
    if accumulated_error > SOME_THRESHOLD_VALUE:
        X = df_db.drop(columns=['produccion', 'fecha'])
        y = df_db['produccion']
        model.fit(X, y)
        accumulated_error = 0  # Reset accumulated error

    return [
        {
            'data': [
                go.Scatter(x=df_db['fecha'], y=df_db['produccion'], mode='lines', name='Producción observada'),
                go.Scatter(x=new_predicted_data['fecha'], y=new_predicted_data['produccion'], mode='lines+markers', name='Predicción')
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