import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import sqlite3
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Leer datos desde la base de datos SQLite
db_path = "produccion.db"
conn = sqlite3.connect(db_path)
query = "SELECT * FROM produccion"
df_db = pd.read_sql(query, conn)
df_db["fecha"] = pd.to_datetime(df_db["fecha"])

# Ajuste del modelo SARIMA en el conjunto de entrenamiento
train_size = int(0.8 * len(df_db))
train = df_db[:train_size]
test = df_db[train_size:]
model = SARIMAX(train["produccion"], order=(1,1,1), seasonal_order=(1,1,1,24))
results = model.fit()

# Predicción en el conjunto de prueba
forecast = results.get_forecast(steps=len(test))
predicted_values = forecast.predicted_mean
forecast_dates = pd.date_range(train["fecha"].iloc[-1] + pd.Timedelta(hours=1), periods=len(test), freq="H")

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Dashboard de Producción de Laboratorios PiSA"),
    
    dcc.Graph(
        id='produccion-tiempo',
        figure={
            'data': [
                go.Scatter(x=df_db['fecha'], y=df_db['produccion'], mode='lines', name='Producción observada'),
                go.Scatter(x=forecast_dates, y=predicted_values, mode='lines', name='Predicción SARIMA', line=dict(color='red'))
            ],
            'layout': go.Layout(title='Producción y Predicción a lo largo del tiempo')
        }
    ),
    
    dcc.Graph(
        id='defectuosos-tiempo',
        figure={
            'data': [
                go.Scatter(x=df_db['fecha'], y=df_db['defectuosos'], mode='lines', name='Defectuosos')
            ],
            'layout': go.Layout(title='Medicamentos Defectuosos a lo largo del Tiempo')
        }
    ),

    dcc.Graph(
        id='eficiencia-tiempo',
        figure={
            'data': [
                go.Bar(x=df_db['fecha'], y=df_db['eficiencia'], name='Eficiencia')
            ],
            'layout': go.Layout(title='Eficiencia de la Línea de Producción por Día')
        }
    ),
    
    dcc.Graph(
        id='inventario-tiempo',
        figure={
            'data': [
                go.Scatter(x=df_db['fecha'], y=df_db['inventario'], mode='lines', name='Inventario')
            ],
            'layout': go.Layout(title='Nivel de Inventario de Materias Primas a lo largo del Tiempo')
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
