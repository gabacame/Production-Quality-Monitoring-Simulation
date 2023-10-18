import dash
from dash import dcc, html
import plotly.graph_objs as go
import sqlite3
import pandas as pd
import joblib

db_path = "produccion.db"
conn = sqlite3.connect(db_path)
query = "SELECT * FROM produccion"
df_db = pd.read_sql(query, conn)
df_db["fecha"] = pd.to_datetime(df_db["fecha"])
model = joblib.load("optimized_model.pkl")

# Generar datos futuros para predecir
horas_futuras = 24 * 3  # predecir 3 días en el futuro
fechas_futuras = pd.date_range(start=df_db["fecha"].iloc[-1] + pd.Timedelta(hours=1), periods=horas_futuras, freq="H")

# Crear DataFrame para el futuro
df_futuro = pd.DataFrame({
    "fecha": fechas_futuras
})

# Combinar el dataframe original con el futuro para calcular características desplazadas
df_combined = pd.concat([df_db, df_futuro], ignore_index=True)

# Agregar características desplazadas
for i in range(1, 4):
    df_combined[f'produccion_hace_{i}h'] = df_combined["produccion"].shift(i)

df_combined["promedio_produccion_24h"] = df_combined["produccion"].rolling(window=24).mean()

# Tomar solo las fechas futuras con características desplazadas
df_futuro = df_combined.iloc[-horas_futuras:].copy()

# Eliminar filas con NaN y las columnas que no se usaron para entrenar el modelo
df_futuro = df_futuro.dropna(subset=["promedio_produccion_24h"])
features = df_futuro[['produccion_hace_1h', 'produccion_hace_2h', 'produccion_hace_3h', 'promedio_produccion_24h']]

# Realizar predicciones
df_futuro["produccion"] = model.predict(features)

# Combinar datos originales y predicciones
df_total = pd.concat([df_db, df_futuro], ignore_index=True)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Dashboard de Producción"),
    
    dcc.Graph(
        id='produccion-tiempo',
        figure={
            'data': [
                go.Scatter(x=df_total['fecha'], y=df_total['produccion'], mode='lines', name='Producción')
            ],
            'layout': go.Layout(title='Producción a lo largo del Tiempo')
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

