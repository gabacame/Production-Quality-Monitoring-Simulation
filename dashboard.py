import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import sqlite3
import pandas as pd

db_path = "produccion.db"
conn = sqlite3.connect(db_path)

query = "SELECT * FROM produccion"
df_db = pd.read_sql(query, conn)
df_db["fecha"] = pd.to_datetime(df_db["fecha"])

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Dashboard de Producción de Laboratorios PiSA"),
    
    dcc.Graph(
        id='produccion-tiempo',
        figure={
            'data': [
                go.Scatter(x=df_db['fecha'], y=df_db['produccion'], mode='lines', name='Producción')
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


