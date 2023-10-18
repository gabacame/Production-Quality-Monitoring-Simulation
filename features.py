import sqlite3
import pandas as pd


db_path = "produccion.db"
conn = sqlite3.connect(db_path)
query = "SELECT * FROM produccion"
df_db = pd.read_sql(query, conn)
df_db["fecha"] = pd.to_datetime(df_db["fecha"])

# Paso 1: Preprocesamiento de datos

# Crear características basadas en datos anteriores
# Uso de un desplazamiento para obtener la producción de las horas anteriores
df_db['produccion_hace_1h'] = df_db['produccion'].shift(1)
df_db['produccion_hace_2h'] = df_db['produccion'].shift(2)
df_db['produccion_hace_3h'] = df_db['produccion'].shift(3)

# Calculando el promedio de producción de las últimas 24 horas
df_db['promedio_produccion_24h'] = df_db['produccion'].rolling(window=24).mean()

# Eliminar las filas con valores NaN (debido al desplazamiento y al cálculo del promedio)
df_db = df_db.dropna()

# Paso 2: Dividir los datos en conjuntos de entrenamiento y prueba

# Usar el 80% de los datos para entrenamiento y el 20% para prueba
train_size = int(0.8 * len(df_db))
train = df_db.iloc[:train_size]
test = df_db.iloc[train_size:]

# Definir las características (X) y la variable objetivo (y)
features = ['produccion_hace_1h', 'produccion_hace_2h', 'produccion_hace_3h', 'promedio_produccion_24h']
X_train = train[features]
y_train = train['produccion']
X_test = test[features]
y_test = test['produccion']

X_train.head()
