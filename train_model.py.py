import pandas as pd
import numpy as np
import sqlite3
from sklearn.ensemble import RandomForestRegressor
import joblib

# Conectar a la base de datos y obtener datos
conn = sqlite3.connect("produccion.db")
query = "SELECT * FROM produccion"
df_db = pd.read_sql(query, conn)
df_db["fecha"] = pd.to_datetime(df_db["fecha"])

# Preprocesamiento de datos
df_db['produccion_hace_1h'] = df_db['produccion'].shift(1)
df_db['produccion_hace_2h'] = df_db['produccion'].shift(2)
df_db['produccion_hace_3h'] = df_db['produccion'].shift(3)
df_db['promedio_produccion_24h'] = df_db['produccion'].rolling(window=24).mean()
df_db = df_db.dropna()

# Dividir los datos
train_size = int(0.8 * len(df_db))
train = df_db.iloc[:train_size]
test = df_db.iloc[train_size:]
features = ['produccion_hace_1h', 'produccion_hace_2h', 'produccion_hace_3h', 'promedio_produccion_24h']
X_train = train[features]
y_train = train['produccion']

# Entrenar el modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Guardar el modelo entrenado y el conjunto de prueba
joblib.dump(model, "trained_model.pkl")
test.to_csv("test_data.csv", index=False)



