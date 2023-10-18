import pandas as pd
import numpy as np
import sqlite3
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
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
features = ['produccion_hace_1h', 'produccion_hace_2h', 'produccion_hace_3h', 'promedio_produccion_24h']
X_train = train[features]
y_train = train['produccion']

# Definir el modelo y la cuadrícula de hiperparámetros
model = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
}

# Búsqueda de cuadrícula con validación cruzada
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Guardar el modelo con los mejores hiperparámetros encontrados
best_model = grid_search.best_estimator_
joblib.dump(best_model, "optimized_model.pkl")

# Imprimir los mejores hiperparámetros encontrados
print(f"Mejores hiperparámetros: {grid_search.best_params_}")
