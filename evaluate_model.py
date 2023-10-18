import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib

# Cargar el modelo entrenado y el conjunto de prueba
model = joblib.load("optimized_model.pkl")
#model = joblib.load("trained_model.pkl")
test = pd.read_csv("test_data.csv")
test["fecha"] = pd.to_datetime(test["fecha"])

# Hacer predicciones
features = ['produccion_hace_1h', 'produccion_hace_2h', 'produccion_hace_3h', 'promedio_produccion_24h']
X_test = test[features]
y_test = test['produccion']
y_pred = model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse}")
