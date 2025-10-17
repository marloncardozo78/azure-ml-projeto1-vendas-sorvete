import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Caminhos dos diretórios
DATA_PATH = "../inputs/icecream_sales.csv"
MODEL_PATH = "../models/modelo_vendas_sorvete.pkl"

# Carregar dados
dados = pd.read_csv(DATA_PATH)
X = dados[['temperature_c']]
y = dados['sales']

# Treinar modelo
modelo = LinearRegression()
modelo.fit(X, y)

# Salvar modelo
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(modelo, MODEL_PATH)

print("Pipeline executado com sucesso! Modelo salvo em:", MODEL_PATH)
