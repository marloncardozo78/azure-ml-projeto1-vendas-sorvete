import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import mlflow
import mlflow.sklearn

# Caminhos
DATA_PATH = "../inputs/icecream_sales.csv"
MODEL_PATH = "../models/modelo_vendas_sorvete.pkl"

# MLflow – nome do experimento
mlflow.set_experiment("Projeto1-Azure-ML-VendasSorvete")

with mlflow.start_run():
    # Autolog captura params/metrics/modelo automaticamente
    mlflow.sklearn.autolog(log_models=True)

    # Carregar dados
    dados = pd.read_csv(DATA_PATH)
    X = dados[['temperature_c']]
    y = dados['sales']

    # Treinar
    modelo = LinearRegression()
    modelo.fit(X, y)

    # Métrica principal (R²)
    r2 = modelo.score(X, y)
    mlflow.log_metric("r2", r2)

    # Salvar modelo no repositório
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(modelo, MODEL_PATH)

print("Pipeline executado com sucesso! Modelo salvo em:", MODEL_PATH)
