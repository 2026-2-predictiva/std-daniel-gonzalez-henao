"""
Entrena el modelo de precios de vivienda y lo guarda para el despliegue.

Genera el archivo que verifica tests/test_src.py:

    submission/house_predictor.pkl -> LinearRegression sobre las 7 variables
                                      que pide el formulario de web_app.py

El modelo se evalua primero en una particion de prueba y despues se
reentrena con todos los datos antes de guardarlo.
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

FOLDER = Path(__file__).resolve().parent.parent

MODEL_PATH = FOLDER / "submission" / "house_predictor.pkl"

# Deben coincidir con los campos del formulario en templates/index.html.
FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "condition",
]
TARGET = "price"

RANDOM_STATE = 42


def load_dataset():
    """Carga el dataset de ventas de vivienda."""

    return pd.read_csv(FOLDER / "data" / "house_data.csv")


def evaluate(x, y):
    """Mide el desempeno del modelo en una particion de prueba."""

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    modelo = LinearRegression().fit(x_train, y_train)
    prediccion = modelo.predict(x_test)

    return {
        "r2_train": modelo.score(x_train, y_train),
        "r2_test": r2_score(y_test, prediccion),
        "mae_test": mean_absolute_error(y_test, prediccion),
    }


def main():
    """Evalua, reentrena con todos los datos y guarda el modelo."""

    dataframe = load_dataset()

    x = dataframe[FEATURES]
    y = dataframe[TARGET]

    for nombre, valor in evaluate(x, y).items():
        print(f"{nombre}: {valor:,.3f}")

    modelo = LinearRegression().fit(x, y)

    MODEL_PATH.parent.mkdir(exist_ok=True)

    with open(MODEL_PATH, "wb") as archivo:
        pickle.dump(modelo, archivo)

    print(f"Modelo guardado en {MODEL_PATH}")

    return modelo


if __name__ == "__main__":
    main()
