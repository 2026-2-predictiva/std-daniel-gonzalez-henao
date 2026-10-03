"""
Selecciona los hiperparametros de un modelo de calidad del vino y lo guarda.

Genera el archivo que verifica tests/test_src.py:

    submission/estimator.pkl -> GridSearchCV ya ajustado; predice con el
                                mejor Pipeline(StandardScaler, ElasticNet)

Los hiperparametros (alpha y l1_ratio) se eligen con validacion cruzada
de 5 particiones sobre el conjunto de entrenamiento. El conjunto de prueba
solo se usa al final para medir el modelo elegido.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FOLDER = Path(__file__).resolve().parent.parent

ESTIMATOR_PATH = FOLDER / "submission" / "estimator.pkl"

TARGET = "quality"

RANDOM_STATE = 12345

PARAM_GRID = {
    "regressor__alpha": np.logspace(-4, 1, 11),
    "regressor__l1_ratio": np.linspace(0.1, 1.0, 10),
}


def load_dataset():
    """Carga el dataset de vinos tintos."""

    return pd.read_csv(FOLDER / "data" / "winequality-red.csv")


def make_grid_search():
    """Construye la busqueda en malla sobre el pipeline escalado + ElasticNet."""

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", ElasticNet(max_iter=10_000)),
        ]
    )

    return GridSearchCV(
        estimator=pipeline,
        param_grid=PARAM_GRID,
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="neg_mean_squared_error",
        refit=True,
        return_train_score=True,
    )


def report(estimator, x, y):
    """Calcula las metricas de regresion de un estimador ajustado."""

    prediccion = estimator.predict(x)

    return {
        "mse": mean_squared_error(y, prediccion),
        "mae": mean_absolute_error(y, prediccion),
        "r2": r2_score(y, prediccion),
    }


def main():
    """Busca los hiperparametros, evalua el modelo elegido y lo guarda."""

    dataframe = load_dataset()

    x = dataframe.drop(columns=TARGET)
    y = dataframe[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    estimator = make_grid_search().fit(x_train, y_train)

    print("Mejores hiperparametros:")
    for nombre, valor in estimator.best_params_.items():
        print(f"  {nombre}: {valor:.4g}")
    print(f"MSE validacion cruzada: {-estimator.best_score_:.4f}")

    for particion, (x_part, y_part) in {
        "train": (x_train, y_train),
        "test": (x_test, y_test),
    }.items():
        metricas = report(estimator, x_part, y_part)
        print(particion, " ".join(f"{k}={v:.4f}" for k, v in metricas.items()))

    ESTIMATOR_PATH.parent.mkdir(exist_ok=True)

    with open(ESTIMATOR_PATH, "wb") as archivo:
        pickle.dump(estimator, archivo)

    print(f"Estimador guardado en {ESTIMATOR_PATH}")

    return estimator


if __name__ == "__main__":
    main()
