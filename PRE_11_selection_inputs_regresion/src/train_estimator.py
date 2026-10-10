"""
Selecciona las variables de entrada de un modelo de regresion para Auto MPG.

Genera el archivo que verifica tests/test_src.py:

    submission/estimator.pkl -> GridSearchCV ya ajustado; predice con el
                                mejor Pipeline(StandardScaler, SelectKBest,
                                LinearRegression)

La cantidad de variables que se conservan (k) se elige con validacion
cruzada de 5 particiones sobre el conjunto de entrenamiento. La seleccion
vive dentro del Pipeline para que se ajuste solo con cada particion de
entrenamiento. El conjunto de prueba solo se usa al final.
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FOLDER = Path(__file__).resolve().parent.parent

ESTIMATOR_PATH = FOLDER / "submission" / "estimator.pkl"

TARGET = "MPG"

RANDOM_STATE = 12345


def load_dataset():
    """Carga el dataset, elimina nulos y codifica el origen."""

    dataset = pd.read_csv(FOLDER / "data" / "auto_mpg.csv").dropna()
    dataset["Origin"] = dataset["Origin"].map(
        {1: "USA", 2: "Europe", 3: "Japan"},
    )
    dataset = pd.get_dummies(dataset, columns=["Origin"], prefix="", prefix_sep="")
    y = dataset.pop(TARGET)

    return dataset, y


def make_grid_search(n_features):
    """Construye la busqueda sobre la cantidad de variables a conservar."""

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(score_func=f_regression)),
            ("regressor", LinearRegression()),
        ]
    )

    return GridSearchCV(
        estimator=pipeline,
        param_grid={"selector__k": list(range(1, n_features + 1))},
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
    """Busca el mejor k, evalua el modelo elegido y lo guarda."""

    x, y = load_dataset()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    estimator = make_grid_search(x.shape[1]).fit(x_train, y_train)

    mask = estimator.best_estimator_["selector"].get_support()
    print(f"Variables conservadas (k = {estimator.best_params_['selector__k']}):")
    for nombre in x.columns[mask]:
        print(f"  {nombre}")
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
