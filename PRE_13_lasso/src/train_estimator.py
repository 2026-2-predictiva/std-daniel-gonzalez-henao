"""
Selecciona las variables de entrada de un modelo de regresion del consumo
de combustible (mpg) usando Lasso.

Genera el archivo que verifica tests/test_src.py:

    submission/estimator.pkl -> GridSearchCV ya ajustado; predice con el
                                mejor Pipeline(StandardScaler, Lasso)

La regularizacion (alpha) se elige con validacion cruzada de 5 particiones
sobre el conjunto de entrenamiento. El escalado vive dentro del Pipeline para
que se ajuste solo con cada particion de entrenamiento, y las variables con
coeficiente cero son las que Lasso descarta. El conjunto de prueba solo se
usa al final.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FOLDER = Path(__file__).resolve().parent.parent

ESTIMATOR_PATH = FOLDER / "submission" / "estimator.pkl"

TARGET = "mpg"

RANDOM_STATE = 12345


def load_dataset():
    """Carga el dataset y separa la variable objetivo."""

    dataset = pd.read_csv(FOLDER / "data" / "mtcars.csv")
    y = dataset.pop(TARGET)

    return dataset, y


def make_grid_search():
    """Construye la busqueda sobre la regularizacion de Lasso."""

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lasso", Lasso(max_iter=100000)),
        ]
    )

    return GridSearchCV(
        estimator=pipeline,
        param_grid={"lasso__alpha": np.logspace(-3, 1, 50)},
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="neg_mean_squared_error",
        refit=True,
        return_train_score=True,
    )


def main():
    """Busca el alpha, evalua el modelo elegido y lo guarda."""

    x, y = load_dataset()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    estimator = make_grid_search().fit(x_train, y_train)

    lasso = estimator.best_estimator_["lasso"]

    print(f"alpha: {estimator.best_params_['lasso__alpha']:.4f}")
    print(f"MSE validacion cruzada: {-estimator.best_score_:.4f}")
    print("Coeficientes (0 = variable descartada):")
    for nombre, coef in zip(x.columns, lasso.coef_):
        print(f"  {nombre:6s} {coef: .4f}")

    y_pred = estimator.predict(x_test)
    print(f"MSE prueba: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"R2 prueba: {r2_score(y_test, y_pred):.4f}")

    ESTIMATOR_PATH.parent.mkdir(exist_ok=True)

    with open(ESTIMATOR_PATH, "wb") as archivo:
        pickle.dump(estimator, archivo)

    print(f"Estimador guardado en {ESTIMATOR_PATH}")

    return estimator


if __name__ == "__main__":
    main()
