"""
Selecciona las variables de entrada de un modelo de clasificacion de
enfermedad cardiaca.

Genera el archivo que verifica tests/test_src.py:

    submission/estimator.pkl -> GridSearchCV ya ajustado; predice con el
                                mejor Pipeline(StandardScaler, SelectKBest,
                                LogisticRegression)

La cantidad de variables que se conservan (k) y la regularizacion (C) se
eligen con validacion cruzada estratificada de 5 particiones sobre el
conjunto de entrenamiento. La seleccion vive dentro del Pipeline para que
se ajuste solo con cada particion de entrenamiento. El conjunto de prueba
solo se usa al final.
"""

import pickle
import warnings
from pathlib import Path

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FOLDER = Path(__file__).resolve().parent.parent

ESTIMATOR_PATH = FOLDER / "submission" / "estimator.pkl"

TARGET = "target"

RANDOM_STATE = 12345

# thal trae dos filas con valores sucios ("1" y "2"); sus dummies quedan
# constantes en algunas particiones y f_classif avisa. SelectKBest las deja
# al final del ranking, asi que el aviso no afecta la seleccion.
warnings.filterwarnings("ignore", message="Features .* are constant")
warnings.filterwarnings("ignore", message="invalid value encountered in divide")


def load_dataset():
    """Carga el dataset y codifica la variable categorica thal."""

    dataset = pd.read_csv(FOLDER / "data" / "heart_disease.csv")
    y = dataset.pop(TARGET)
    x = pd.get_dummies(dataset, columns=["thal"], dtype=float)

    return x, y


def make_grid_search(n_features):
    """Construye la busqueda sobre k y la regularizacion del clasificador."""

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(score_func=f_classif)),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    return GridSearchCV(
        estimator=pipeline,
        param_grid={
            "selector__k": list(range(1, n_features + 1)),
            "classifier__C": [0.01, 0.1, 1, 10],
        },
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="balanced_accuracy",
        refit=True,
        return_train_score=True,
    )


def main():
    """Busca los hiperparametros, evalua el modelo elegido y lo guarda."""

    x, y = load_dataset()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    estimator = make_grid_search(x.shape[1]).fit(x_train, y_train)

    mask = estimator.best_estimator_["selector"].get_support()
    print(f"Variables conservadas (k = {estimator.best_params_['selector__k']}):")
    for nombre in x.columns[mask]:
        print(f"  {nombre}")
    print(f"C: {estimator.best_params_['classifier__C']}")
    print(f"Balanced accuracy validacion cruzada: {estimator.best_score_:.4f}")
    print(f"Accuracy train: {estimator.score(x_train, y_train):.4f} (balanced)")

    print(classification_report(y_test, estimator.predict(x_test)))

    ESTIMATOR_PATH.parent.mkdir(exist_ok=True)

    with open(ESTIMATOR_PATH, "wb") as archivo:
        pickle.dump(estimator, archivo)

    print(f"Estimador guardado en {ESTIMATOR_PATH}")

    return estimator


if __name__ == "__main__":
    main()
