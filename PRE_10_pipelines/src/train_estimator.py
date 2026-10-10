"""
Entrena un pipeline de clasificacion de texto para frases financieras.

Genera el archivo que verifica tests/test_src.py:

    submission/estimator.pkl -> GridSearchCV ya ajustado; recibe texto crudo
                                y predice con el mejor
                                Pipeline(TfidfVectorizer, LinearSVC)

El vectorizador y el clasificador viajan en un solo Pipeline, de modo que
la validacion cruzada ajusta el vocabulario solo con cada particion de
entrenamiento. El conjunto de prueba solo se usa al final.
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

FOLDER = Path(__file__).resolve().parent.parent

ESTIMATOR_PATH = FOLDER / "submission" / "estimator.pkl"

RANDOM_STATE = 12345

PARAM_GRID = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [1, 2, 3],
    "clf__C": [0.1, 0.5, 1, 2, 5],
}


def load_dataset():
    """Carga las frases y sus etiquetas."""

    dataframe = pd.read_csv(
        FOLDER / "data" / "sentences.csv.zip",
        index_col=False,
        compression="zip",
    )

    return dataframe.phrase, dataframe.target


def make_grid_search():
    """Construye la busqueda en malla sobre el pipeline TF-IDF + LinearSVC.

    No se remueven stopwords: palabras como "no" o "down" marcan la
    polaridad de la frase.
    """

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, sublinear_tf=True)),
            ("clf", LinearSVC(random_state=RANDOM_STATE)),
        ]
    )

    return GridSearchCV(
        estimator=pipeline,
        param_grid=PARAM_GRID,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="balanced_accuracy",
        refit=True,
        return_train_score=True,
        n_jobs=-1,
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

    estimator = make_grid_search().fit(x_train, y_train)

    print("Mejores hiperparametros:")
    for nombre, valor in estimator.best_params_.items():
        print(f"  {nombre}: {valor}")
    print(f"Balanced accuracy validacion cruzada: {estimator.best_score_:.4f}")

    for particion, (x_part, y_part) in {
        "train": (x_train, y_train),
        "test": (x_test, y_test),
    }.items():
        print(f"{particion} accuracy={accuracy_score(y_part, estimator.predict(x_part)):.4f}")

    print(classification_report(y_test, estimator.predict(x_test)))

    ESTIMATOR_PATH.parent.mkdir(exist_ok=True)

    with open(ESTIMATOR_PATH, "wb") as archivo:
        pickle.dump(estimator, archivo)

    print(f"Estimador guardado en {ESTIMATOR_PATH}")

    return estimator


if __name__ == "__main__":
    main()
