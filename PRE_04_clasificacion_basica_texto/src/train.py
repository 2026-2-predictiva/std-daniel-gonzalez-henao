"""
Entrenamiento del modelo de clasificacion de texto para el dataset de
frases financieras.

Genera los artefactos que evalua tests/test_src.py:

    submission/vectorizer.pkl -> TfidfVectorizer ya ajustado
    submission/clf.pkl        -> LinearSVC ya entrenado

El test aplica vectorizer.transform(...) y luego clf.predict(...), por lo
que los dos objetos se serializan por separado y el vocabulario debe
quedar dentro del vectorizador.
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

FOLDER = Path(__file__).resolve().parent.parent


def load_dataset():
    """Carga el dataset igual que el test."""

    dataframe = pd.read_csv(
        FOLDER / "data" / "sentences.csv.zip",
        index_col=False,
        compression="zip",
    )

    return dataframe.phrase, dataframe.target


def make_vectorizer():
    """Construye el vectorizador de texto.

    No se remueven stopwords: en este dominio palabras como "no", "down"
    o "against" pertenecen a la lista de sklearn pero son justamente las
    que marcan la polaridad de la frase.
    """

    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )


def make_classifier():
    """Construye el clasificador lineal."""

    return LinearSVC(C=2, random_state=42)


def train():
    """Ajusta el vectorizador y el clasificador, y los serializa."""

    x, y = load_dataset()

    vectorizer = make_vectorizer()
    x_vectorized = vectorizer.fit_transform(x)

    clf = make_classifier()
    clf.fit(x_vectorized, y)

    accuracy = accuracy_score(y_true=y, y_pred=clf.predict(x_vectorized))
    print(f"Accuracy sobre el dataset completo: {accuracy:.4f}  (meta: > 0.854)")
    print(f"Terminos en el vocabulario: {len(vectorizer.vocabulary_)}")

    submission = FOLDER / "submission"
    submission.mkdir(exist_ok=True)

    with open(submission / "vectorizer.pkl", "wb") as file:
        pickle.dump(vectorizer, file)

    with open(submission / "clf.pkl", "wb") as file:
        pickle.dump(clf, file)

    print(f"Artefactos guardados en {submission}")

    return accuracy


if __name__ == "__main__":
    train()
