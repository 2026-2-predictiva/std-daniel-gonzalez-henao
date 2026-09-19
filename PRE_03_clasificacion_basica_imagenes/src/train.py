"""
Entrenamiento del modelo de clasificacion para el dataset Digits.

Genera el artefacto que evalua tests/test_src.py:

    submission/estimator.pkl -> Pipeline ya entrenado

El test llama directamente estimator.predict(data) sobre las imagenes
crudas de sklearn.datasets.load_digits, de modo que el escalamiento debe
viajar dentro del propio estimador. Por eso se serializa un Pipeline que
encadena el StandardScaler con el clasificador.
"""

import pickle
from pathlib import Path

from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FOLDER = Path(__file__).resolve().parent.parent


def load_dataset():
    """Carga el dataset igual que el test."""

    digits = datasets.load_digits(return_X_y=True)
    data, target = digits

    return data, target


def make_estimator():
    """Construye el pipeline de escalamiento y clasificacion."""

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=5,
                    gamma=0.005,
                    random_state=42,
                ),
            ),
        ]
    )


def train():
    """Entrena el pipeline sobre todo el dataset y lo serializa."""

    x, y = load_dataset()

    estimator = make_estimator()
    estimator.fit(x, y)

    accuracy = accuracy_score(y_true=y, y_pred=estimator.predict(x))
    print(f"Accuracy sobre el dataset completo: {accuracy:.4f}  (meta: > 0.96)")

    submission = FOLDER / "submission"
    submission.mkdir(exist_ok=True)

    with open(submission / "estimator.pkl", "wb") as file:
        pickle.dump(estimator, file)

    print(f"Artefacto guardado en {submission}")

    return accuracy


if __name__ == "__main__":
    train()
