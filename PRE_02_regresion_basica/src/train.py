"""
Entrenamiento del modelo de regresion para el dataset Auto MPG.

Genera los artefactos que evalua tests/test_src.py:

    submission/features_scaler.pkl -> StandardScaler ya ajustado
    submission/mlp.pkl             -> MLPRegressor ya entrenado

El preprocesamiento replica exactamente el del test, de modo que el orden
y el tipo de las columnas coincidan con los que vera el evaluador.
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

FOLDER = Path(__file__).resolve().parent.parent


def load_dataset():
    """Carga y preprocesa el dataset igual que el test."""

    dataset = pd.read_csv(FOLDER / "data" / "auto_mpg.csv")
    dataset = dataset.dropna()
    dataset["Origin"] = dataset["Origin"].map(
        {1: "USA", 2: "Europe", 3: "Japan"},
    )
    dataset = pd.get_dummies(dataset, columns=["Origin"], prefix="", prefix_sep="")
    y = dataset.pop("MPG")

    return dataset, y


def train():
    """Entrena el escalador y la red neuronal, y los serializa."""

    x, y = load_dataset()

    features_scaler = StandardScaler()
    x_scaled = features_scaler.fit_transform(x)

    mlp = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        max_iter=3000,
        random_state=42,
    )
    mlp.fit(x_scaled, y)

    mse = mean_squared_error(y_true=y, y_pred=mlp.predict(x_scaled))
    print(f"MSE sobre el dataset completo: {mse:.4f}  (meta: < 7.745)")

    submission = FOLDER / "submission"
    submission.mkdir(exist_ok=True)

    with open(submission / "features_scaler.pkl", "wb") as file:
        pickle.dump(features_scaler, file)

    with open(submission / "mlp.pkl", "wb") as file:
        pickle.dump(mlp, file)

    print(f"Artefactos guardados en {submission}")

    return mse


if __name__ == "__main__":
    train()
