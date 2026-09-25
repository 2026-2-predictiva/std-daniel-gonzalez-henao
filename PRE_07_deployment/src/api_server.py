"""
API REST que expone el modelo de precios de vivienda.

Uso:

    python PRE_07_deployment/src/api_server.py

Endpoint:

    POST http://127.0.0.1:5001/predict

    Cuerpo: un objeto JSON con las 7 variables, o una lista de objetos.
    Respuesta: {"prediction": [precio, ...]}

Requiere haber ejecutado antes train_model.py.
"""

import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request

FOLDER = Path(__file__).resolve().parent.parent

MODEL_PATH = FOLDER / "submission" / "house_predictor.pkl"

FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "condition",
]

PORT = 5001

app = Flask(__name__)


def load_model():
    """Carga el modelo entrenado por train_model.py."""

    with open(MODEL_PATH, "rb") as archivo:
        return pickle.load(archivo)


MODELO = load_model()


@app.route("/predict", methods=["POST"])
def predict():
    """Recibe una o varias viviendas en JSON y devuelve sus precios."""

    datos = request.get_json(silent=True)

    if datos is None:
        return jsonify({"error": "El cuerpo debe ser JSON."}), 400

    if isinstance(datos, dict):
        datos = [datos]

    faltantes = sorted(
        {variable for fila in datos for variable in FEATURES if variable not in fila}
    )

    if faltantes:
        return jsonify({"error": f"Faltan variables: {', '.join(faltantes)}"}), 400

    try:
        entrada = pd.DataFrame(datos)[FEATURES].astype(float)
    except (TypeError, ValueError):
        return jsonify({"error": "Todas las variables deben ser numericas."}), 400

    prediccion = MODELO.predict(entrada)

    return jsonify({"prediction": [round(float(valor), 2) for valor in prediccion]})


if __name__ == "__main__":
    app.run(port=PORT, debug=True)
