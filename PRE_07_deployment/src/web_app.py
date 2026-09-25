"""
Aplicacion web que estima el precio de una vivienda con un formulario.

Uso:

    python PRE_07_deployment/src/web_app.py

y abrir http://127.0.0.1:5000 en el navegador. Requiere haber ejecutado
antes train_model.py.
"""

import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request

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

app = Flask(__name__)


def load_model():
    """Carga el modelo entrenado por train_model.py."""

    with open(MODEL_PATH, "rb") as archivo:
        return pickle.load(archivo)


MODELO = load_model()


def parse_form(formulario):
    """Convierte los campos del formulario en una fila para el modelo.

    El formulario envia waterfront como Yes/No; el modelo espera 1/0.
    """

    fila = {
        "bedrooms": float(formulario["bedrooms"]),
        "bathrooms": float(formulario["bathrooms"]),
        "sqft_living": float(formulario["sqft_living"]),
        "sqft_lot": float(formulario["sqft_lot"]),
        "floors": float(formulario["floors"]),
        "waterfront": 1 if formulario["waterfront"] == "Yes" else 0,
        "condition": int(formulario["condition"]),
    }

    return pd.DataFrame([fila], columns=FEATURES)


@app.route("/", methods=["GET", "POST"])
def index():
    """Muestra el formulario y, si se envio, el precio estimado."""

    prediccion = ""

    if request.method == "POST":
        try:
            precio = MODELO.predict(parse_form(request.form))[0]
            prediccion = f"$ {precio:,.0f}"
        except (KeyError, ValueError):
            prediccion = "Revise los datos: todos los campos deben ser numericos."

    return render_template("index.html", prediction=prediccion)


if __name__ == "__main__":
    app.run(debug=True)
