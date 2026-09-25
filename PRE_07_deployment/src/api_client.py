"""
Cliente de ejemplo para la API de api_server.py.

Uso, con el servidor ya corriendo en otra terminal:

    python PRE_07_deployment/src/api_client.py
"""

import requests

URL = "http://127.0.0.1:5001/predict"

VIVIENDAS = [
    {
        "bedrooms": 3,
        "bathrooms": 1,
        "sqft_living": 1180,
        "sqft_lot": 5650,
        "floors": 1,
        "waterfront": 0,
        "condition": 3,
    },
    {
        "bedrooms": 4,
        "bathrooms": 2.5,
        "sqft_living": 2800,
        "sqft_lot": 8000,
        "floors": 2,
        "waterfront": 1,
        "condition": 4,
    },
]


def main():
    """Envia las viviendas de ejemplo e imprime el precio estimado."""

    respuesta = requests.post(URL, json=VIVIENDAS, timeout=10)
    respuesta.raise_for_status()

    for vivienda, precio in zip(VIVIENDAS, respuesta.json()["prediction"]):
        print(
            f"{vivienda['bedrooms']} hab, {vivienda['bathrooms']} banos, "
            f"{vivienda['sqft_living']} sqft -> $ {precio:,.0f}"
        )


if __name__ == "__main__":
    main()
