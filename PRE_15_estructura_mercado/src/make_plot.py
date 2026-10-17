"""
Visualiza la estructura del mercado accionario a partir de la variacion
diaria (cierre - apertura) de 56 acciones.

Genera el archivo que verifica tests/test_src.py:

    submission/stocks.png

Pasos:

1. La correlacion parcial entre acciones se estima con GraphicalLassoCV
   (covarianza inversa dispersa).
2. Las acciones se agrupan con Affinity Propagation sobre esa covarianza.
3. Las acciones se proyectan a 2D con un embedding LLE.
4. Cada accion se dibuja en la proyeccion, coloreada por grupo, y se unen
   con lineas las parejas con correlacion parcial fuerte.
"""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from sklearn import cluster, covariance, manifold

FOLDER = Path(__file__).resolve().parent.parent

OUTPUT_PATH = FOLDER / "submission" / "stocks.png"

RANDOM_STATE = 12345

# Las lineas se dibujan solo entre parejas con correlacion parcial mayor a
# este valor (en valor absoluto).
EDGE_THRESHOLD = 0.1


def load_variation():
    """Devuelve la variacion diaria (cierre - apertura) por accion."""

    variation = {}
    for path in sorted((FOLDER / "data").glob("*.csv.zip")):
        symbol = path.name.removesuffix(".csv.zip")
        quotes = pd.read_csv(path)
        variation[symbol] = (quotes["close"] - quotes["open"]).to_numpy()

    return pd.DataFrame(variation)


def main():
    """Ajusta el modelo, agrupa las acciones y guarda el grafico."""

    variation = load_variation()
    symbols = variation.columns.to_numpy()

    # Se estandariza para trabajar con correlaciones y no con covarianzas.
    x = variation.to_numpy()
    x = x / x.std(axis=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = covariance.GraphicalLassoCV().fit(x)

    _, labels = cluster.affinity_propagation(
        model.covariance_, random_state=RANDOM_STATE
    )
    n_labels = labels.max() + 1

    for i in range(n_labels):
        print(f"Grupo {i + 1}: {', '.join(symbols[labels == i])}")

    # Posiciones en 2D.
    node_position_model = manifold.LocallyLinearEmbedding(
        n_components=2, eigen_solver="dense", n_neighbors=6, random_state=RANDOM_STATE
    )
    embedding = node_position_model.fit_transform(x.T).T

    # Correlacion parcial a partir de la matriz de precision.
    partial_correlations = model.precision_.copy()
    d = 1 / np.sqrt(np.diag(partial_correlations))
    partial_correlations *= d
    partial_correlations *= d[:, np.newaxis]
    non_zero = np.abs(np.triu(partial_correlations, k=1)) > EDGE_THRESHOLD

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(
        embedding[0],
        embedding[1],
        s=100 * d**2,
        c=labels,
        cmap=plt.cm.nipy_spectral,
        zorder=3,
    )

    start, stop = np.where(non_zero)
    segments = [
        [embedding[:, s], embedding[:, e]] for s, e in zip(start, stop)
    ]
    values = np.abs(partial_correlations[non_zero])
    lines = LineCollection(
        segments,
        zorder=2,
        cmap=plt.cm.hot_r,
        norm=plt.Normalize(0, 0.7 * values.max()),
    )
    lines.set_array(values)
    lines.set_linewidths(15 * values)
    ax.add_collection(lines)

    for index, (name, label, (x_pos, y_pos)) in enumerate(
        zip(symbols, labels, embedding.T)
    ):
        ax.text(
            x_pos + (0.002 if index % 2 else -0.002),
            y_pos + (0.002 if index % 3 else -0.002),
            name,
            size=9,
            ha="left" if index % 2 else "right",
            va="bottom" if index % 3 else "top",
            bbox=dict(
                facecolor=plt.cm.nipy_spectral(label / float(n_labels)),
                edgecolor="none",
                alpha=0.6,
                boxstyle="round,pad=0.2",
            ),
            zorder=4,
        )

    ax.set_title("Estructura del mercado accionario")
    ax.set_xlim(
        embedding[0].min() - 0.15 * np.ptp(embedding[0]),
        embedding[0].max() + 0.10 * np.ptp(embedding[0]),
    )
    ax.set_ylim(
        embedding[1].min() - 0.03 * np.ptp(embedding[1]),
        embedding[1].max() + 0.03 * np.ptp(embedding[1]),
    )
    ax.axis("off")

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)

    print(f"Grafico guardado en {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
