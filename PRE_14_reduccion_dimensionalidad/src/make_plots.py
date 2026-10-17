"""
Reduce a dos dimensiones el dataset de digitos manuscritos de scikit-learn
con PCA, t-SNE y UMAP, y guarda un grafico de dispersion por metodo.

Genera los archivos que verifica tests/test_src.py:

    submission/digits_pca.png
    submission/digits_tsne.png
    submission/digits_umap.png

Cada punto es una imagen de 8x8 pixeles (64 variables) coloreada por el
digito que representa.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import umap
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

FOLDER = Path(__file__).resolve().parent.parent

OUTPUT_FOLDER = FOLDER / "submission"

RANDOM_STATE = 12345


def load_dataset():
    """Carga los digitos y estandariza las 64 variables de pixeles."""

    digits = load_digits()
    x = StandardScaler().fit_transform(digits.data)

    return x, digits.target


def save_scatter(embedding, y, title, filename):
    """Guarda el grafico de dispersion de la representacion en 2D."""

    fig, ax = plt.subplots(figsize=(8, 7))

    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=y,
        cmap="tab10",
        s=12,
        alpha=0.8,
    )
    ax.legend(
        *scatter.legend_elements(),
        title="Digito",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
    )
    ax.set_title(title)
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")

    fig.tight_layout()
    fig.savefig(OUTPUT_FOLDER / filename, dpi=150)
    plt.close(fig)


def main():
    """Calcula las tres reducciones y guarda los graficos."""

    x, y = load_dataset()

    OUTPUT_FOLDER.mkdir(exist_ok=True)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    embedding = pca.fit_transform(x)
    varianza = pca.explained_variance_ratio_.sum()
    print(f"Varianza explicada por los 2 componentes de PCA: {varianza:.4f}")
    save_scatter(embedding, y, "Digitos - PCA", "digits_pca.png")

    tsne = TSNE(n_components=2, random_state=RANDOM_STATE)
    save_scatter(tsne.fit_transform(x), y, "Digitos - t-SNE", "digits_tsne.png")

    reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE)
    save_scatter(reducer.fit_transform(x), y, "Digitos - UMAP", "digits_umap.png")

    print(f"Graficos guardados en {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
