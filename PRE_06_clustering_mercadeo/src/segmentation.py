"""
Segmentacion de mercadeo de adolescentes a partir de su perfil en una red social.

Genera los archivos que verifica tests/test_src.py:

    submission/segmented.csv          -> cada perfil con su segmento asignado
    submission/segmentos-perfiles.png -> los intereses que distinguen cada segmento

Los 36 terminos de interes son conteos muy sesgados: la mayoria de los
perfiles usa cada palabra cero veces y unos pocos la usan decenas de veces.
Si se estandarizan los conteos crudos, esos pocos extremos forman grupos de
un solo perfil. Por eso se aplica log(1 + x) antes de estandarizar.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

FOLDER = Path(__file__).resolve().parent.parent

N_CLUSTERS = 5
RANDOM_STATE = 42

# Edades fuera de este rango no son creibles para estudiantes de secundaria.
EDAD_MINIMA = 13
EDAD_MAXIMA = 20

GRIS = "#8a8a85"
TINTA = "#0b0b0b"
TINTA_SUAVE = "#52514e"


def load_dataset():
    """Carga el dataset de perfiles."""

    return pd.read_csv(FOLDER / "data" / "snsdata.csv")


def interest_columns(dataframe):
    """Devuelve las columnas de terminos de interes (despues de friends)."""

    return list(dataframe.columns[dataframe.columns.get_loc("friends") + 1 :])


def clean(dataframe):
    """Limpia edad y genero.

    Las edades no creibles se marcan como faltantes y todas las faltantes
    se imputan con la edad media de su ano de graduacion. El genero
    faltante se conserva como una categoria propia.
    """

    dataframe = dataframe.copy()

    edad = dataframe["age"].where(
        dataframe["age"].between(EDAD_MINIMA, EDAD_MAXIMA, inclusive="left")
    )
    dataframe["age"] = edad.fillna(
        edad.groupby(dataframe["gradyear"]).transform("mean")
    ).round(3)

    dataframe["gender"] = dataframe["gender"].fillna("NA")

    return dataframe


def transform(intereses):
    """Aplica log(1 + x) y estandariza cada termino."""

    return StandardScaler().fit_transform(np.log1p(intereses))


def fit_clusters(matriz):
    """Agrupa los perfiles con KMeans."""

    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        n_init=20,
        random_state=RANDOM_STATE,
    )
    kmeans.fit(matriz)

    return kmeans


def order_clusters(etiquetas):
    """Numera los segmentos de mayor a menor tamano.

    Las etiquetas de KMeans son arbitrarias; ordenarlas por tamano deja
    una numeracion estable entre ejecuciones.
    """

    orden = np.argsort(-np.bincount(etiquetas), kind="stable")

    return {antigua: nueva for nueva, antigua in enumerate(orden)}


def describe_clusters(dataframe, matriz, intereses):
    """Resume cada segmento: tamano, demografia y terminos distintivos."""

    puntajes = pd.DataFrame(matriz, columns=intereses)
    descripciones = []

    for etiqueta in sorted(dataframe["cluster"].unique()):
        miembros = dataframe["cluster"] == etiqueta
        grupo = dataframe[miembros]
        centro = puntajes[miembros.to_numpy()].mean()
        destacados = centro[centro > 0].nlargest(4)

        descripciones.append(
            {
                "cluster": etiqueta,
                "n": len(grupo),
                "mujeres": (grupo["gender"] == "F").mean(),
                "edad": grupo["age"].mean(),
                "amigos": grupo["friends"].median(),
                "palabras": grupo[intereses].sum(axis=1).median(),
                "terminos": ", ".join(destacados.index) or "ninguno sobre la media",
            }
        )

    return pd.DataFrame(descripciones)


def plot_perfiles(dataframe, matriz, intereses, descripciones, destino):
    """Mapa de calor del puntaje medio de cada termino por segmento."""

    puntajes = pd.DataFrame(matriz, columns=intereses)
    centros = puntajes.groupby(dataframe["cluster"].to_numpy()).mean()

    # Los terminos se ordenan por el segmento en el que mas destacan, para
    # que cada segmento quede como un bloque de color en la diagonal.
    orden = sorted(
        intereses,
        key=lambda termino: (centros[termino].idxmax(), -centros[termino].max()),
    )
    centros = centros[orden]

    fig, ax = plt.subplots(figsize=(13, 4.8))

    limite = 2.5
    imagen = ax.imshow(
        centros.to_numpy(),
        cmap="RdBu_r",
        vmin=-limite,
        vmax=limite,
        aspect="auto",
    )

    ax.set_xticks(range(len(orden)))
    ax.set_xticklabels(orden, rotation=60, ha="right", fontsize=8, color=TINTA_SUAVE)
    ax.set_yticks(range(N_CLUSTERS))
    ax.set_yticklabels(
        [
            f"Segmento {int(fila['cluster']) + 1}  ({int(fila['n']):,} perfiles)"
            for _, fila in descripciones.iterrows()
        ],
        fontsize=9,
        color=TINTA_SUAVE,
    )
    ax.tick_params(length=0)

    for lado in ax.spines.values():
        lado.set_visible(False)

    barra = fig.colorbar(imagen, ax=ax, fraction=0.025, pad=0.01)
    barra.set_label("Puntaje medio (desviaciones estandar)", color=TINTA_SUAVE, fontsize=8)
    barra.ax.tick_params(labelsize=8, colors=TINTA_SUAVE)
    barra.outline.set_edgecolor(GRIS)

    ax.set_title(
        "Intereses que distinguen a cada segmento  (log(1 + conteo), estandarizado; "
        f"valores recortados a +/-{limite})",
        color=TINTA,
        fontsize=11,
        loc="left",
    )

    fig.tight_layout()
    fig.savefig(destino, dpi=150, facecolor="white")
    plt.close(fig)


def main():
    """Ejecuta la segmentacion y guarda los resultados."""

    dataframe = clean(load_dataset())
    intereses = interest_columns(dataframe)

    matriz = transform(dataframe[intereses])
    kmeans = fit_clusters(matriz)

    dataframe["cluster"] = pd.Series(kmeans.labels_).map(order_clusters(kmeans.labels_))

    descripciones = describe_clusters(dataframe, matriz, intereses)
    print(descripciones.to_string(index=False, float_format="%.2f"))

    submission = FOLDER / "submission"
    submission.mkdir(exist_ok=True)

    dataframe.to_csv(submission / "segmented.csv")
    plot_perfiles(
        dataframe,
        matriz,
        intereses,
        descripciones,
        submission / "segmentos-perfiles.png",
    )

    print(f"Resultados guardados en {submission}")

    return dataframe


if __name__ == "__main__":
    main()
