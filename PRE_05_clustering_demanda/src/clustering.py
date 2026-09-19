"""
Agrupamiento de las curvas diarias de demanda comercial.

Genera las tres figuras que verifica tests/test_src.py:

    submission/demanda-comercial.png                   -> los datos crudos
    submission/demanda-comercial-patrones-ejemplo.png  -> un dia por patron
    submission/demanda-comercial-perfiles.png          -> los perfiles hallados

El agrupamiento se hace sobre la curva normalizada por su propio maximo
diario. Sin esa normalizacion el KMeans separa por nivel de consumo (que
crece con los anos y con la epoca del ano) y no por la forma de la curva,
que es lo que interesa.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

FOLDER = Path(__file__).resolve().parent.parent

N_CLUSTERS = 4
RANDOM_STATE = 42

# Paleta categorica validada para fondo claro.
COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]

GRIS = "#8a8a85"
TINTA = "#0b0b0b"
TINTA_SUAVE = "#52514e"

DIAS = [
    "lunes",
    "martes",
    "miercoles",
    "jueves",
    "viernes",
    "sabado",
    "domingo",
]


def load_dataset():
    """Carga el dataset de demanda horaria."""

    dataframe = pd.read_csv(
        FOLDER / "data" / "demanda_comercial.csv.zip",
        compression="zip",
    )

    dataframe["Fecha"] = pd.to_datetime(dataframe["Fecha"])

    return dataframe


def normalize(curvas):
    """Divide cada curva diaria por su propio maximo."""

    return curvas.div(curvas.max(axis=1), axis=0)


def fit_clusters(normalizadas):
    """Agrupa las curvas normalizadas con KMeans."""

    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        n_init=20,
        random_state=RANDOM_STATE,
    )
    kmeans.fit(normalizadas)

    return kmeans


def order_clusters(dataframe):
    """Reordena las etiquetas de KMeans en un orden interpretable.

    Las etiquetas de KMeans son arbitrarias. Se ordenan por dia de la
    semana tipico y, para desempatar los grupos de dias habiles, por ano
    promedio, de modo que la numeracion sea estable entre ejecuciones.
    """

    claves = {
        etiqueta: (
            grupo["Fecha"].dt.dayofweek.median(),
            grupo["Fecha"].dt.year.mean(),
        )
        for etiqueta, grupo in dataframe.groupby("cluster")
    }

    orden = sorted(claves, key=lambda etiqueta: claves[etiqueta])

    return {antigua: nueva for nueva, antigua in enumerate(orden)}


def describe_clusters(dataframe):
    """Arma la descripcion de cada grupo a partir de sus miembros.

    Los dos grupos de dias habiles no se distinguen por el dia de la
    semana sino por el ano, asi que la descripcion combina el tipo de dia
    predominante con la mediana del ano.
    """

    descripciones = []

    for etiqueta in sorted(dataframe["cluster"].unique()):
        grupo = dataframe[dataframe["cluster"] == etiqueta]
        dias = grupo["Fecha"].dt.dayofweek

        habiles = (dias <= 4).mean()
        sabados = (dias == 5).mean()
        domingos = (dias == 6).mean()

        if sabados > 0.5:
            tipo = f"{sabados:.0%} sabados"
        elif domingos > 0.5:
            tipo = f"{domingos:.0%} domingos + {int((dias <= 4).sum())} festivos"
        else:
            tipo = f"{habiles:.0%} dias habiles"

        descripciones.append(
            {
                "cluster": etiqueta,
                "n": len(grupo),
                "tipo": tipo,
                "dia_dominante": DIAS[int(dias.mode().iloc[0])],
                "habiles": habiles,
                "ano_mediano": int(grupo["Fecha"].dt.year.median()),
            }
        )

    return pd.DataFrame(descripciones)


def spread_labels(valores, separacion):
    """Separa verticalmente etiquetas que quedarian encimadas.

    Devuelve una posicion por etiqueta, respetando el orden vertical
    original pero garantizando una distancia minima entre vecinas.
    """

    posiciones = list(valores)
    orden = sorted(range(len(posiciones)), key=lambda i: posiciones[i])

    for anterior, actual in zip(orden, orden[1:]):
        if posiciones[actual] - posiciones[anterior] < separacion:
            posiciones[actual] = posiciones[anterior] + separacion

    return posiciones


def style_axes(ax):
    """Aplica el estilo comun: reja tenue y sin marco superior ni derecho."""

    ax.grid(axis="y", color=GRIS, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)

    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRIS)

    ax.tick_params(colors=TINTA_SUAVE, labelsize=8)


def plot_datos(dataframe, horas, destino):
    """Figura 1: la serie diaria y todas las curvas horarias crudas."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    total = dataframe[horas].sum(axis=1) / 1e6

    axes[0].plot(
        dataframe["Fecha"],
        total,
        color=COLORES[0],
        linewidth=0.5,
        alpha=0.3,
        label="Dia a dia",
    )
    axes[0].plot(
        dataframe["Fecha"],
        total.rolling(7, center=True).mean(),
        color=COLORES[0],
        linewidth=2,
        label="Media movil de 7 dias",
    )
    axes[0].set_title(
        "Demanda total por dia, 2017-2022",
        color=TINTA,
        fontsize=11,
        loc="left",
    )
    axes[0].set_ylabel("GWh por dia", color=TINTA_SUAVE, fontsize=9)
    axes[0].legend(
        loc="lower right",
        frameon=False,
        fontsize=9,
        labelcolor=TINTA_SUAVE,
    )
    style_axes(axes[0])

    curvas = dataframe[horas].to_numpy() / 1e6

    axes[1].plot(
        np.arange(1, 25),
        curvas.T,
        color=COLORES[0],
        linewidth=0.4,
        alpha=0.05,
    )
    axes[1].set_title(
        f"Las {len(dataframe)} curvas horarias superpuestas",
        color=TINTA,
        fontsize=11,
        loc="left",
    )
    axes[1].set_xlabel("Hora del dia", color=TINTA_SUAVE, fontsize=9)
    axes[1].set_ylabel("GWh por hora", color=TINTA_SUAVE, fontsize=9)
    axes[1].set_xticks([1, 6, 12, 18, 24])
    style_axes(axes[1])

    fig.suptitle(
        "Demanda comercial horaria: el nivel varia mucho mas que la forma",
        color=TINTA,
        fontsize=13,
        x=0.01,
        ha="left",
    )

    fig.tight_layout()
    fig.savefig(destino, dpi=150, facecolor="white")
    plt.close(fig)


def plot_patrones_ejemplo(dataframe, normalizadas, descripciones, destino):
    """Figura 2: el dia mas representativo de cada patron."""

    fig, ax = plt.subplots(figsize=(9, 5))

    horas_eje = np.arange(1, 25)
    valores = normalizadas.to_numpy()
    etiquetas = dataframe["cluster"].to_numpy()

    finales = []

    for _, fila in descripciones.iterrows():
        etiqueta = int(fila["cluster"])
        miembros = etiquetas == etiqueta

        centro = valores[miembros].mean(axis=0)
        distancias = np.linalg.norm(valores[miembros] - centro, axis=1)
        medoide = np.flatnonzero(miembros)[distancias.argmin()]

        fecha = dataframe["Fecha"].iloc[medoide]
        curva = valores[medoide]

        ax.plot(
            horas_eje,
            curva,
            color=COLORES[etiqueta],
            linewidth=2,
            label=f"Patron {etiqueta + 1}: {DIAS[fecha.dayofweek]} {fecha:%d/%m/%Y}",
        )

        finales.append(curva[-1])

    for etiqueta, altura in enumerate(spread_labels(finales, 0.018)):
        ax.annotate(
            f"Patron {etiqueta + 1}",
            xy=(24.4, altura),
            color=COLORES[etiqueta],
            fontsize=9,
            va="center",
        )

    ax.set_title(
        "Un dia representativo de cada patron",
        color=TINTA,
        fontsize=13,
        loc="left",
    )
    ax.set_xlabel("Hora del dia", color=TINTA_SUAVE, fontsize=9)
    ax.set_ylabel("Demanda relativa al maximo del dia", color=TINTA_SUAVE, fontsize=9)
    ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
    ax.set_xlim(1, 26)
    style_axes(ax)

    ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=9,
        labelcolor=TINTA_SUAVE,
    )

    fig.tight_layout()
    fig.savefig(destino, dpi=150, facecolor="white")
    plt.close(fig)


def plot_perfiles(dataframe, normalizadas, descripciones, destino):
    """Figura 3: los perfiles promedio y los dias que agrupa cada uno."""

    fig = plt.figure(figsize=(12, 7.5))
    grid = fig.add_gridspec(2, N_CLUSTERS, height_ratios=[1.3, 1], hspace=0.45)

    horas_eje = np.arange(1, 25)
    valores = normalizadas.to_numpy()
    etiquetas = dataframe["cluster"].to_numpy()

    centros = np.vstack(
        [
            valores[etiquetas == etiqueta].mean(axis=0)
            for etiqueta in range(N_CLUSTERS)
        ]
    )

    resumen = fig.add_subplot(grid[0, :])

    for etiqueta in range(N_CLUSTERS):
        resumen.plot(
            horas_eje,
            centros[etiqueta],
            color=COLORES[etiqueta],
            linewidth=2,
            label=f"Perfil {etiqueta + 1}: {descripciones.iloc[etiqueta]['tipo']}",
        )

    for etiqueta, altura in enumerate(spread_labels(centros[:, -1], 0.014)):
        resumen.annotate(
            f"Perfil {etiqueta + 1}",
            xy=(24.4, altura),
            color=COLORES[etiqueta],
            fontsize=9,
            va="center",
        )

    resumen.set_title(
        "Perfiles promedio de demanda",
        color=TINTA,
        fontsize=11,
        loc="left",
    )
    resumen.set_xlabel("Hora del dia", color=TINTA_SUAVE, fontsize=9)
    resumen.set_ylabel("Demanda relativa", color=TINTA_SUAVE, fontsize=9)
    resumen.set_xticks([1, 4, 8, 12, 16, 20, 24])
    resumen.set_xlim(1, 26)
    style_axes(resumen)
    resumen.legend(
        loc="upper left",
        frameon=False,
        fontsize=9,
        labelcolor=TINTA_SUAVE,
        ncols=2,
    )

    for etiqueta in range(N_CLUSTERS):
        ax = fig.add_subplot(grid[1, etiqueta])
        miembros = valores[etiquetas == etiqueta]

        ax.plot(
            horas_eje,
            miembros.T,
            color=COLORES[etiqueta],
            linewidth=0.4,
            alpha=0.04,
        )
        ax.plot(
            horas_eje,
            centros[etiqueta],
            color=COLORES[etiqueta],
            linewidth=2,
        )

        fila = descripciones.iloc[etiqueta]

        ax.set_title(
            f"Perfil {etiqueta + 1}  ({int(fila['n'])} dias)\n"
            f"{fila['tipo']}\n"
            f"ano mediano {int(fila['ano_mediano'])}",
            color=TINTA_SUAVE,
            fontsize=9,
            loc="left",
        )
        ax.set_xticks([1, 12, 24])
        ax.set_ylim(0.55, 1.03)
        style_axes(ax)

        if etiqueta == 0:
            ax.set_ylabel("Demanda relativa", color=TINTA_SUAVE, fontsize=9)

    fig.suptitle(
        "Cuatro perfiles de demanda comercial diaria",
        color=TINTA,
        fontsize=13,
        x=0.01,
        y=0.98,
        ha="left",
    )

    fig.subplots_adjust(left=0.07, right=0.97, top=0.89, bottom=0.08, wspace=0.3)
    fig.savefig(destino, dpi=150, facecolor="white")
    plt.close(fig)


def main():
    """Ejecuta el agrupamiento y genera las tres figuras."""

    dataframe = load_dataset()
    horas = [columna for columna in dataframe.columns if columna.startswith("H")]

    normalizadas = normalize(dataframe[horas])
    kmeans = fit_clusters(normalizadas)

    dataframe["cluster"] = kmeans.labels_
    dataframe["cluster"] = dataframe["cluster"].map(order_clusters(dataframe))

    descripciones = describe_clusters(dataframe)
    print(descripciones.to_string(index=False))

    submission = FOLDER / "submission"
    submission.mkdir(exist_ok=True)

    plot_datos(
        dataframe,
        horas,
        submission / "demanda-comercial.png",
    )
    plot_patrones_ejemplo(
        dataframe,
        normalizadas,
        descripciones,
        submission / "demanda-comercial-patrones-ejemplo.png",
    )
    plot_perfiles(
        dataframe,
        normalizadas,
        descripciones,
        submission / "demanda-comercial-perfiles.png",
    )

    print(f"Figuras guardadas en {submission}")

    return dataframe


if __name__ == "__main__":
    main()
