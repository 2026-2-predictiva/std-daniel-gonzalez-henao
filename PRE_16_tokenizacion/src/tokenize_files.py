"""
Tokeniza los textos de data/ y guarda un token por linea.

Genera los archivos que verifica tests/test_src.py:

    submission/file1.txt
    submission/file2.txt
    submission/file3.txt

Cada archivo de salida corresponde al archivo de entrada con el mismo nombre.
El texto se pasa a minusculas y los tokens son secuencias de letras y
digitos (se descarta la puntuacion y los simbolos).
"""

import re
from pathlib import Path

FOLDER = Path(__file__).resolve().parent.parent

DATA_FOLDER = FOLDER / "data"
OUTPUT_FOLDER = FOLDER / "submission"

TOKEN_PATTERN = re.compile(r"\w+")


def tokenize(text):
    """Devuelve la lista de tokens del texto en minusculas."""

    return TOKEN_PATTERN.findall(text.lower())


def main():
    """Tokeniza cada archivo de data/ y guarda el resultado en submission/."""

    OUTPUT_FOLDER.mkdir(exist_ok=True)

    for path in sorted(DATA_FOLDER.glob("*.txt")):
        tokens = tokenize(path.read_text(encoding="utf-8"))

        (OUTPUT_FOLDER / path.name).write_text(
            "\n".join(tokens) + "\n", encoding="utf-8"
        )

        print(f"{path.name}: {len(tokens)} tokens")


if __name__ == "__main__":
    main()
