"""Generate the six-question modelling-path figure."""

from pathlib import Path
import sys

LECTURE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LECTURE_DIR))

from diagram_figure_gen import configure, modelling_path


if __name__ == "__main__":
    configure()
    modelling_path()
