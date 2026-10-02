"""Generate the close-up and full-period stock-direction evidence figure."""

from pathlib import Path
import sys

LECTURE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LECTURE_DIR))

from stock_figure_gen import generate_direction_check


if __name__ == "__main__":
    generate_direction_check()
