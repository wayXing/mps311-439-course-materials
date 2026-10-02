"""Generate the paired long-period stock-level comparison figures."""

from pathlib import Path
import sys

LECTURE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LECTURE_DIR))

from stock_figure_gen import generate_level_comparison


if __name__ == "__main__":
    generate_level_comparison()
