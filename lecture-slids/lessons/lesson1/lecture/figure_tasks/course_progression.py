"""Generate the four-stage course-progression figure."""

from pathlib import Path
import sys

LECTURE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LECTURE_DIR))

from diagram_figure_gen import configure, course_progression


if __name__ == "__main__":
    configure()
    course_progression()
