"""Generate every Lesson 1 figure task from one command."""

from course_progression import course_progression, configure as configure_diagrams
from modelling_path import modelling_path
from stock_direction_check import generate_direction_check
from stock_level_comparison import generate_level_comparison


if __name__ == "__main__":
    generate_level_comparison()
    generate_direction_check()
    configure_diagrams()
    modelling_path()
    course_progression()
