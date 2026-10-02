# Lesson 1 figure tasks

Figure code is organised by teaching task, not by output file. A task may produce a pair of controlled-comparison images or one image with several panels.

- `stock_level_comparison.py` generates the two long-period price-level comparisons.
- `stock_direction_check.py` generates the 20-day close-up and five-year baseline comparison as one multi-panel figure.
- `modelling_path.py` generates the reusable six-question modelling path.
- `course_progression.py` generates the four-stage course progression.
- `generate_all.py` runs every Lesson 1 figure task.

Shared stock data, calculations, palette and export settings remain in `../stock_figure_gen.py`. Shared diagram styling remains in `../diagram_figure_gen.py`. The task entry points define ownership without duplicating common logic.
