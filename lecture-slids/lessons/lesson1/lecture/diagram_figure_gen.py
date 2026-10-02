"""Generate Lesson 1's two editable course-structure diagrams.

Outputs are intentionally designed for projection and remain editable as SVG.
Run from any working directory with: python diagram_figure_gen.py
"""

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, PathPatch
from matplotlib.path import Path as MplPath


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "figures"

# Shared, low-saturation course palette.
INK = "#20313F"
MUTED = "#6E7B85"
FAINT = "#DCE4E8"
BLUE = "#3E6F8E"
BLUE_LIGHT = "#DDEBF2"
TEAL = "#3D8279"
TEAL_LIGHT = "#DDEDE9"
PURPLE = "#766E9E"
PURPLE_LIGHT = "#E8E5F1"
PAPER = "#FAFBFC"


def configure() -> None:
    """Set stable typography and SVG text behaviour."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Avenir Next", "Avenir", "DejaVu Sans"],
            "font.weight": "normal",
            "text.color": INK,
            "axes.facecolor": PAPER,
            "figure.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "svg.fonttype": "none",  # Preserve text as editable text in SVG.
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    """Export an editable SVG and a high-resolution PNG."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.svg", bbox_inches="tight", pad_inches=0.12)
    fig.savefig(
        OUTPUT_DIR / f"{stem}.png",
        dpi=240,
        bbox_inches="tight",
        pad_inches=0.12,
    )
    plt.close(fig)


def modelling_path() -> None:
    """Figure 3: one path from a real problem to a defensible conclusion."""
    fig, ax = plt.subplots(figsize=(15, 6.6))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7)
    ax.axis("off")

    xs = [1.35, 3.85, 6.35, 8.85, 11.35, 13.65]
    ys = [3.45, 3.45, 3.45, 3.45, 3.45, 3.45]
    labels = [
        "Real\nproblem",
        "Data\nrepresentation",
        "What counts\nas good",
        "Find an\nanswer",
        "Evidence",
        "Limits",
    ]

    # A single visual current: reality -> representation -> justified conclusion.
    ax.plot([xs[0], xs[-1]], [3.45, 3.45], color=FAINT, lw=16, zorder=0,
            solid_capstyle="round")
    for left, right in zip(xs[:-1], xs[1:]):
        ax.add_patch(
            FancyArrowPatch(
                (left + 0.66, 3.45),
                (right - 0.67, 3.45),
                arrowstyle="-|>",
                mutation_scale=18,
                linewidth=2.2,
                color=BLUE,
                zorder=2,
            )
        )

    for i, (x, y, label) in enumerate(zip(xs, ys, labels), start=1):
        if i == 1:
            face, edge = BLUE_LIGHT, BLUE
        elif i == 6:
            face, edge = TEAL_LIGHT, TEAL
        else:
            face, edge = PAPER, BLUE
        ax.add_patch(Circle((x, y), 0.68, facecolor=face, edgecolor=edge,
                            linewidth=2.6, zorder=3))
        ax.text(x, y, str(i), ha="center", va="center", fontsize=20,
                color=edge, fontweight="bold", zorder=4)
        ax.text(x, 2.12, label, ha="center", va="top", fontsize=19,
                color=INK, linespacing=1.16)

    # A quiet return path: limits refine the question without making a dense cycle.
    return_path = MplPath(
        [(13.65, 4.18), (12.4, 6.05), (4.2, 6.05), (1.35, 4.18)],
        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
    )
    ax.add_patch(PathPatch(return_path, fill=False, color=TEAL, linewidth=1.8,
                           alpha=0.82, zorder=1))
    ax.add_patch(
        FancyArrowPatch(
            (1.37, 4.2), (1.31, 4.02), arrowstyle="-|>", mutation_scale=17,
            linewidth=1.8, color=TEAL, zorder=2
        )
    )
    ax.text(7.55, 5.65, "Refine the question", ha="center", va="center",
            fontsize=17, color=TEAL, fontweight="medium")

    ax.text(1.0, 0.58, "FROM REALITY", fontsize=14, color=MUTED,
            fontweight="bold", ha="left")
    ax.text(14.0, 0.58, "TO A JUSTIFIED CONCLUSION", fontsize=14, color=TEAL,
            fontweight="bold", ha="right")

    save(fig, "fig_modelling_path")


def course_progression() -> None:
    """Figure 4: an expanding, cumulative progression of capabilities."""
    fig, ax = plt.subplots(figsize=(15, 6.6))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # One widening ribbon communicates accumulation rather than replacement.
    x = [0.8, 4.25, 7.65, 11.05, 14.45]
    top = [3.95, 4.25, 4.58, 4.94, 5.32]
    bottom = [3.05, 2.75, 2.42, 2.06, 1.68]
    verts = list(zip(x, top)) + list(zip(reversed(x), reversed(bottom))) + [(x[0], top[0])]
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 2) + [MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=BLUE_LIGHT,
                           edgecolor="none", zorder=0))

    centres = [2.05, 5.45, 8.85, 12.25]
    radii = [0.43, 0.53, 0.63, 0.73]
    fills = [BLUE, "#4B788E", TEAL, PURPLE]
    labels = [
        "Predict\na number",
        "Make a classification\ndecision",
        "Discover structure\nwithout labels",
        "Learn useful\nrepresentations",
    ]
    stage_labels = ["STAGE 1", "STAGE 2", "STAGE 3", "STAGE 4"]

    # A continuous centre line binds the four expanding capabilities.
    ax.plot([centres[0], centres[-1]], [3.5, 3.5], color=PAPER, lw=7,
            solid_capstyle="round", zorder=1)
    ax.add_patch(
        FancyArrowPatch(
            (centres[0], 3.5), (13.35, 3.5), arrowstyle="-|>",
            mutation_scale=24, linewidth=2.4, color=INK, alpha=0.72, zorder=2
        )
    )

    for i, (cx, radius, fill, label, stage) in enumerate(
        zip(centres, radii, fills, labels, stage_labels), start=1
    ):
        ax.add_patch(Circle((cx, 3.5), radius, facecolor=fill,
                            edgecolor=PAPER, linewidth=3.2, zorder=3))
        ax.text(cx, 3.5, str(i), ha="center", va="center", color="white",
                fontsize=20 + i, fontweight="bold", zorder=4)
        ax.text(cx, 5.82, stage, ha="center", va="center", fontsize=13,
                color=MUTED, fontweight="bold")
        ax.text(cx, 5.15, label, ha="center", va="top", fontsize=19,
                color=INK, linespacing=1.18, fontweight="medium")

    ax.text(0.85, 0.55, "EACH STAGE KEEPS EARLIER SKILLS",
            ha="left", va="center", fontsize=14, color=BLUE, fontweight="bold")
    ax.text(14.15, 0.55, "AND EXPANDS THE PROBLEMS YOU CAN HANDLE",
            ha="right", va="center", fontsize=14, color=PURPLE, fontweight="bold")

    save(fig, "fig_course_progression")


if __name__ == "__main__":
    configure()
    modelling_path()
    course_progression()
    print(f"Generated diagram figures in {OUTPUT_DIR}")
