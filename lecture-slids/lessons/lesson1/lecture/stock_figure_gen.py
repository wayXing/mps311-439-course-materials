"""Generate the two controlled-comparison stock figures for Lesson 1.

The script intentionally uses the same dates, actual-price series, axes, and
styling in both figures.  It requires only NumPy and Matplotlib and works
offline with the supplied FRED CSV file.
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "quarto" / "data" / "sp500_fred.csv"
OUTPUT_DIR = HERE / "figures"

START_DATE = date(2020, 1, 1)
END_DATE = date(2024, 12, 31)
N_LAGS = 5
CLOSEUP_WINDOW = 20

# Course-site palette, used with stronger functional contrast inside figures.
ACTUAL_COLOUR = "#171713"
AI_COLOUR = "#EF6A4D"
BASELINE_COLOUR = "#5E7EE8"
TEXT_COLOUR = "#171713"
MUTED_TEXT = "#55554D"
GRID_COLOUR = "#DED9CD"
BACKGROUND = "#FFFEFA"


def load_series(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load valid observations, ignoring FRED's blank non-trading days."""
    dates: list[date] = []
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                value = float(row["SP500"])
            except (TypeError, ValueError):
                continue
            dates.append(date.fromisoformat(row["observation_date"]))
            values.append(value)
    return np.asarray(dates, dtype=object), np.asarray(values, dtype=float)


def lagged_design(values: np.ndarray, n_lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Return rows [t-1, ..., t-n_lags] and their observed value at t."""
    features = np.asarray(
        [values[index - n_lags : index][::-1] for index in range(n_lags, len(values))]
    )
    targets = values[n_lags:]
    return features, targets


def fit_pre_2020_model(dates: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Fit linear least squares with an intercept using observations before 2020."""
    features, targets = lagged_design(values, N_LAGS)
    target_dates = dates[N_LAGS:]
    training = target_dates < START_DATE
    design = np.column_stack([np.ones(training.sum()), features[training]])
    coefficients, *_ = np.linalg.lstsq(design, targets[training], rcond=None)
    return coefficients


def build_evaluation_series(
    dates: np.ndarray, values: np.ndarray, coefficients: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build identical 2020--2024 evaluation rows for both comparisons."""
    features, targets = lagged_design(values, N_LAGS)
    target_dates = dates[N_LAGS:]
    evaluation = (target_dates >= START_DATE) & (target_dates <= END_DATE)

    eval_features = features[evaluation]
    ai_prediction = np.column_stack(
        [np.ones(evaluation.sum()), eval_features]
    ) @ coefficients
    yesterday = eval_features[:, 0]
    return target_dates[evaluation], targets[evaluation], ai_prediction, yesterday


def shared_y_limits(*series: np.ndarray) -> tuple[float, float]:
    minimum = min(float(values.min()) for values in series)
    maximum = max(float(values.max()) for values in series)
    padding = 0.055 * (maximum - minimum)
    return minimum - padding, maximum + padding


def draw_chart(
    dates: np.ndarray,
    actual: np.ndarray,
    comparison: np.ndarray,
    comparison_label: str,
    comparison_colour: str,
    output_file: Path,
    y_limits: tuple[float, float],
) -> None:
    """Draw one member of the controlled visual comparison."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Avenir Next", "Avenir", "Helvetica Neue", "DejaVu Sans"],
            "font.size": 18,
            "axes.labelcolor": TEXT_COLOUR,
            "xtick.color": MUTED_TEXT,
            "ytick.color": MUTED_TEXT,
        }
    )

    fig, ax = plt.subplots(figsize=(16, 9), dpi=160)
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    # The dashed comparison sits above the observed series. Its gaps preserve
    # the actual line, while its dashes remain visible on close inspection.
    actual_line, = ax.plot(
        dates,
        actual,
        color=ACTUAL_COLOUR,
        linewidth=2.8,
        label="Actual closing level",
        zorder=2,
    )
    comparison_line, = ax.plot(
        dates,
        comparison,
        color=comparison_colour,
        linewidth=3.0,
        linestyle=(0, (8, 4)),
        label=comparison_label,
        zorder=3,
    )

    ax.set_xlim(date(2020, 1, 1), date(2024, 12, 31))
    ax.set_ylim(*y_limits)
    ax.set_ylabel("S&P 500 closing level", fontsize=21, labelpad=18)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))

    ax.grid(axis="y", color=GRID_COLOUR, linewidth=1.0)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="both", which="major", labelsize=18, length=0, pad=10)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID_COLOUR)
    ax.spines["bottom"].set_linewidth(1.1)

    legend = ax.legend(
        handles=[actual_line, comparison_line],
        loc="upper left",
        frameon=False,
        fontsize=19,
        handlelength=3.2,
        borderaxespad=0.2,
        labelspacing=0.8,
    )
    for label in legend.get_texts():
        label.set_color(TEXT_COLOUR)

    fig.text(
        0.985,
        0.022,
        "Source: S&P Dow Jones Indices LLC via FRED (SP500)",
        ha="right",
        va="bottom",
        fontsize=15,
        color=MUTED_TEXT,
    )
    fig.subplots_adjust(left=0.105, right=0.975, top=0.955, bottom=0.115)
    fig.savefig(output_file, facecolor=BACKGROUND, bbox_inches=None)
    plt.close(fig)


def draw_level_vs_change(
    dates: np.ndarray,
    actual: np.ndarray,
    ai_prediction: np.ndarray,
    output_file: Path,
) -> tuple[date, date]:
    """Make the failure visible locally, then establish it over all test days.

    The left panel uses the final 20 observations, a fixed close-up rather than
    a hand-picked failure. The right panel summarises every unseen day from
    2020--2024 in plain language: did the prediction get the direction right?
    """
    all_change_dates = dates[1:]
    all_actual_change = np.diff(actual)
    all_predicted_change = np.diff(ai_prediction)
    direction_correct = np.sign(all_actual_change) == np.sign(all_predicted_change)
    correct_per_100 = int(round(100 * float(direction_correct.mean())))
    always_up_per_100 = int(round(100 * float((all_actual_change > 0).mean())))

    change_dates = all_change_dates[-CLOSEUP_WINDOW:]
    actual_change = all_actual_change[-CLOSEUP_WINDOW:]
    predicted_change = all_predicted_change[-CLOSEUP_WINDOW:]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Avenir Next", "Avenir", "Helvetica Neue", "DejaVu Sans"],
            "font.size": 18,
            "axes.labelcolor": TEXT_COLOUR,
            "xtick.color": MUTED_TEXT,
            "ytick.color": MUTED_TEXT,
        }
    )

    fig, (change_ax, summary_ax) = plt.subplots(
        1, 2, figsize=(16, 9), dpi=160, gridspec_kw={"width_ratios": [1.22, 0.78]}
    )
    fig.patch.set_facecolor(BACKGROUND)
    for ax in (change_ax, summary_ax):
        ax.set_facecolor(BACKGROUND)

    change_ax.grid(axis="y", color=GRID_COLOUR, linewidth=1.0)
    change_ax.grid(axis="x", visible=False)
    change_ax.tick_params(axis="both", which="major", labelsize=16, length=0, pad=9)
    for side in ("top", "right", "left"):
        change_ax.spines[side].set_visible(False)
    change_ax.spines["bottom"].set_color(GRID_COLOUR)
    change_ax.spines["bottom"].set_linewidth(1.1)
    change_ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    change_ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    change_ax.axhline(0, color="#B9B2A3", linewidth=1.4, zorder=1)
    change_ax.plot(
        change_dates,
        actual_change,
        color=ACTUAL_COLOUR,
        linewidth=2.8,
        marker="o",
        markersize=5.5,
        label="What happened",
        zorder=2,
    )
    change_ax.plot(
        change_dates,
        predicted_change,
        color=AI_COLOUR,
        linewidth=2.8,
        linestyle=(0, (7, 4)),
        marker="x",
        markersize=7,
        markeredgewidth=2.1,
        label="AI prediction",
        zorder=3,
    )
    change_ax.set_title(
        "A closer look: the last 20 days",
        loc="left",
        fontsize=25,
        fontweight="semibold",
        color=TEXT_COLOUR,
        pad=22,
    )
    change_ax.set_ylabel("Change from the previous day", fontsize=18, labelpad=16)
    change_ax.yaxis.set_major_locator(plt.MaxNLocator(7))

    # Two plain-language bars make the baseline comparison visible without
    # requiring students to understand a technical performance metric.
    summary_ax.set_xlim(0, 100)
    summary_ax.set_ylim(-0.9, 2.15)
    summary_ax.axis("off")
    summary_ax.text(
        0, 1.96, "Across five years", ha="left", va="center",
        fontsize=25, fontweight="semibold", color=TEXT_COLOUR,
    )
    summary_ax.text(
        0, 1.66, "Correct directions out of every 100 days", ha="left", va="center",
        fontsize=16.5, color=MUTED_TEXT,
    )
    rows = [
        (0.95, "AI-generated model", correct_per_100, AI_COLOUR),
        (0.05, "Simple rule: always say ‘up’", always_up_per_100, BASELINE_COLOUR),
    ]
    for y, label, value, colour in rows:
        summary_ax.text(0, y + 0.32, label, ha="left", va="center",
                        fontsize=17, fontweight="semibold", color=TEXT_COLOUR)
        summary_ax.barh(y, 100, height=0.31, color="#EBE6DA", edgecolor="none")
        summary_ax.barh(y, value, height=0.31, color=colour, edgecolor="none")
        summary_ax.text(value + 2.2, y, f"{value}", ha="left", va="center",
                        fontsize=24, fontweight="bold", color=colour)
    summary_ax.text(
        50, -0.58, "The rule that ignores recent prices\ndoes better in this period.",
        ha="center", va="center", fontsize=15.5, color=MUTED_TEXT,
    )

    handles, labels = change_ax.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=2,
        frameon=False,
        fontsize=18,
        handlelength=3.0,
        columnspacing=2.2,
    )
    for label in legend.get_texts():
        label.set_color(TEXT_COLOUR)

    fig.text(
        0.5,
        0.045,
        "The long-term lines overlap—but a rule that ignores recent prices does better.",
        ha="center",
        va="bottom",
        fontsize=20,
        fontweight="semibold",
        color=TEXT_COLOUR,
    )
    fig.text(
        0.985,
        0.018,
        "Source: S&P Dow Jones Indices LLC via FRED (SP500)",
        ha="right",
        va="bottom",
        fontsize=13,
        color=MUTED_TEXT,
    )
    # Keep the shared legend, panel titles, and plots in separate visual bands.
    fig.subplots_adjust(left=0.085, right=0.975, top=0.78, bottom=0.15, wspace=0.28)
    fig.savefig(output_file, facecolor=BACKGROUND, bbox_inches=None)
    plt.close(fig)
    return change_dates[0], change_dates[-1]


def prepare_stock_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the fixed evaluation dates, observations, model output and baseline."""
    dates, values = load_series(DATA_FILE)
    coefficients = fit_pre_2020_model(dates, values)
    return build_evaluation_series(
        dates, values, coefficients
    )


def generate_level_comparison() -> None:
    """Task 1: generate the paired long-period level comparison."""
    eval_dates, actual, ai_prediction, yesterday = prepare_stock_data()
    y_limits = shared_y_limits(actual, ai_prediction, yesterday)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    draw_chart(
        eval_dates,
        actual,
        ai_prediction,
        "AI-generated prediction",
        AI_COLOUR,
        OUTPUT_DIR / "fig_ai_prediction.png",
        y_limits,
    )
    draw_chart(
        eval_dates,
        actual,
        yesterday,
        "Yesterday's closing level",
        BASELINE_COLOUR,
        OUTPUT_DIR / "fig_yesterday_baseline.png",
        y_limits,
    )

    print(f"Generated level comparison for {len(eval_dates)} observations.")
    print(f"Shared y-axis limits: {y_limits[0]:.2f} to {y_limits[1]:.2f}.")


def generate_direction_check() -> None:
    """Task 2: generate the close-up and full-period direction comparison."""
    eval_dates, actual, ai_prediction, _ = prepare_stock_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bridge_start, bridge_end = draw_level_vs_change(
        eval_dates,
        actual,
        ai_prediction,
        OUTPUT_DIR / "fig_level_vs_change.png",
    )

    print(
        f"Bridge figure uses the final {CLOSEUP_WINDOW} observations: "
        f"{bridge_start} to {bridge_end}."
    )


def main() -> None:
    """Compatibility entry point: generate both stock-evidence tasks."""
    generate_level_comparison()
    generate_direction_check()
    print("Output size: 2560 x 1440 pixels.")


if __name__ == "__main__":
    main()
