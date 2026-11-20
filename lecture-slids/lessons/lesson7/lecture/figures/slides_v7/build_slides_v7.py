"""Lesson 7 Slides V7: the one figure not taken from the English Note.
S1  height against arm span, points only (the vote page before the line is drawn)
Run: uv run --no-project --python 3.12 --with matplotlib python build_slides_v7.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
N = json.load(open(os.path.join(HERE, "../note_v2/numbers_v2.json")))
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "svg.fonttype": "path", "font.size": 16,
                     "axes.spines.top": False, "axes.spines.right": False})
h, a = np.array(N["arm"]["h"]), np.array(N["arm"]["a"])
fig, ax = plt.subplots(figsize=(6.4, 6.0))
ax.scatter(h, a, s=22, color="#0072B2", alpha=0.75)
m = np.array([h.mean(), a.mean()])
ax.set_xlim(m[0] - 24, m[0] + 24); ax.set_ylim(m[1] - 24, m[1] + 24); ax.set_aspect("equal")
ax.set_xlabel("Height (cm)"); ax.set_ylabel("Arm span (cm)")
ax.set_title("200 people: keep one number each", fontsize=17)
fig.savefig(os.path.join(HERE, "S1.svg"), bbox_inches="tight")
print("slides ok")

# ---------------------------------------------------------------------------
# Model pages
# S2  one point is a column of d numbers: a person (2 x 1), a car (5 x 1), an image (784 x 1)
# S3  a direction is a column of length 1: u, 2u and -u lie on one line
# S4  keep one number: z = u^T x, as a row times a column, and as the foot of the perpendicular
# S5  K directions: x is rebuilt as z1 u1 + ... + zK uK, i.e. U z with U of size d x K
from matplotlib.patches import Rectangle, FancyArrowPatch
from sklearn.model_selection import train_test_split
import pandas as pd
B, G, R, O, GREY = "#0072B2", "#00795A", "#C84B00", "#E69F00", "#9a9a9a"
hc, ac = h - h.mean(), a - a.mean()
P = 6                                          # the person used on every model page
xp = np.array([hc[P], ac[P]])
u = np.array([0.689, 0.725]); u = u / np.linalg.norm(u)
zp = float(u @ xp)
LESSONS = os.path.abspath(os.path.join(HERE, "../../../.."))
mn = np.load(os.path.join(LESSONS, "data/mnist10k.npz"))
MX, MY = mn["X"] / 255.0, mn["y"].astype(int)
Xtr, _, ytr, _ = train_test_split(MX, MY, test_size=0.25, random_state=0, stratify=MY)
_, Xv, _, yv = train_test_split(Xtr, ytr, test_size=0.25, random_state=0, stratify=ytr)
six = Xv[5]
car = pd.read_csv(os.path.join(LESSONS, "lesson2/lab/auto_mpg.csv")).dropna().iloc[0]


def column(ax, x0, y0, values, labels=None, w=1.0, hcell=1.0, fmt="{:.1f}", color="#eef3f8", edge="#333", fs=19):
    for k, v in enumerate(values):
        y = y0 - k * hcell
        ax.add_patch(Rectangle((x0, y - hcell), w, hcell, fc=color, ec=edge, lw=1.5))
        ax.text(x0 + w / 2, y - hcell / 2, fmt.format(v) if not isinstance(v, str) else v, ha="center", va="center", fontsize=fs)
        if labels:
            ax.text(x0 - 0.15, y - hcell / 2, labels[k], ha="right", va="center", fontsize=fs - 3, color="#555")


# ---- S2 ----
fig = plt.figure(figsize=(16, 6.2))
ax = fig.add_axes([0.0, 0.08, 0.34, 0.84])
ax.scatter(hc, ac, s=14, color=GREY, alpha=0.6)
ax.axhline(0, color="#ccc", lw=1); ax.axvline(0, color="#ccc", lw=1)
ax.annotate("", xp, (0, 0), arrowprops=dict(arrowstyle="-|>", color=O, lw=3, mutation_scale=22))
ax.scatter(*xp, s=90, color=O, zorder=5)
ax.text(xp[0] - 14, xp[1] + 6, "one person", color=O, fontsize=18)
ax.set_xlim(-24, 24); ax.set_ylim(-24, 24); ax.set_aspect("equal")
ax.set_xlabel("height − mean (cm)", fontsize=16); ax.set_ylabel("arm span − mean (cm)", fontsize=16)
ax.set_title("centred: the mean is at 0", fontsize=18)
ax2 = fig.add_axes([0.36, 0.0, 0.64, 1.0]); ax2.set_xlim(0, 30); ax2.set_ylim(-1, 12); ax2.axis("off")
column(ax2, 3.2, 8.0, [xp[0], xp[1]], ["height", "arm span"], w=2.2, hcell=1.4)
ax2.text(4.3, 9.0, "x", fontsize=24, ha="center", weight="bold")
ax2.text(4.3, 4.3, "2 × 1", fontsize=19, ha="center", color=O, weight="bold")
vals = [car["cylinders"], car["displacement"], car["horsepower"], car["weight"], car["acceleration"]]
column(ax2, 13.0, 9.6, vals, ["cylinders", "displacement", "horsepower", "weight", "acceleration"], w=2.6, hcell=1.4, fmt="{:g}")
ax2.text(14.3, 10.6, "one car", fontsize=18, ha="center")
ax2.text(14.3, 1.6, "5 × 1", fontsize=19, ha="center", color=O, weight="bold")
ax4 = fig.add_axes([0.745, 0.50, 0.11, 0.30]); ax4.imshow(six.reshape(28, 28), cmap="Greys", vmin=0, vmax=1); ax4.axis("off")
ax2.text(23.6, 10.6, "one image", fontsize=18, ha="center")
ax2.text(23.6, 3.6, "row 1, then row 2,\n… then row 28", fontsize=15, ha="center", color="#555")
fig.patches.append(FancyArrowPatch((0.86, 0.65), (0.925, 0.65), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=22, color="#555", lw=2))
ax3 = fig.add_axes([0.94, 0.06, 0.022, 0.86])
ax3.imshow(six.reshape(784, 1), aspect="auto", cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
ax3.set_xticks([]); ax3.set_yticks([0, 783]); ax3.set_yticklabels(["1", "784"], fontsize=13)
ax2.text(29.9, -0.6, "784 × 1", fontsize=19, ha="center", color=O, weight="bold")
fig.savefig(os.path.join(HERE, "S2.svg"), bbox_inches="tight"); plt.close(fig)

# ---- S3 ----
fig, ax = plt.subplots(figsize=(7.2, 6.6))
t = np.linspace(0, 2 * np.pi, 300)
ax.plot(np.cos(t), np.sin(t), color="#bbb", lw=1.5)
ax.plot([-2.4 * u[0], 2.4 * u[0]], [-2.4 * u[1], 2.4 * u[1]], color=G, lw=1.2, ls=":")
for s, col, lab, off in ((2, GREY, "2u", (0.12, -0.18)), (-1, GREY, "−u", (-0.35, -0.1)), (1, G, "u", (0.12, -0.22))):
    ax.annotate("", s * u, (0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=3 if s == 1 else 2, mutation_scale=22, ls="-" if s == 1 else "--"))
    ax.text(*(s * u + np.array(off)), lab, fontsize=22, color=col, weight="bold")
ax.text(-2.3, 1.95, "length 1:  u₁² + u₂² = 1", fontsize=20)
ax.text(-2.3, 1.55, f"u = ({u[0]:.3f}, {u[1]:.3f})", fontsize=20, color=G)
ax.text(0.15, -1.65, "u, 2u and −u\ngive the same line", fontsize=18, color="#555")
ax.set_xlim(-2.4, 2.4); ax.set_ylim(-2.0, 2.4); ax.set_aspect("equal"); ax.axis("off")
fig.savefig(os.path.join(HERE, "S3.svg"), bbox_inches="tight"); plt.close(fig)

# ---- S4 ----
fig = plt.figure(figsize=(16, 6.4))
ax = fig.add_axes([0.0, 0.0, 0.5, 1.0]); ax.set_xlim(0, 20); ax.set_ylim(-2, 10); ax.axis("off")
for k, v in enumerate(u):
    ax.add_patch(Rectangle((0.5 + 2.4 * k, 6.0), 2.4, 1.4, fc="#e6f2ec", ec=G, lw=2)); ax.text(1.7 + 2.4 * k, 6.7, f"{v:.3f}", ha="center", va="center", fontsize=18)
ax.text(2.9, 7.9, "uᵀ   (1 × 2)", ha="center", fontsize=18, color=G, weight="bold")
ax.text(5.9, 6.7, "×", fontsize=26, va="center")
column(ax, 7.0, 8.1, [xp[0], xp[1]], w=2.2, hcell=1.4, color="#fdf1e2", edge=O)
ax.text(8.1, 8.6, "x   (2 × 1)", ha="center", fontsize=18, color=O, weight="bold")
ax.text(10.1, 6.7, "=", fontsize=26, va="center")
ax.add_patch(Rectangle((11.0, 6.0), 2.4, 1.4, fc="#e8f0fa", ec=B, lw=2)); ax.text(12.2, 6.7, f"{zp:.1f}", ha="center", va="center", fontsize=19, weight="bold")
ax.text(12.2, 7.9, "z   (1 × 1)", ha="center", fontsize=18, color=B, weight="bold")
ax.text(0.5, 3.4, f"z = {u[0]:.3f} × {xp[0]:.1f} + {u[1]:.3f} × {xp[1]:.1f} = {zp:.1f}", fontsize=22)
ax.text(0.5, 1.6, "d numbers in, one number out:", fontsize=21, color="#555")
ax.text(0.5, 0.2, "(1 × d) times (d × 1) is 1 × 1", fontsize=21, color="#555")
ax = fig.add_axes([0.55, 0.05, 0.42, 0.9])
ax.scatter(hc, ac, s=10, color=GREY, alpha=0.35)
L = 20; ax.plot([-L * u[0], L * u[0]], [-L * u[1], L * u[1]], color=G, lw=2)
ax.annotate("", u * 5, (0, 0), arrowprops=dict(arrowstyle="-|>", color=G, lw=3, mutation_scale=20)); ax.text(*(u * 5 + [-3.2, 0.6]), "u", color=G, fontsize=20, weight="bold")
foot = zp * u
ax.plot(*zip(xp, foot), color=R, lw=2.5)
nrm = np.array([-u[1], u[0]]) * 0.9; sq = [foot + u * 0.9, foot + u * 0.9 - nrm * np.sign(np.dot(xp - foot, -nrm)) * -1, foot - nrm * np.sign(np.dot(xp - foot, -nrm)) * -1]
ax.plot([foot[0] - u[0] * 0.9, foot[0] - u[0] * 0.9 + (xp - foot)[0] / np.linalg.norm(xp - foot) * 0.9, foot[0] + (xp - foot)[0] / np.linalg.norm(xp - foot) * 0.9],
        [foot[1] - u[1] * 0.9, foot[1] - u[1] * 0.9 + (xp - foot)[1] / np.linalg.norm(xp - foot) * 0.9, foot[1] + (xp - foot)[1] / np.linalg.norm(xp - foot) * 0.9], color=R, lw=1.5)
ax.scatter(*xp, s=90, color=O, zorder=5); ax.text(xp[0] + 0.8, xp[1] - 0.2, "x", color=O, fontsize=20, weight="bold")
ax.scatter(*foot, s=90, color=B, zorder=5); ax.text(foot[0] - 4.2, foot[1] + 0.9, "z u", color=B, fontsize=20, weight="bold")
off = np.array([u[1], -u[0]]) * 1.6
ax.annotate("", foot + off, off, arrowprops=dict(arrowstyle="<->", color=B, lw=1.8))
ax.text(*(foot / 2 + off * 2.2 + [0.3, -0.4]), f"z = {zp:.1f}", color=B, fontsize=22)
ax.set_xlim(-6, 16); ax.set_ylim(-6, 16); ax.set_aspect("equal"); ax.axis("off")
fig.savefig(os.path.join(HERE, "S4.svg"), bbox_inches="tight"); plt.close(fig)

# ---- S5 ----
fig, ax = plt.subplots(figsize=(16, 6.0)); ax.set_xlim(0, 32); ax.set_ylim(-1.2, 11.5); ax.axis("off")
def tall(x0, lab, col, sub, width=1.3, top=10, height=9):
    ax.add_patch(Rectangle((x0, top - height), width, height, fc=col, ec="#333", lw=1.5))
    ax.text(x0 + width / 2, top + 0.5, lab, ha="center", fontsize=24)
    ax.text(x0 + width / 2, top - height - 0.8, sub, ha="center", fontsize=18, color="#555")
tall(0.5, "$x$", "#fdf1e2", "d × 1")
ax.text(2.4, 5.5, "≈", fontsize=30, va="center")
xs = 3.6
for k, lab in enumerate(["1", "2"]):
    ax.text(xs, 5.5, f"$z_{k+1}$", fontsize=28, va="center", color=B)
    tall(xs + 1.4, f"$u_{k+1}$", "#e6f2ec", "d × 1")
    ax.text(xs + 3.0, 5.5, "+", fontsize=28, va="center"); xs += 4.4
ax.text(xs, 5.5, "…  +", fontsize=26, va="center"); xs += 2.6
ax.text(xs, 5.5, "$z_K$", fontsize=28, va="center", color=B)
tall(xs + 1.8, "$u_K$", "#e6f2ec", "d × 1")
ax.text(xs + 4.0, 5.5, "=", fontsize=30, va="center")
x0 = xs + 5.2
ax.add_patch(Rectangle((x0, 1), 4.2, 9, fc="#e6f2ec", ec="#333", lw=1.5))
for k in range(1, 4):
    ax.plot([x0 + 1.05 * k, x0 + 1.05 * k], [1, 10], color="#333", lw=0.8)
ax.text(x0 + 2.1, 10.5, "$U$", ha="center", fontsize=24); ax.text(x0 + 2.1, 0.2, "d × K", ha="center", fontsize=18, color="#555")
ax.add_patch(Rectangle((x0 + 4.8, 6.0), 1.0, 4.0, fc="#e8f0fa", ec="#333", lw=1.5))
ax.text(x0 + 5.3, 10.5, "$z$", ha="center", fontsize=24, color=B); ax.text(x0 + 5.3, 5.2, "K × 1", ha="center", fontsize=18, color="#555")
fig.savefig(os.path.join(HERE, "S5.svg"), bbox_inches="tight"); plt.close(fig)
print("model pages ok", round(zp, 2), xp.round(1))
