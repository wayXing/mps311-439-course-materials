"""Lesson 6 Slides: English, projector-size figures. Reads numbers_v4.json; writes ../slides_v2/.
F1 Titanic depth-2 tree; F2a training and CV only; F2 with the five validation splits; F6 the investigation;
S1 two hand-written rule cards; S2 three impurity curves; S3 the loss of a cut is a staircase; S4 one passenger's path.
Survived = blue, died = grey, model green (as in the Note figures)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "..", "slides_v2")
N = json.load(open(os.path.join(HERE, "numbers_v4.json")))
B, G, R, GREY, LIGHT = "#0072B2", "#00795A", "#C84B00", "#6b6b6b", "#d9d9d9"
plt.rcParams.update({"font.family": ["DejaVu Sans"], "svg.fonttype": "path", "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 16, "axes.titlesize": 17,
                     "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14})
def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.svg"), bbox_inches="tight")
    if os.environ.get("PNGDIR"):
        fig.savefig(os.path.join(os.environ["PNGDIR"], f"s_{name}.png"), bbox_inches="tight", dpi=60)
    plt.close(fig)
def qbox(ax, x, y, text, sub=None, w=2.9, hi=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - 0.3), w, 0.6, boxstyle="round,pad=0.04", fc="#eef3f8", ec=B, lw=3 if hi else 1.4))
    ax.text(x, y + (0.08 if sub else 0), text, ha="center", va="center", fontsize=16, weight="bold")
    if sub: ax.text(x, y - 0.16, sub, ha="center", va="center", fontsize=13, color="#444")
def leaf(ax, x, y, text, sub, good, w=2.4, hi=False, faded=False):
    col = B if good else GREY
    ax.add_patch(FancyBboxPatch((x - w / 2, y - 0.3), w, 0.6, boxstyle="round,pad=0.04", fc="white", ec=col, lw=3.2 if hi else 1.8, alpha=0.45 if faded else 1))
    ax.text(x, y + (0.09 if sub else 0), text, ha="center", va="center", fontsize=15, color=col, weight="bold", alpha=0.45 if faded else 1)
    ax.text(x, y - 0.15, sub, ha="center", va="center", fontsize=13, color="#444", alpha=0.45 if faded else 1)
def arrow(ax, x0, y0, x1, y1, lab, hi=False):
    ax.annotate("", xy=(x1, y1 + 0.32), xytext=(x0, y0 - 0.3), arrowprops=dict(arrowstyle="->", color=B if hi else "#888", lw=3 if hi else 1.2))
    ax.text((x0 + x1) / 2 + (-0.18 if lab == "yes" else 0.18), (y0 + y1) / 2 + 0.05, lab, ha="center", fontsize=14, color="#333")

# F1
nodes = N["t2"]["nodes"]; q = {"sex": "Male?", "age": "Age ≤ 13?", "pclass": "1st or 2nd class?"}
pos = {0: (5, 3), 1: (2.6, 1.7), 4: (7.4, 1.7), 2: (1.15, 0.45), 3: (3.75, 0.45), 5: (6.25, 0.45), 6: (8.85, 0.45)}
fig, ax = plt.subplots(figsize=(13, 5.4)); ax.axis("off"); ax.set_xlim(-0.4, 10.4); ax.set_ylim(-0.2, 3.7)
for nd in nodes:
    x, y = pos[nd["i"]]; s, d = nd["surv"], nd["n"] - nd["surv"]
    if nd["left"] >= 0:
        qbox(ax, x, y, q[nd["feat"]], f"{s} : {d}")
        for ch, lab in [(nd["left"], "yes"), (nd["right"], "no")]:
            arrow(ax, x, y, *pos[ch], lab)
    else:
        leaf(ax, x, y, "survived" if s > d else "died", f"{s} : {d}", s > d)
ax.text(-0.3, 3.5, "numbers: survived : died (535 training passengers)", fontsize=13, color="#444")
save(fig, "F1")

# F2a and F2
rows = N["sweep"]["rows"]; dep = [r["d"] for r in rows]
for name, full in [("F2a", False), ("F2", True)]:
    fig, ax = plt.subplots(figsize=(10, 5.2))
    if full:
        for k, sp in enumerate(N["valsplits"]["splits"]):
            ax.plot(dep, sp["acc"], "-", color=G, lw=1.4, alpha=0.5, label="5 validation splits (134 each)" if k == 0 else None)
            ax.scatter([sp["best"]], [sp["acc"][sp["best"] - 1]], color=G, s=50, zorder=5)
    ax.plot(dep, [r["train"] for r in rows], "o-", color=B, lw=3, ms=7, label="training (535)")
    ax.plot(dep, [r["cv"] for r in rows], "s-", color="#333", lw=3, ms=7, label="5-fold cross-validation")
    ax.set(xlabel="max depth of the tree", ylabel="Accuracy", xticks=dep, ylim=(0.68, 1.0))
    ax.legend(frameon=False, loc="upper left")
    save(fig, name)

# F6
inv = N["inv"]["drops"]; full = N["best"]["imp"]
nm = {"sex": "sex", "pclass": "class", "age": "age", "fare": "fare", "sibsp": "siblings/spouse", "parch": "parents/children"}
panels = [(full, f"all 6 features ({N['best']['cv']:.3f})"), (inv["pclass"]["imp"], f"without class ({inv['pclass']['cv']:.3f})"),
          (inv["sex"]["imp"], f"without sex ({inv['sex']['cv']:.3f})")]
fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), sharex=True, gridspec_kw=dict(wspace=0.75))
order = ["sex", "pclass", "age", "fare", "sibsp", "parch"]
for ax, (imp, title) in zip(axes, panels):
    keys = [k for k in order if k in imp]
    ax.barh([nm[k] for k in keys], [imp[k] for k in keys], color=[G if k == "fare" else LIGHT for k in keys], height=0.6)
    for j, k in enumerate(keys):
        ax.text(imp[k] + 0.01, j, f"{imp[k]:.3f}", va="center", fontsize=13)
    ax.set(xlim=(0, 0.78), title=title, xlabel="feature importance")
    ax.invert_yaxis()
save(fig, "F6")

# S1: two rule cards
fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw=dict(wspace=0.12))
cards = [("A loan officer", "Income < 20k?", ("high risk", False), "Debt > half of income?", ("high risk", False), ("low risk", True)),
         ("Lifeboats, 1912", "A woman?", ("to the boat", True), "A child?", ("to the boat", True), ("wait", False))]
for ax, (title, q1, l1, q2, l2, l3) in zip(axes, cards):
    ax.axis("off"); ax.set_xlim(0.2, 8.4); ax.set_ylim(-0.2, 3.6); ax.set_title(title, fontsize=18, weight="bold")
    qbox(ax, 4, 2.9, q1, w=3.6); leaf(ax, 1.7, 1.6, l1[0], "", l1[1], w=2.5)
    arrow(ax, 4, 2.9, 1.7, 1.6, "yes"); arrow(ax, 4, 2.9, 5.6, 1.6, "no")
    qbox(ax, 5.6, 1.6, q2, w=4.2 if len(q2) > 12 else 3.0)
    leaf(ax, 4.0, 0.3, l2[0], "", l2[1], w=2.5); leaf(ax, 7.0, 0.3, l3[0], "", l3[1], w=2.3)
    arrow(ax, 5.6, 1.6, 4.0, 0.3, "yes"); arrow(ax, 5.6, 1.6, 7.0, 0.3, "no")
save(fig, "S1")

# S2: three ways to score a side with share p
p = np.linspace(0.0005, 0.9995, 400)
fig, ax = plt.subplots(figsize=(8.6, 5))
ax.plot(p, np.minimum(p, 1 - p), color=GREY, lw=3, label="error rate  min(p, 1−p)")
ax.plot(p, 2 * p * (1 - p), color=G, lw=3.4, label="Gini  2p(1−p)")
ax.plot(p, -(p * np.log2(p) + (1 - p) * np.log2(1 - p)) / 2, color=B, lw=2.4, ls="--", label="entropy (halved)")
ax.set(xlabel="share of class 1 on one side, p", ylabel="how mixed", xlim=(0, 1), ylim=(0, 0.56))
ax.legend(frameon=False, loc="lower center", fontsize=13)
save(fig, "S2")

# S3: the loss of one cut is a staircase
S = N["stair"]
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.plot(S["grid"], S["wgini"], color=G, lw=3, drawstyle="steps-post")
ages, labs = np.array(S["ages"]), np.array(S["labels"])
ymin = min(S["wgini"]) - 0.03
ax.scatter(ages[labs == 1], np.full((labs == 1).sum(), ymin), color=B, s=60, zorder=5, label="survived")
ax.scatter(ages[labs == 0], np.full((labs == 0).sum(), ymin), facecolor="none", edgecolor=GREY, s=60, lw=1.6, zorder=5, label="died")
ax.set(xlabel="cut on age (30 men)", ylabel="weighted Gini", xlim=(0, 75))
ax.legend(frameon=False, loc="upper right")
save(fig, "S3")

# S4: the depth-3 tree with one passenger's path
T = N["t3"]["nodes"]; path = set(N["t3"]["path"])
qq = {"sex": lambda t: "Male?", "age": lambda t: f"Age ≤ {t:g}?", "pclass": lambda t: "1st class?" if t < 2 else "1st or 2nd class?",
      "sibsp": lambda t: f"Siblings/spouse ≤ {int(t)}?", "fare": lambda t: f"Fare ≤ {t:g}?"}
def place(i, x, y, dx, out):
    out[i] = (x, y)
    if T[i]["left"] >= 0:
        place(T[i]["left"], x - dx, y - 1.25, dx / 2, out); place(T[i]["right"], x + dx, y - 1.25, dx / 2, out)
P = {}; place(0, 8, 3.9, 4, P)
fig, ax = plt.subplots(figsize=(17, 6.2)); ax.axis("off"); ax.set_xlim(-0.3, 16.3); ax.set_ylim(-0.4, 4.4)
for nd in T:
    x, y = P[nd["i"]]; s, d = nd["surv"], nd["n"] - nd["surv"]; hi = nd["i"] in path
    if nd["left"] >= 0:
        qbox(ax, x, y, qq[nd["feat"]](nd["thr"]), f"{s} : {d}", w=3.3 if y > 2 else 2.8, hi=hi)
        for ch, lab in [(nd["left"], "yes"), (nd["right"], "no")]:
            arrow(ax, x, y, *P[ch], lab, hi=hi and ch in path)
    else:
        leaf(ax, x, y, "survived" if s > d else "died", f"{s} : {d}", s > d, w=1.75, hi=hi, faded=not hi)
ax.text(-0.2, 4.3, "Third class, woman, 24, fare 16.7", fontsize=15, color=B, weight="bold")
save(fig, "S4")

# ---------- visual-first pages (teacher 2026-11-05: more figures, less text) ----------
X2 = json.load(open(os.path.join(HERE, "..", "note_v2", "numbers_v2.json")))["xor"]
xx, yy2, ll = np.array(X2["x"]), np.array(X2["y"]), np.array(X2["label"])
def xor_ax(ax):
    ax.scatter(xx[ll == 1], yy2[ll == 1], color=B, s=22, alpha=0.85)
    ax.scatter(xx[ll == 0], yy2[ll == 0], facecolor="none", edgecolor=GREY, s=20, lw=1)
    ax.set(xlim=(-1, 6), ylim=(-1, 6), xticks=[], yticks=[], xlabel="$x_1$", ylabel="$x_2$")
# S5: three cuts in the plane = a tree of three questions
fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw=dict(width_ratios=[1, 1.5], wspace=0.05))
ax = axes[0]; xor_ax(ax)
for (x0, x1, y0, y1, c) in [(-1, 2.5, -1, 2.5, 0), (-1, 2.5, 2.5, 6, 1), (2.5, 6, -1, 2.5, 1), (2.5, 6, 2.5, 6, 0)]:
    if c: ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, color=G, alpha=0.13, lw=0))
ax.plot([2.5, 2.5], [-1, 6], color="#222", lw=2.6); ax.plot([-1, 2.5], [2.5, 2.5], color="#222", lw=2.2); ax.plot([2.5, 6], [2.5, 2.5], color="#222", lw=2.2)
ax.text(2.6, 5.6, "cut 1", fontsize=14); ax.text(-0.9, 2.65, "cut 2", fontsize=14); ax.text(4.9, 2.65, "cut 3", fontsize=14)
ax.set_title("accuracy 0.988", fontsize=16)
ax = axes[1]; ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(-0.2, 3.6)
qbox(ax, 5, 3, "$x_1 \\leq 2.5$?", w=2.6); qbox(ax, 2.6, 1.7, "$x_2 \\leq 2.5$?", w=2.6); qbox(ax, 7.4, 1.7, "$x_2 \\leq 2.5$?", w=2.6)
arrow(ax, 5, 3, 2.6, 1.7, "yes"); arrow(ax, 5, 3, 7.4, 1.7, "no")
for x, c, px in [(1.3, 0, 2.6), (3.9, 1, 2.6), (6.1, 1, 7.4), (8.7, 0, 7.4)]:
    leaf(ax, x, 0.4, f"class {c}", "", c == 1, w=2.0); arrow(ax, px, 1.7, x, 0.4, "yes" if x in (1.3, 6.1) else "no")
save(fig, "S5")

# F1m: the tree with the two assumptions marked on it
nodes = N["t2"]["nodes"]
fig, ax = plt.subplots(figsize=(14, 6.0)); ax.axis("off"); ax.set_xlim(-0.6, 11.6); ax.set_ylim(-0.75, 3.7)
for nd in nodes:
    x, y = pos[nd["i"]]; s, d = nd["surv"], nd["n"] - nd["surv"]
    if nd["left"] >= 0:
        qbox(ax, x, y, q[nd["feat"]], f"{s} : {d}")
        for ch, lab in [(nd["left"], "yes"), (nd["right"], "no")]:
            arrow(ax, x, y, *pos[ch], lab)
    else:
        leaf(ax, x, y, "survived" if s > d else "died", f"{s} : {d}", s > d)
ax.annotate("one feature,\none number", xy=(8.85, 1.7), xytext=(10.1, 2.6), fontsize=15, color=R, ha="center",
            arrowprops=dict(arrowstyle="->", color=R, lw=1.8))
ax.annotate("115 / 123 = 0.935\na probability", xy=(6.25, 0.15), xytext=(6.25, -0.6), fontsize=15, color=R, ha="center", va="center",
            arrowprops=dict(arrowstyle="->", color=R, lw=1.8))
ax.text(-0.5, 3.5, "survived : died, 535 training passengers", fontsize=13, color="#444")
save(fig, "F1m")

# S7: the first cut as bars
rows7 = [("all 535", 217, 318, 0.482), ("male, 333", 70, 263, 0.332), ("female, 202", 147, 55, 0.396)]
fig, ax = plt.subplots(figsize=(11, 4.0))
for k, (lab, s, d, g) in enumerate(rows7):
    yv = 2 - k - (0.35 if k else 0)
    ax.barh(yv, s / (s + d), color=B, height=0.55); ax.barh(yv, d / (s + d), left=s / (s + d), color=LIGHT, height=0.55)
    ax.text(-0.02, yv, lab, ha="right", va="center", fontsize=16)
    ax.text(1.03, yv, f"Gini {g:.3f}", va="center", fontsize=17, color=G, weight="bold")
ax.annotate("", xy=(0.5, 1.0), xytext=(0.5, 1.68), arrowprops=dict(arrowstyle="->", lw=2, color="#555"))
ax.text(0.52, 1.36, "cut on sex", fontsize=14, color="#555")
ax.text(1.03, 0.1, "weighted 0.356", fontsize=15, color=G)
ax.set_xlim(0, 1); ax.axis("off")
ax.text(0.0, 2.45, "survived", color=B, fontsize=14); ax.text(0.86, 2.45, "died", color=GREY, fontsize=14)
save(fig, "S7")

# S8: importance = each feature's share of the Gini taken away
dec = N["t2"]["dec"]; take = {"sex": 67.3, "pclass": 27.0, "age": 10.7}
fig, ax = plt.subplots(figsize=(12, 2.6)); left = 0
cols8 = {"sex": G, "pclass": "#4f9e86", "age": "#9fcbbd"}
for k in ["sex", "pclass", "age"]:
    ax.barh(0, take[k], left=left, color=cols8[k], height=0.6, edgecolor="white", lw=2)
    ax.text(left + take[k] / 2, 0, f"{nm[k]}\n{take[k]:.1f}", ha="center", va="center", fontsize=15, color="white" if k != "age" else "#123", weight="bold")
    ax.text(left + take[k] / 2, -0.52, f"{take[k] / 105.0:.3f}", ha="center", fontsize=16, color="#222")
    left += take[k]
ax.text(0, 0.48, "Gini taken away by each feature's cuts (count × Gini), 105.0 in all", fontsize=14, color="#444")
ax.text(-1, -0.52, "importance", ha="right", fontsize=14, color="#444")
ax.set_xlim(0, 105); ax.set_ylim(-0.75, 0.7); ax.axis("off")
save(fig, "S8")

# S9: iris, petal length against petal width, with the two cuts
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
ir = load_iris(); Xi_tr, _, yi_tr, _ = train_test_split(ir.data, ir.target, test_size=0.25, random_state=0, stratify=ir.target)
fig, ax = plt.subplots(figsize=(8.6, 5.4))
for k, (c, nme, mk) in enumerate([(GREY, "setosa", "o"), (B, "versicolor", "s"), (R, "virginica", "^")]):
    m = yi_tr == k
    ax.scatter(Xi_tr[m, 2], Xi_tr[m, 3], color=c, marker=mk, s=40, alpha=0.85, label=nme)
ax.axhline(0.8, color="#222", lw=2.4); ax.plot([0.8, 7.2], [1.75, 1.75], color="#222", lw=2.2)
ax.text(0.9, 0.86, "petal width ≤ 0.80?", ha="left", fontsize=14); ax.text(0.9, 1.81, "petal width ≤ 1.75?", ha="left", fontsize=14)
ax.set(xlabel="petal length (cm)", ylabel="petal width (cm)", xlim=(0.8, 7.2), ylim=(0, 2.7))
ax.legend(frameon=False, loc="upper left")
save(fig, "S9")

# S10: the tree next to a straight line, same 5-fold CV
acc = [("four clusters", 0.487, 0.975), ("Titanic", 0.783, 0.822), ("iris", 0.947, 0.955), ("breast lumps", 0.979, 0.927)]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw=dict(width_ratios=[2.4, 1], wspace=0.45))
ax = axes[0]
for k, (lab, a, b) in enumerate(acc):
    ax.plot([a, b], [k, k], color="#bbb", lw=3, zorder=1)
    ax.scatter([a], [k], color=GREY, s=150, zorder=3, label="straight line" if k == 0 else None)
    ax.scatter([b], [k], color=G, s=150, zorder=3, marker="D", label="tree" if k == 0 else None)
ax.set_yticks(range(4)); ax.set_yticklabels([a[0] for a in acc]); ax.invert_yaxis()
ax.set(xlim=(0.45, 1.0), xlabel="accuracy (5-fold CV)"); ax.legend(frameon=False, loc="lower left")
ax = axes[1]
ax.bar(["straight\nline", "tree"], [3.56, 3.22], color=[GREY, G], width=0.6)
for k, v in enumerate([3.56, 3.22]): ax.text(k, v + 0.05, f"{v:.2f}", ha="center", fontsize=15)
ax.set(ylabel="RMSE (mpg)", ylim=(0, 4.2), title="Auto MPG, all 6 features")
save(fig, "S10")

# S11: four limits, one small picture each
fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), gridspec_kw=dict(wspace=0.35))
ax = axes[0]; rs = np.random.default_rng(3); P = rs.uniform(0, 1, (160, 2)); lab = (P[:, 0] + P[:, 1] > 1).astype(int)
ax.scatter(P[lab == 1, 0], P[lab == 1, 1], color=B, s=12); ax.scatter(P[lab == 0, 0], P[lab == 0, 1], facecolor="none", edgecolor=GREY, s=12)
st = np.linspace(0, 1, 7); xs_ = np.repeat(st, 2)[1:]; ys_ = np.repeat(1 - st, 2)[:-1]
ax.plot(xs_, ys_, color=G, lw=2.4); ax.plot([0, 1], [1, 0], color="#222", lw=1.2, ls="--")
ax.set(xticks=[], yticks=[], title="slanted boundary:\nmany steps")
ax = axes[1]; xor_ax(ax); ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title("equal clusters:\ngreedy finds no first cut")
ax = axes[2]; cnt = {"age ≤ 3.5": 5, "age ≤ 13": 2, "age ≤ 8.5": 1, "fare": 2}
ax.barh(list(cnt), list(cnt.values()), color=G, height=0.6); ax.invert_yaxis()
ax.set(xlabel="of 10 resamples", title="new sample, new tree:\nthe cut for men")
ax = axes[3]; ax.bar(["fare, all\nfeatures", "fare, without\nclass"], [0.045, 0.155], color=[LIGHT, G], width=0.6)
ax.set(ylim=(0, 0.2), title="importance:\nshared by look-alikes")
for k, v in enumerate([0.045, 0.155]): ax.text(k, v + 0.005, f"{v:.3f}", ha="center", fontsize=14)
save(fig, "S11")

# S12: with fare and without fare, fold by fold
base = np.array(N["inv"]["base_folds"]); wo = base + np.array(N["inv"]["drops"]["fare"]["diff"])
fig, ax = plt.subplots(figsize=(10, 4.8))
for k in range(5):
    ax.plot([k + 1 - 0.12, k + 1 + 0.12], [base[k], wo[k]], color="#bbb", lw=2, zorder=1)
ax.scatter(np.arange(1, 6) - 0.12, base, color=G, s=120, zorder=3, label=f"with fare (mean {base.mean():.3f})")
ax.scatter(np.arange(1, 6) + 0.12, wo, color=GREY, s=120, zorder=3, marker="D", label=f"without fare (mean {wo.mean():.3f})")
ax.set(xticks=range(1, 6), xticklabels=[f"fold {k}" for k in range(1, 6)], ylabel="Accuracy", ylim=(0.76, 0.9), xlim=(0.4, 5.6))
ax.legend(frameon=False, loc="upper right")
save(fig, "S12")

# S13: fares differ a lot by class; without class, fare takes over
fbc = N["inv"]["fare_by_class"]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(wspace=0.4))
ax = axes[0]; ax.bar(["1st", "2nd", "3rd"], [fbc["1"], fbc["2"], fbc["3"]], color=[B, "#5b9bc8", "#a9c9e3"], width=0.6)
for k, v in enumerate([fbc["1"], fbc["2"], fbc["3"]]): ax.text(k, v + 1.5, f"{v:.1f}", ha="center", fontsize=15)
ax.set(ylabel="median fare", title="fare by class", ylim=(0, 78))
ax = axes[1]; vals = [N["best"]["imp"]["fare"], N["inv"]["drops"]["pclass"]["imp"]["fare"]]
ax.bar(["with class", "without class"], vals, color=[LIGHT, G], width=0.6)
for k, v in enumerate(vals): ax.text(k, v + 0.005, f"{v:.3f}", ha="center", fontsize=15)
ax.set(ylabel="importance of fare", title="fare's importance (accuracy 0.822 → 0.783)", ylim=(0, 0.2))
save(fig, "S13")
print("visual pages ok")
