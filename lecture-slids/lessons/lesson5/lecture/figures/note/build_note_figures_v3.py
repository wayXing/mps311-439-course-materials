"""Lesson 5 Note: the two figures for the shrinkage section (breast lumps), and two slide figures (S8, S9).
Data: breast cancer, the Lesson 4 split (test_size=0.25, random_state=0, stratify), the same 5 folds.
Run (from this folder):  python build_note_figures_v3.py            -> F3_v3.svg, F4_v3.svg here (Chinese)
                         python build_note_figures_v3.py --lang en  -> the same files in ../note_en/
                         python build_note_figures_v3.py --slides   -> English, projector-size text, plus S8 and S9, in ../slides/
Colours (fixed across the course notes): data blue, model green, error red.
Classes: class 1 / malignant = filled blue dots, class 0 / benign = hollow grey circles.
"""
import os, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text
from scipy.stats import norm
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
SLIDES = "--slides" in os.sys.argv
EN = SLIDES or "en" in " ".join(os.sys.argv[1:]).split()
OUTDIR = os.path.join(HERE, "..", "slides" if SLIDES else "note_en") if EN else HERE

def T(zh, en):  # every visible string, in the language of the build
    return en if EN else zh
B, G, R = "#0072B2", "#00795A", "#C84B00"
GREY, LIGHT = "#6b6b6b", "#d9d9d9"
plt.rcParams.update({
    "font.family": ["DejaVu Sans", "Hiragino Sans GB", "Arial Unicode MS"],
    "svg.fonttype": "path", "axes.unicode_minus": False, "mathtext.fontset": "dejavusans",
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11, "figure.dpi": 100,
})
if SLIDES:  # projector: text roughly 1.5 times larger at the same figure size
    plt.rcParams.update({"font.size": 16, "axes.titlesize": 17, "axes.labelsize": 16,
                         "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14})

def save(fig, name):
    if SLIDES:  # projector: no label smaller than 14 pt
        for txt in fig.findobj(matplotlib.text.Text):
            if txt.get_text() and txt.get_fontsize() < 14:
                txt.set_fontsize(14)
    fig.savefig(os.path.join(OUTDIR, name + ".svg"), bbox_inches="tight")
    plt.close(fig)

d = load_breast_cancer()
X, y = d.data, 1 - d.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
kf = KFold(5, shuffle=True, random_state=0)
sc = StandardScaler().fit(Xtr)
Xs = sc.transform(Xtr)

# ---------------- F3_v3: radius and perimeter almost determine each other ----------------
mal = Xtr[ytr == 1]
ben = Xtr[ytr == 0]
r_mal = np.corrcoef(mal[:, 0], mal[:, 2])[0, 1]
fig, ax = plt.subplots(figsize=(5.6, 4.0))
ax.scatter(ben[:, 0], ben[:, 2], s=18, facecolors="none", edgecolors=GREY, linewidths=0.8, label=T("良性（训练集 267 个）", "benign (267 training lumps)"))
ax.scatter(mal[:, 0], mal[:, 2], s=16, color=B, label=T("恶性（训练集 159 个）", "malignant (159 training lumps)"))
ax.set_xlabel(T("平均半径", "mean radius"))
ax.set_ylabel(T("平均周长", "mean perimeter"))
ax.text(0.03, 0.95, T(f"恶性一类里两者的相关系数 {r_mal:.3f}\n点几乎排成一条线：\n垂直于这条线的方向几乎没有厚度",
          (f"malignant: correlation {r_mal:.3f}\nalmost no thickness across the line" if SLIDES else
           f"correlation within the malignant class: {r_mal:.3f}\nthe points almost form a line:\nacross the line there is almost no thickness")),
        transform=ax.transAxes, va="top", fontsize=10)
ax.legend(loc="lower right", frameon=False, fontsize=10)
save(fig, "F3_v3")

# ---------------- F4_v3: choosing reg_param by cross-validation ----------------
grid = [0.0001, 0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1]
scores = [cross_val_score(QuadraticDiscriminantAnalysis(reg_param=r), Xs, ytr, cv=kf).mean() for r in grid]
s0 = cross_val_score(QuadraticDiscriminantAnalysis(), Xs, ytr, cv=kf).mean()
s_lda = cross_val_score(LinearDiscriminantAnalysis(), Xs, ytr, cv=kf).mean()
s_lr = cross_val_score(LogisticRegression(), Xs, ytr, cv=kf).mean()
x0 = 0.00003  # where "no shrinking" is drawn on the log axis
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.axvspan(0.1, 0.3, color=LIGHT, alpha=0.6, lw=0)
ax.text(0.17, 0.925, T("过窄的\n搜索范围", "search range\ntoo narrow"), ha="center", va="bottom", fontsize=9, color=GREY)
ax.plot(grid, scores, "-o", color=G, ms=5, lw=1.6)
ax.plot([x0], [s0], "o", color=G, ms=5, mfc="white")
ax.annotate(T(f"不收缩 {s0:.3f}", f"no shrinkage {s0:.3f}"), (x0, s0), xytext=(4, -26 if SLIDES else -16), textcoords="offset points", fontsize=9)
best = int(np.argmax(scores))
ax.annotate(T(f"最好 {grid[best]:g}：{scores[best]:.3f}", f"best {grid[best]:g}: {scores[best]:.3f}"), (grid[best], scores[best]), xytext=(-30, 10), textcoords="offset points", fontsize=9)
ax.annotate(f"{scores[-1]:.3f}", (grid[-1], scores[-1]), xytext=(-12, -14), textcoords="offset points", fontsize=9)
ax.axhline(s_lr, color=GREY, ls="--", lw=1, label=f"logistic regression {s_lr:.3f}")
ax.axhline(s_lda, color=GREY, ls=":", lw=1.2, label=f"LDA {s_lda:.3f}")
if SLIDES:  # projector text is larger, so the two reference lines are named in a legend
    ax.legend(frameon=False, loc="lower left", fontsize=14)
else:
    ax.text(x0, s_lr + 0.001, f"logistic regression {s_lr:.3f}", fontsize=9, color=GREY, va="bottom")
    ax.text(0.002, s_lda + 0.0005, f"LDA {s_lda:.3f}", fontsize=9, color=GREY, va="bottom")
ax.set_xscale("log")
ax.set_xticks([x0, 0.0001, 0.001, 0.01, 0.1, 1])
ax.set_xticklabels(["0", "0.0001", "0.001", "0.01", "0.1", "1"])
ax.set_ylim(0.92, 0.99)
ax.set_xlabel(T("reg_param（往圆收缩的比例 r）", "reg_param (share r pulled towards a circle)"))
ax.set_ylabel(T("5 折交叉验证准确率", "5-fold cross-validation accuracy"))
save(fig, "F4_v3")

if SLIDES:
    rad_b = Xtr[ytr == 0, 0]
    rad_m = Xtr[ytr == 1, 0]
    mu_b, sd_b = rad_b.mean(), rad_b.std(ddof=1)
    mu_m, sd_m = rad_m.mean(), rad_m.std(ddof=1)
    pi_m = ytr.mean()
    pi_b = 1 - pi_m
    bins = np.arange(6, 30, 1.0)

    # ---------------- S8: the model can generate lumps ----------------
    rng = np.random.default_rng(0)
    fake_b = rng.normal(mu_b, sd_b, len(rad_b))   # as many imaginary lumps as real ones
    fake_m = rng.normal(mu_m, sd_m, len(rad_m))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 3.9), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, real, fake, col, title in [(axes[0], rad_b, fake_b, GREY, "benign"), (axes[1], rad_m, fake_m, B, "malignant")]:
        ax.hist(fake, bins=bins, color=col, alpha=0.25, label="drawn from the model")
        ax.hist(real, bins=bins, histtype="step", color=col, lw=2.0, label="real training lumps")
        ax.set_title(title, color=col)
        ax.set_xlabel("mean radius")
        ax.legend(frameon=False, loc="upper right")
    axes[0].set_ylabel("number of lumps")
    save(fig, "S8")

    # ---------------- S9: each training lump gets "share x height" ----------------
    xs = np.linspace(6, 29, 400)
    fig, ax = plt.subplots(figsize=(11.0, 4.0))
    ax.plot(xs, pi_b * norm.pdf(xs, mu_b, sd_b), color=GREY, lw=2.2, label=f"benign: P(Y=0) = {pi_b:.3f}, times its bell curve")
    ax.plot(xs, pi_m * norm.pdf(xs, mu_m, sd_m), color=B, lw=2.2, label=f"malignant: P(Y=1) = {pi_m:.3f}, times its bell curve")
    picks_b = np.sort(rad_b)[[30, 133, 240]]   # three benign lumps: small, typical, large
    picks_m = np.sort(rad_m)[[15, 80, 150]]    # three malignant lumps
    for xv in picks_b:
        h = pi_b * norm.pdf(xv, mu_b, sd_b)
        ax.plot([xv, xv], [0, h], color=GREY, lw=3)
        ax.plot([xv], [0], "o", ms=9, mfc="white", mec=GREY, mew=1.5, clip_on=False)
    for xv in picks_m:
        h = pi_m * norm.pdf(xv, mu_m, sd_m)
        ax.plot([xv, xv], [0, h], color=B, lw=3)
        ax.plot([xv], [0], "o", ms=9, color=B, clip_on=False)
    xv = picks_m[2]
    h = pi_m * norm.pdf(xv, mu_m, sd_m)
    ax.annotate(f"a malignant lump, radius {xv:.1f}:\n{pi_m:.3f} × {norm.pdf(xv, mu_m, sd_m):.3f} = {h:.3f}", (xv, h), xytext=(xv - 1.5, h + 0.035),
                ha="center", fontsize=14, arrowprops=dict(arrowstyle="-", color=B))
    ax.set_xlabel("mean radius")
    ax.set_ylabel("P(Y) × height")
    ax.set_ylim(0, 0.16)
    ax.legend(frameon=False, loc="upper right")
    save(fig, "S9")

print("corr", round(r_mal, 4), "no shrink", round(s0, 3), "LDA", round(s_lda, 3), "LR", round(s_lr, 3))
for r, s in zip(grid, scores):
    print(r, round(s, 3))
