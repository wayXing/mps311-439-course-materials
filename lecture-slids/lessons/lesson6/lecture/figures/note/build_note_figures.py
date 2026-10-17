"""Lesson 6 Note: static figures. Reads numbers.json written by build_numbers.py.
Run (from this folder):  python build_note_figures.py            -> Chinese figures here
                         python build_note_figures.py --lang en  -> English figures in ../note_en/
                         python build_note_figures.py --slides   -> English, projector-size text in ../slides/
Colours (fixed across the course notes): data blue, model green, error red.
Classes: class 1 / malignant = filled blue dots, class 0 / benign = hollow grey circles.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text
from matplotlib.patches import Rectangle, FancyBboxPatch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
SLIDES = "--slides" in os.sys.argv
EN = SLIDES or "en" in " ".join(os.sys.argv[1:]).split()  # also accepts "--lang en" passed as one word
OUTDIR = os.path.join(HERE, "..", "slides" if SLIDES else "note_en") if EN else HERE
os.makedirs(OUTDIR, exist_ok=True)
if EN:  # every visible string goes through the translation table; leftover Chinese stops the build
    from i18n_en import tr, check
    _set_text = matplotlib.text.Text.set_text
    matplotlib.text.Text.set_text = lambda self, s: _set_text(self, check(tr(str(s)), "figure text") if s is not None else s)
N = json.load(open(os.path.join(HERE, "numbers.json")))
B, G, R = "#0072B2", "#00795A", "#C84B00"
YACC = "Accuracy" if EN else "准确率"  # axis label, capitalised in English
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
        for t in fig.findobj(matplotlib.text.Text):
            if t.get_text() and t.get_fontsize() < 14:
                t.set_fontsize(14)
    fig.savefig(os.path.join(OUTDIR, f"{name}.svg"), bbox_inches="tight")
    if os.environ.get("PNGDIR"):  # optional PNG copies for checking the figures by eye
        fig.savefig(os.path.join(os.environ["PNGDIR"], f"{name}{'_en' if EN and not SLIDES else ''}{'_s' if SLIDES else ''}.png"),
                    bbox_inches="tight", dpi=90)
    plt.close(fig)


def classes(ax, xs, ys, labels, size=16, names=("良性", "恶性")):
    ax.scatter(xs[labels == 0], ys[labels == 0], s=size, facecolor="none", edgecolor=GREY, lw=0.8, label=names[0])
    ax.scatter(xs[labels == 1], ys[labels == 1], s=size, color=B, alpha=0.75, lw=0, label=names[1])


C2 = N["bc2"]
tx, ty, tl = np.array(C2["train_x"]), np.array(C2["train_y"]), np.array(C2["train_label"])

# ---------------- F1: a depth-2 tree on two features, drawn as questions ----------------
d = load_breast_cancer()
y = (d.target == 0).astype(int)
Xtr, Xte, ytr, yte = train_test_split(d.data, y, test_size=0.25, random_state=0, stratify=y)
Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.25, random_state=0, stratify=ytr)
t2 = DecisionTreeClassifier(max_depth=2, random_state=0).fit(Xa[:, [0, 1]], ya)
T = t2.tree_
fname = {0: "半径", 1: "纹理"}
cl, cr = T.children_left, T.children_right
pos = {0: (5, 3), cl[0]: (2.6, 1.7), cr[0]: (7.4, 1.7)}
for xx, nd in zip([1.15, 3.75, 6.25, 8.85], [cl[cl[0]], cr[cl[0]], cl[cr[0]], cr[cr[0]]]):
    pos[nd] = (xx, 0.45)
fig, ax = plt.subplots(figsize=(10.4, 4.6))
ax.axis("off")
ax.set_xlim(-0.3, 10.3)
ax.set_ylim(-0.25, 3.7)


def node_text(i):
    mal, ben = int(round(T.value[i][0][1] * T.n_node_samples[i])), int(round(T.value[i][0][0] * T.n_node_samples[i]))
    return mal, ben


kids = {}
for i in range(T.node_count):
    if T.children_left[i] >= 0:
        kids[T.children_left[i]] = (i, "是")
        kids[T.children_right[i]] = (i, "否")
for i in range(T.node_count):
    mal, ben = node_text(i)
    if T.children_left[i] >= 0:
        x, yy = pos[i]
        ax.add_patch(FancyBboxPatch((x - 1.35, yy - 0.28), 2.7, 0.56, boxstyle="round,pad=0.04", fc="#eef3f8", ec=B, lw=1.4))
        ax.text(x, yy + 0.07, f"{fname[T.feature[i]]} ≤ {T.threshold[i]:.2f}？", ha="center", va="center", fontsize=12, weight="bold")
        ax.text(x, yy - 0.15, f"{mal} : {ben}", ha="center", va="center", fontsize=10.5, color="#444")
    else:
        x, yy = pos[i]
        col = B if mal > ben else GREY
        ax.add_patch(FancyBboxPatch((x - 1.15, yy - 0.3), 2.3, 0.6, boxstyle="round,pad=0.04", fc="white", ec=col, lw=1.8))
        ax.text(x, yy + 0.1, "判为恶性" if mal > ben else "判为良性", ha="center", va="center", fontsize=11.5, color=col, weight="bold")
        ax.text(x, yy - 0.14, f"{mal} : {ben}", ha="center", va="center", fontsize=10.5, color="#444")
    if i in kids:
        p, lab = kids[i]
        px, py = pos[p]
        ax.annotate("", xy=(pos[i][0], pos[i][1] + 0.32), xytext=(px, py - 0.3), arrowprops=dict(arrowstyle="->", color="#666", lw=1.2))
        ax.text((px + pos[i][0]) / 2 + (-0.15 if lab == "是" else 0.15), (py + pos[i][1]) / 2 + 0.05, lab, ha="center", fontsize=11, color="#333")
ax.text(3.7, 3.0, "根 →", ha="right", va="center", fontsize=10.5, color="#666")
ax.text(0, 3.45, "方框里的数：恶性 : 良性（训练肿块）", fontsize=10.5, color="#444")
ax.text(5, -0.2, "最下面一行是叶", ha="center", fontsize=10.5, color="#666")
save(fig, "F1")

# ---------------- F2: training and validation accuracy against depth (two features) ----------------
dep = [int(k) for k in C2["curve"]]
fig, ax = plt.subplots(figsize=(7.2, 3.8))
ax.plot(dep, [C2["curve"][str(k)]["train"] for k in dep], "o-", color=B, lw=2.2, label="训练（319 个，拟合用）")
ax.plot(dep, [C2["curve"][str(k)]["val"] for k in dep], "s-", color=G, lw=2.2, label="验证集（107 个，没参与拟合）")
ax.annotate(f"{C2['curve']['10']['train']:.3f}", (10, C2["curve"]["10"]["train"]), xytext=(6, -2), textcoords="offset points", color=B, fontsize=10)
ax.annotate(f"{C2['curve']['10']['val']:.3f}", (10, C2["curve"]["10"]["val"]), xytext=(6, -4), textcoords="offset points", color=G, fontsize=10)
ax.set(xlabel="树的最大深度", ylabel=YACC, xticks=dep, ylim=(0.8, 1.01))
ax.legend(frameon=False, loc="lower left")
save(fig, "F2")

# ---------------- F3: which depth wins depends on which 107 lumps validate ----------------
fig, ax = plt.subplots(figsize=(7.2, 3.8))
for k, p in enumerate(C2["picks"]):
    ax.plot(dep, p["val"], "-", color=G, lw=1.2, alpha=0.55, label="5 种验证集分法" if k == 0 else None)
    bi = dep.index(p["best"])
    ax.scatter([p["best"]], [p["val"][bi]], color=G, s=30, zorder=5)
ax.plot(dep, [C2["cv"][str(k)] for k in dep], "o-", color="#333", lw=2.6, label="5 折交叉验证（426 个）")
ax.set(xlabel="树的最大深度", ylabel=YACC, xticks=dep)
ax.legend(frameon=False, loc="lower left")
bests = sorted(set(p["best"] for p in C2["picks"]))
ax.set_title(f"每种分法选出的最好深度：{'、'.join(str(p['best']) for p in C2['picks'])}")
save(fig, "F3")

# ---------------- F4a: what the MNIST subset looks like ----------------
_mn = np.load(os.path.join(HERE, "../../../../data/mnist10k.npz"))
fig, axes = plt.subplots(1, 10, figsize=(10, 1.5))
for d in range(10):
    i = int(np.where(_mn["y"] == d)[0][0])
    axes[d].imshow(_mn["X"][i].reshape(28, 28), cmap="Greys")
    axes[d].set_title(f"{d}", fontsize=11)
    axes[d].axis("off")
save(fig, "F4a")

# ---------------- F4: MNIST subset, choose the depth on the validation set ----------------
D = N["digits"]
dd = [int(k) for k in D["curve"]]
fig, ax = plt.subplots(figsize=(7.6, 3.8))
ax.plot(dd, [D["curve"][str(k)]["train"] for k in dd], "o-", color=B, lw=2.0, ms=4, label=f"训练（{D['fit']} 张）")
ax.plot(dd, [D["curve"][str(k)]["val"] for k in dd], "s-", color=G, lw=2.0, ms=4, label=f"验证（{D['val']} 张）")
ax.scatter([D["best"]], [D["best_val"]], s=150, facecolor="none", edgecolor=G, lw=2.2, zorder=5)
ax.annotate(f"最好：深度 {D['best']}，{D['best_val']:.3f}", (D["best"], D["best_val"]), xytext=(-10, -46), textcoords="offset points",
            color=G, fontsize=11, weight="bold", arrowprops=dict(arrowstyle="-", color=G, lw=0.8))
if D["natural_depth"] <= 20:  # the tree stops growing inside the range
    ax.axvline(D["natural_depth"], color="#999", ls=":", lw=1)
    ax.text(D["natural_depth"] + 0.3, 0.5, f"深度 {D['natural_depth']} 以后\n树不再长", fontsize=10, color="#555")
ax.set(xlabel="树的最大深度（搜索 1 到 20）", ylabel=YACC, xticks=[1, 5, 10, 15, 20], ylim=(0.15, 1.03))
ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.0))
save(fig, "F4")

# ---------------- F5: 30 features: four methods with the same cross-validation; what the tree uses ----------------
C3 = N["bc30"]
fig, axes = plt.subplots(1, 2, figsize=(12.0, 3.6), gridspec_kw=dict(wspace=0.9, width_ratios=[1, 1]))
names = ["logistic regression", "LDA", "QDA", f"决策树（深度 {C3['best']}）"]
vals = [C3["lr"], C3["lda"], C3["qda"], C3["best_cv"]]
ax = axes[0]
ax.barh(names, vals, color=[LIGHT, LIGHT, LIGHT, G], height=0.55)
for k, v in enumerate(vals):
    ax.text(v + 0.003, k, f"{v:.3f}", va="center", fontsize=10)
ax.set(xlim=(0, 1.0), xticks=[0, 0.25, 0.5, 0.75, 1.0], xlabel="准确率（训练集上的 5 折交叉验证）")
ax.set_xlim(0, 1.15)                                  # room for the numbers after the bars
ax.invert_yaxis()
ax = axes[1]
top = C3["top"]
ax.barh([t[0] for t in top], [t[1] for t in top], color=G, height=0.55)
for k, t in enumerate(top):
    ax.text(t[1] + 0.01, k, f"{t[1]:.2f}", va="center", fontsize=10)
ax.set(xlim=(0, 0.85), xlabel="特征重要性（决策树）")
ax.invert_yaxis()
save(fig, "F5")

# ---------------- F6: back to the ring: a depth-4 tree finds the box by itself ----------------
Rg = N["ring"]
rx, ry, rl = np.array(Rg["x"]), np.array(Rg["y"]), np.array(Rg["label"])
g = np.array(Rg["grid"])
P = np.array([int(c) for c in Rg["grid4"]]).reshape(len(g), len(g))
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), gridspec_kw=dict(wspace=0.25))
ax = axes[0]
ax.add_patch(Rectangle((-1.6, -1.6), 3.2, 3.2, facecolor="#e3efe9", edgecolor=G, lw=2.4))
classes(ax, rx, ry, rl, size=12, names=("类 0", "类 1"))
ax.set(aspect="equal", xlim=(-3.5, 3.5), ylim=(-3.5, 3.5), xlabel="x₁", ylabel="x₂", title=f"手写的方框：准确率 {Rg['acc_box']:.2f}")
ax = axes[1]
ax.contourf(g, g, P, levels=[0.5, 1.5], colors=["#e3efe9"])
ax.contour(g, g, P, levels=[0.5], colors=[G], linewidths=2.4)
classes(ax, rx, ry, rl, size=12, names=("类 0", "类 1"))
ax.set(aspect="equal", xlim=(-3.5, 3.5), ylim=(-3.5, 3.5), xlabel="x₁",
       title=f"深度 4 的树（{Rg['trees']['4']['leaves']} 片叶）：准确率 {Rg['trees']['4']['acc']:.2f}")
save(fig, "F6")

# ---------------- F7: the whole chain in one picture ----------------
steps = [("直线不行", "环形数据：人能写出“如果……就……”的规则", B),
         ("一次切一刀", "按 Gini 选：两边越纯越好；错分率常常看不出差别", G),
         ("两边再切", "同样的办法用在每一边，长成一棵树", G),
         ("切到底", "训练集满分，没见过的数据变差", R),
         ("选深度", "数据少用交叉验证，数据多用验证集；选出的深度不能在搜索范围的两端", B),
         ("检验", "测试集只读一次；与别的方法同一交叉验证比较", B)]
fig, ax = plt.subplots(figsize=(11.4 if EN else 8.4, 4.0))
ax.axis("off")
ax.set_xlim(0, 10)
ax.set_ylim(0, len(steps))
heads = [ax.text(0, len(steps) - 0.5 - k, s[0], ha="center", va="center", weight="bold", color=s[2], fontsize=13)
         for k, s in enumerate(steps)]
fig.canvas.draw()
to_data = ax.transData.inverted()
BW = max(np.ptp(to_data.transform(t.get_window_extent())[:, 0]) for t in heads) + 0.5
for k, (h, sub, col) in enumerate(steps):
    yy = len(steps) - 1 - k
    ax.add_patch(Rectangle((0.2, yy + 0.12), BW, 0.76, facecolor=col, alpha=0.14, edgecolor=col, lw=1.4))
    heads[k].set_x(0.2 + BW / 2)
    ax.text(0.5 + BW, yy + 0.5, sub, va="center", fontsize=12.5, color="#333")
    if k < len(steps) - 1:
        ax.annotate("", xy=(0.2 + BW / 2, yy - 0.12), xytext=(0.2 + BW / 2, yy + 0.12),
                    arrowprops=dict(arrowstyle="->", color="#666", lw=1.2))
save(fig, "F7")

# ---------------- slides only ----------------
if SLIDES:
    # S1: the ring with the straight line of logistic regression
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    classes(ax, rx, ry, rl, size=18, names=("类 0", "类 1"))
    from sklearn.linear_model import LogisticRegression as _LR
    _m = _LR().fit(np.c_[rx, ry], rl)
    _xs = np.linspace(-3.5, 3.5, 2)
    ax.plot(_xs, -(_m.coef_[0, 0] * _xs + _m.intercept_[0]) / _m.coef_[0, 1], color=G, lw=3)
    ax.set(aspect="equal", xlim=(-3.5, 3.5), ylim=(-3.5, 3.5), xlabel="x₁", ylabel="x₂",
           title=f"直线（logistic regression）：准确率 {Rg['acc_line']:.2f}")
    save(fig, "S1")
    # S2: the deepest tree's rectangles on the two features (many tiny boxes)
    gx, gy = np.array(C2["gx"]), np.array(C2["gy"])
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, k in zip(axes, ["2", "10"]):
        Pg = np.array([int(c) for c in C2["grids"][k]]).reshape(len(gy), len(gx))
        ax.contourf(gx, gy, Pg, levels=[0.5, 1.5], colors=["#dbe8f3"])
        classes(ax, tx, ty, tl, size=12)
        ax.set(xlabel="细胞核平均半径", title=f"深度 {k}：{C2['curve'][k]['leaves']} 片叶")
    axes[0].set_ylabel("细胞核平均纹理")
    save(fig, "S2")
if SLIDES:
    # S3: cross-validation against one validation set, as bars
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 3.8), gridspec_kw=dict(wspace=0.25))
    ax = axes[0]
    for k in range(5):
        for j in range(5):
            ax.add_patch(Rectangle((j, 4 - k), 0.96, 0.8, facecolor=G if j == k else "#dbe8f3", edgecolor="white"))
        ax.text(5.15, 4 - k + 0.4, f"拟合 {k + 1}", va="center", fontsize=13)
    ax.set(xlim=(0, 6.6), ylim=(-0.2, 5), xticks=[], yticks=[], title="交叉验证：每个深度拟合 5 次")
    ax.axis("off")
    ax.set_title("交叉验证：每个深度拟合 5 次", fontsize=15)
    ax = axes[1]
    ax.add_patch(Rectangle((0, 2), 3.3, 0.8, facecolor="#dbe8f3", edgecolor="white"))
    ax.add_patch(Rectangle((3.3, 2), 1.9, 0.8, facecolor=G, edgecolor="white"))
    ax.add_patch(Rectangle((5.3, 2), 1.7, 0.8, facecolor=LIGHT, edgecolor="white"))
    ax.text(1.65, 2.4, "拟合", ha="center", va="center", fontsize=14)
    ax.text(4.25, 2.4, "验证", ha="center", va="center", fontsize=14, color="white")
    ax.text(6.15, 2.4, "测试", ha="center", va="center", fontsize=14)
    ax.annotate("", xy=(0, 1.85), xytext=(5.2, 1.85), arrowprops=dict(arrowstyle="<->", color="#555"))
    ax.text(2.6, 1.45, "训练集", ha="center", fontsize=13, color="#333")
    ax.text(6.15, 1.45, "最后只读一次", ha="center", fontsize=13, color="#333")
    ax.set(xlim=(0, 7), ylim=(0.8, 4.2))
    ax.axis("off")
    ax.set_title("验证集：每个深度拟合 1 次", fontsize=15)
    save(fig, "S3")
print("figures ok")
