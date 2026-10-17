"""Lesson 4 Note: static figures. Reads numbers.json written by build_numbers.py.
Run (from this folder):  python build_note_figures.py            -> Chinese figures here
                         python build_note_figures.py --lang en  -> English figures in ../note_en/
                         python build_note_figures.py --slides   -> English, projector-size text in ../slides/
Colours (fixed across the course notes): data blue, model green, error red.
Classes: malignant = filled blue dots, benign = hollow grey circles.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
SLIDES = "--slides" in os.sys.argv
EN = SLIDES or ("--lang" in os.sys.argv and os.sys.argv[os.sys.argv.index("--lang") + 1] == "en")
OUTDIR = os.path.join(HERE, "..", "slides" if SLIDES else "note_en") if EN else HERE
os.makedirs(OUTDIR, exist_ok=True)
if EN:  # every visible string goes through the translation table; leftover Chinese stops the build
    import matplotlib.text
    from i18n_en import tr, check
    _set_text = matplotlib.text.Text.set_text
    matplotlib.text.Text.set_text = lambda self, s: _set_text(self, check(tr(str(s)), "figure text") if s is not None else s)
N = json.load(open(os.path.join(HERE, "numbers.json")))
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

d = load_breast_cancer()
y = (d.target == 0).astype(int)
Xtr, Xte, ytr, yte = train_test_split(d.data, y, test_size=0.25, random_state=0, stratify=y)
rad = Xtr[:, list(d.feature_names).index("mean radius")]
jit = np.random.default_rng(1).uniform(-0.04, 0.04, len(rad))
grid = np.linspace(6, 42, 400)
L = N["line"]


def save(fig, name):
    fig.savefig(os.path.join(OUTDIR, f"{name}.svg"), bbox_inches="tight")
    if os.environ.get("PNGDIR"):  # optional PNG copies for checking the figures by eye
        fig.savefig(os.path.join(os.environ["PNGDIR"], f"{name}.png"), bbox_inches="tight", dpi=90)
    plt.close(fig)


def classes(ax, xs, ys, labels, size=16):
    ax.scatter(xs[labels == 0], ys[labels == 0], s=size, facecolor="none", edgecolor=GREY, lw=0.8, label="良性（0）")
    ax.scatter(xs[labels == 1], ys[labels == 1], s=size, color=B, alpha=0.7, lw=0, label="恶性（1）")


# ---------------- F1: radius by class, the two groups overlap ----------------
fig, ax = plt.subplots(figsize=(7.2, 3.4))
bins = np.linspace(6, 29, 35)
ax.hist(rad[ytr == 0], bins=bins, histtype="step", color=GREY, lw=1.8, label="良性（267 例）")
ax.hist(rad[ytr == 1], bins=bins, color=B, alpha=0.55, label="恶性（159 例）")
ax.axvline(N["tumour"]["radius"], color="#333", ls="--", lw=1.2)
ax.annotate(f"开场的肿块：半径 {N['tumour']['radius']}", (N["tumour"]["radius"], 30), xytext=(17.0, 36),
            fontsize=10, va="center", arrowprops=dict(arrowstyle="->", color="#333", lw=0.9))
ax.set(xlabel="细胞核平均半径", ylabel="肿块数")
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
save(fig, "F1")

# ---------------- F2: a straight line on 0/1 labels; 20 large tumours move its boundary ----------------
fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.9), sharey=True)
for k, ax in enumerate(axes):
    xs, ys = (rad, ytr) if k == 0 else (np.r_[rad, np.full(20, L["added_radius"])], np.r_[ytr, np.ones(20, int)])
    jj = np.r_[jit, np.random.default_rng(2).uniform(-0.04, 0.04, 20)][:len(xs)]
    classes(ax, xs, ys + jj, ys)
    lin = LinearRegression().fit(xs[:, None], ys)
    ax.axhspan(1, 2, color=R, alpha=0.08)
    ax.axhspan(-1, 0, color=R, alpha=0.08)
    ax.plot(grid, lin.predict(grid[:, None]), color=G, lw=2.4)
    bnd = L["boundary"] if k == 0 else L["boundary_added"]
    ax.axvline(bnd, color=G, ls="--", lw=1.2)
    ax.text(bnd + 0.4, -0.38, f"按 0.5 切：{bnd}", color=G, fontsize=10)
    ax.set(xlim=(5, 42), ylim=(-0.5, 2.0), xlabel="细胞核平均半径")
    ax.set_title("训练数据" if k == 0 else f"再加 20 个半径 {L['added_radius']:.0f} 的假想肿块")
axes[0].set_ylabel("标签 / 直线的预测")
axes[0].text(31, 1.2, "超出 0 到 1", color=R, fontsize=10)
axes[0].legend(frameon=False, loc="upper left")
save(fig, "F2")

# ---------------- F3: logistic regression on the same data; the boundary does not move ----------------
fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.9), sharey=True)
for k, ax in enumerate(axes):
    xs, ys = (rad, ytr) if k == 0 else (np.r_[rad, np.full(20, L["added_radius"])], np.r_[ytr, np.ones(20, int)])
    jj = np.r_[jit, np.random.default_rng(2).uniform(-0.04, 0.04, 20)][:len(xs)]
    classes(ax, xs, ys + jj, ys)
    m = LogisticRegression().fit(xs[:, None], ys)
    ax.plot(grid, m.predict_proba(grid[:, None])[:, 1], color=G, lw=2.4)
    bnd = -m.intercept_[0] / m.coef_[0, 0]
    ax.axvline(bnd, color=G, ls="--", lw=1.2)
    ax.text(bnd + 0.4, 0.42, f"p = 0.5：{bnd:.2f}", color=G, fontsize=10)
    ax.set(xlim=(5, 42), ylim=(-0.15, 1.15), xlabel="细胞核平均半径")
    ax.set_title("训练数据" if k == 0 else f"再加 20 个半径 {L['added_radius']:.0f} 的假想肿块")
axes[0].set_ylabel("恶性的概率 p")
save(fig, "F3")

# ---------------- F4: squared error vs cross-entropy for a true 1 ----------------
fig, ax = plt.subplots(figsize=(7.0, 3.8))
p = np.linspace(0.002, 1, 500)
ax.plot(p, (1 - p) ** 2, color=GREY, lw=2.2, label="平方误差 (1 − p)²")
ax.plot(p, -np.log(p), color=R, lw=2.4, label="交叉熵 −log p")
for q in [0.1, 0.01]:
    ax.plot([q, q], [(1 - q) ** 2, -np.log(q)], color="#999", lw=0.8, ls=":")
    ax.scatter([q, q], [(1 - q) ** 2, -np.log(q)], color=["#555", R], s=22, zorder=5)
    ax.text(q + 0.02, -np.log(q), f"p = {q}：{-np.log(q):.1f}", color=R, fontsize=10, va="center")
ax.set(xlim=(0, 1), ylim=(0, 5.2), xlabel="模型给“恶性”的概率 p（实际是恶性）", ylabel="这一例的损失")
ax.legend(frameon=False)
save(fig, "F4")

# ---------------- F5: cross-entropy along one weight, and gradient descent steps ----------------
gd = N["gd"]
fig, ax = plt.subplots(figsize=(7.0, 3.8))
ax.plot(gd["grid"], gd["loss"], color=R, lw=2.4)
pw = [s["w"] for s in gd["path"]]
pl = [s["loss"] for s in gd["path"]]
ax.plot(pw, pl, "o-", color=G, ms=5, lw=1.2)
for k, off in [(0, (8, 6)), (1, (-4, 14)), (len(pw) - 1, (-10, 14))]:
    ax.annotate(f"第 {k} 步", (pw[k], pl[k]), xytext=off, textcoords="offset points", fontsize=10, color=G)
ax.axvline(gd["w_opt"], color="#999", ls=":", lw=1)
ax.text(gd["w_opt"] + 0.1, max(gd["loss"]) * 0.9, f"最低点 w = {gd['w_opt']:.2f}", fontsize=10, color="#555")
ax.set(xlabel="半径（标准化后）的权重 w", ylabel="训练集上的平均交叉熵", xlim=(-1, 7))
save(fig, "F5")

# ---------------- F6: confusion matrices from cross-validated predictions ----------------


def cm_plot(ax, c, title):
    M = np.array([[c["tp"], c["fn"]], [c["fp"], c["tn"]]])
    ax.imshow(np.array([[1, 0], [0, 1]]), cmap=matplotlib.colors.ListedColormap(["#fbe3d6", "#e3efe9"]), vmin=0, vmax=1)
    names = [["抓到\n（真阳性）", "漏诊\n（假阴性）"], ["误报\n（假阳性）", "排除\n（真阴性）"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i - 0.12, str(M[i, j]), ha="center", va="center", fontsize=18, weight="bold",
                    color=R if i != j else G)
            ax.text(j, i + 0.27, names[i][j], ha="center", va="center", fontsize=12, color="#333", linespacing=1.0)
    ax.set_xticks([0, 1], ["预测恶性", "预测良性"])
    ax.set_yticks([0, 1], ["实际恶性", "实际良性"])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title)


fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2), gridspec_kw=dict(wspace=0.75))
cm_plot(axes[0], N["cv1"], f"只用半径：准确率 {N['cv1']['acc']}")
cm_plot(axes[1], N["cv30"], f"30 个特征：准确率 {N['cv30']['acc']}")
save(fig, "F6")

# ---------------- F7: MNIST subset, 10 x 10 confusion matrix and the weight images ----------------
D = N["digits"]
fig = plt.figure(figsize=(12.0, 6.4))
ax = fig.add_axes([0.0, 0.07, 0.42, 0.82])
cm = np.array(D["cm"])
ax.imshow(np.log1p(cm), cmap="Greens")
for i in range(10):
    for j in range(10):
        if cm[i, j]:
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=10.5, color="white" if i == j else R,
                    weight="bold" if i != j and cm[i, j] >= 10 else "normal")
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set(xlabel="预测的数字", ylabel="实际的数字", title=f"测试集 {D['test']} 张：准确率 {D['acc']}")
_mn = np.load(os.path.join(HERE, "../../../../data/mnist10k.npz"))      # one real image of each digit, to show what the data look like
fig.text(0.735, 0.905, "数据长这样：每个数字一张 28 × 28 的图", ha="center", fontsize=11)
for k in range(10):
    a = fig.add_axes([0.47 + k * 0.053, 0.76, 0.048, 0.13])
    a.imshow(_mn["X"][np.flatnonzero(_mn["y"] == k)[0]].reshape(28, 28), cmap="gray_r")
    a.set_xticks([])
    a.set_yticks([])
    for sp in a.spines.values():
        sp.set_color("#bbb")
fig.text(0.735, 0.68, "每一类的 784 个权重排成 28 × 28（红为正，蓝为负）", ha="center", fontsize=11)
for k in range(10):
    a = fig.add_axes([0.47 + (k % 5) * 0.106, 0.36 - (k // 5) * 0.33, 0.095, 0.28])
    w = np.array(D["coef"][k]).reshape(28, 28)
    lim = np.abs(w).max()
    a.imshow(w, cmap="RdBu_r", vmin=-lim, vmax=lim)
    a.set_xticks([])
    a.set_yticks([])
    a.set_title(f"数字 {k}", fontsize=10)
    for sp in a.spines.values():
        sp.set_visible(False)
save(fig, "F7")

# ---------------- F8: a ring cannot be split by a straight line; a distance feature can ----------------
Rg = N["ring"]
rx, ry, rl = np.array(Rg["x"]), np.array(Rg["y"]), np.array(Rg["label"])
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), gridspec_kw=dict(wspace=0.3))
classes(axes[0], rx, ry, rl, size=14)
axes[0].set(aspect="equal", xlabel="x₁", ylabel="x₂", title=f"两个原始特征：直线的准确率只有 {Rg['acc_line']}")
dist = np.sqrt(rx ** 2 + ry ** 2)
jj = np.random.default_rng(3).uniform(-0.15, 0.15, len(dist))
classes(axes[1], dist, rl + jj, rl, size=14)
axes[1].axvline(1.7, color=G, ls="--", lw=1.4)
axes[1].set(xlabel="新特征：到中心的距离 √(x₁² + x₂²)", yticks=[0, 1], ylabel="类别",
            title=f"一个新特征：准确率 {Rg['acc_dist']}")
save(fig, "F8")

# ---------------- F9: the whole chain in one picture ----------------
steps = [("答案是类别", "直线的输出超出 0 到 1，还被远处的点拉动", B),
         ("模型：sigmoid", "输出恶性的概率，边界是一条直线", G),
         ("损失：交叉熵", "自信地错了，罚得很重", G),
         ("求解：梯度下降", "没有闭式解，沿下坡一步步走", G),
         ("检验：基线与混淆矩阵", "准确率之外，看漏诊和误报各有多少", B),
         ("决策：阈值", "按错误代价在训练集内选，测试集只读一次", R)]
fig, ax = plt.subplots(figsize=(10.4 if EN else 7.6, 4.0))
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
save(fig, "F9")
# ---------------- F10: ROC curve with the three thresholds ----------------
Rc = N["roc"]
fig, ax = plt.subplots(figsize=(5.6, 4.8))
ax.plot(Rc["fpr"], Rc["tpr"], color="#333", lw=2)
ax.plot([0, 1], [0, 1], color="#bbb", ls="--", lw=1)
for t, xy in [("0.5", (0.04, 0.80)), ("0.3", (0.08, 0.895)), ("0.1", (0.20, 0.93))]:
    q = Rc["points"][t]
    ax.scatter([q["fpr"]], [q["tpr"]], color=G, s=50, zorder=5)
    ax.annotate(f"阈值 {t}", (q["fpr"], q["tpr"]), xytext=xy, color=G, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=G, lw=1))
ax.set(xlim=(-0.005, 0.35), ylim=(0.75, 1.005), xlabel="误报率（误报数 / 良性总数）", ylabel="召回率",
       title=f"ROC 曲线：AUC = {Rc['auc']}（放大的左上角）")
inset = ax.inset_axes([0.58, 0.08, 0.38, 0.38])  # the whole curve, for orientation
inset.plot(Rc["fpr"], Rc["tpr"], color="#333", lw=1.5)
inset.plot([0, 1], [0, 1], color="#bbb", ls="--", lw=0.8)
inset.add_patch(Rectangle((0, 0.75), 0.35, 0.25, fill=False, edgecolor=G, lw=1.2))
inset.set(xlim=(0, 1), ylim=(0, 1), xticks=[0, 1], yticks=[0, 1])
inset.set_title("完整曲线，绿框是左边放大的部分", fontsize=8)
inset.tick_params(labelsize=8)
save(fig, "F10")

# ---------------- slides only: S5 (the loss curve before any step) and S6 (radius-only confusion matrix) ----------------
if SLIDES:
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.plot(gd["grid"], gd["loss"], color=R, lw=2.4)
    ax.scatter([0], [gd["path"][0]["loss"]], color=G, s=60, zorder=5)
    ax.annotate("从 w = 0 出发", (0, gd["path"][0]["loss"]), xytext=(10, 10), textcoords="offset points", color=G)
    ax.set(xlabel="半径（标准化后）的权重 w", ylabel="训练集上的平均交叉熵", xlim=(-1, 7))
    save(fig, "S5")
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    cm_plot(ax, N["cv1"], f"只用半径：准确率 {N['cv1']['acc']}")
    save(fig, "S6")
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    p = np.linspace(0.002, 1, 500)
    ax.plot(p, (1 - p) ** 2, color=GREY, lw=2.6, label="平方误差 (1 − p)²")
    for q, off in [(0.1, (30, 4)), (0.01, (30, 34))]:
        ax.scatter([q], [(1 - q) ** 2], color="#333", s=40, zorder=5)
        ax.annotate(f"p = {q}：{(1 - q) ** 2:.2f}", (q, (1 - q) ** 2), xytext=off, textcoords="offset points", fontsize=12,
                    arrowprops=dict(arrowstyle="-", color="#999", lw=0.8))
    ax.axhline(1, color="#999", ls=":", lw=1)
    ax.text(0.62, 1.06, "最多罚 1", color="#555", fontsize=12)
    ax.set(xlim=(0, 1), ylim=(0, 1.5), xlabel="模型给“恶性”的概率 p（实际是恶性）", ylabel="这一例的损失")
    ax.legend(frameon=False)
    save(fig, "S4")
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    names = ["永远猜良性", "只用半径"]
    vals = [N["baseline"]["acc"], N["cv1"]["acc"]]
    ax.barh(names, vals, color=[LIGHT, B], height=0.55)
    for k, v in enumerate(vals):
        ax.text(v + 0.01, k, f"{v:.3f}", va="center", fontsize=15, weight="bold")
    ax.set(xlim=(0, 1.08), xlabel="准确率（训练集上的交叉验证）")
    ax.invert_yaxis()
    save(fig, "S16")
    # S10: the fitted two-feature boundary is a straight line
    from sklearn.preprocessing import StandardScaler as _SS
    T2 = N["two"]
    z1, z2, lab = np.array(T2["z1"]), np.array(T2["z2"]), np.array(T2["label"])
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    gx, gy = np.meshgrid(np.linspace(-2.5, 4.5, 200), np.linspace(-2.5, 4.5, 200))
    pp = 1 / (1 + np.exp(-(T2["w1"] * gx + T2["w2"] * gy + T2["b"])))
    ax.contourf(gx, gy, pp, levels=[0, 0.5, 1], colors=["#ffffff", "#e3efe9"])
    classes(ax, z1, z2, lab, size=18)
    xs = np.linspace(-2.5, 4.5, 2)
    ax.plot(xs, -(T2["w1"] * xs + T2["b"]) / T2["w2"], color=G, lw=3)
    ax.text(2.2, 3.9, "p ≥ 0.5：判为恶性", color=G, fontsize=13, weight="bold")
    ax.set(xlim=(-2.5, 4.5), ylim=(-2.5, 4.5), xlabel="细胞核平均半径（标准化）", ylabel="细胞核纹理（标准化）")
    save(fig, "S10")
print("figures ok")
