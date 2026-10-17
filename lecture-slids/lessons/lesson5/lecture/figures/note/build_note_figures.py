"""Lesson 5 Note: static figures. Reads numbers.json written by build_numbers.py.
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
from matplotlib.patches import Rectangle
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
SLIDES = "--slides" in os.sys.argv
EN = SLIDES or "en" in " ".join(os.sys.argv[1:]).split()  # also accepts "--lang en" passed as one word
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


def classes(ax, xs, ys, labels, size=16, names=("类 0", "类 1")):
    ax.scatter(xs[labels == 0], ys[labels == 0], s=size, facecolor="none", edgecolor=GREY, lw=0.8, label=names[0])
    ax.scatter(xs[labels == 1], ys[labels == 1], s=size, color=B, alpha=0.75, lw=0, label=names[1])


def ellipse(ax, mu, S, k, **kw):
    vals, vecs = np.linalg.eigh(np.array(S))
    t = np.linspace(0, 2 * np.pi, 200)
    pts = np.array(mu)[:, None] + k * (vecs @ (np.sqrt(vals)[:, None] * np.vstack([np.cos(t), np.sin(t)])))
    ax.plot(pts[0], pts[1], **kw)


# ---------------- F1: one feature, a bell curve for each class ----------------
O = N["one"]
rad, lab = np.array(O["r"]), np.array(O["y"])
fig, ax = plt.subplots(figsize=(7.4, 3.6))
bins = np.linspace(6, 29, 35)
ax.hist(rad[lab == 0], bins=bins, density=True, histtype="step", color=GREY, lw=1.4, alpha=0.8)
ax.hist(rad[lab == 1], bins=bins, density=True, color=B, alpha=0.25)
gx = np.linspace(5, 30, 400)
c1, c0 = O["cls"]["1"], O["cls"]["0"]
ax.plot(gx, norm.pdf(gx, c0["mean"], c0["sd"]), color=GREY, lw=2.4)
ax.plot(gx, norm.pdf(gx, c1["mean"], c1["sd"]), color=B, lw=2.4)
ax.text(5.4, 0.205, f"良性\n平均 {c0['mean']}\n标准差 {c0['sd']}", color=GREY, fontsize=10.5, va="top")
ax.text(22.5, 0.13, f"恶性\n平均 {c1['mean']}\n标准差 {c1['sd']}", color=B, fontsize=10.5, va="bottom")
x0 = O["x"]
ax.plot([x0, x0], [0, O["h0"]], color="#333", ls="--", lw=1.2)
ax.scatter([x0, x0], [O["h0"], O["h1"]], color=[GREY, B], s=40, zorder=5)
ax.annotate(f"高 {O['h0']:.3f}", (x0, O["h0"]), xytext=(10, 4), textcoords="offset points", color=GREY, fontsize=10)
ax.annotate(f"高 {O['h1']:.3f}", (x0, O["h1"]), xytext=(-96, 4), textcoords="offset points", color=B, fontsize=10)
ax.text(x0, -0.03, "12.8", ha="center", va="top", fontsize=10, color="#333", transform=ax.get_xaxis_transform())
ax.set(xlabel="细胞核平均半径", ylabel="密度", xlim=(5, 30))
save(fig, "F1")

# ---------------- F2: breast cancer, 30 features: three methods, the same cross-validation ----------------
bc = N["bc"]
fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.6), gridspec_kw=dict(wspace=0.55, width_ratios=[1.15, 1]))
names = ["永远猜良性", "logistic regression", "LDA（共用散布）", "QDA（各有散布）"]
vals = [bc["baseline"], bc["lr"]["acc"], bc["lda"]["acc"], bc["qda"]["acc"]]
ax = axes[0]
ax.barh(names, vals, color=[LIGHT, G, B, B], height=0.55)
for k, v in enumerate(vals):
    ax.text(v + 0.005, k, f"{v:.3f}", va="center", fontsize=10)
    if k:
        key = ["", "lr", "lda", "qda"][k]
        ax.text(0.56, k, f"漏诊 {bc[key]['missed']}，误报 {bc[key]['false_alarms']}", va="center", fontsize=9.5, color="white")
ax.set(xlim=(0.55, 1.0), xlabel="准确率（训练集上的 5 折交叉验证）")
ax.invert_yaxis()
ax = axes[1]
cn = ["logistic regression", "LDA", "QDA"]
cv = [bc["counts"]["lr"], bc["counts"]["lda"], bc["counts"]["qda"]]
ax.barh(cn, cv, color=[G, B, B], height=0.55)
for k, v in enumerate(cv):
    ax.text(v + 15, k, str(v), va="center", fontsize=10)
ax.axvline(426, color=R, ls="--", lw=1.4)
ax.text(436, -0.45, "训练肿块 426 个", color=R, fontsize=10, va="center")
ax.set(xlim=(0, 1150), ylim=(2.5, -0.75), xlabel="要从数据里估的数")
save(fig, "F2")

# ---------------- F3: MNIST subset, the centre of each class and the pixels that are always 0 ----------------
D = N["digits"]
fig = plt.figure(figsize=(12.0, 3.2))
for k in range(10):
    a = fig.add_axes([0.0 + k * 0.074, 0.18, 0.068, 0.62])
    a.imshow(np.array(D["means"][k]).reshape(28, 28), cmap="Greys", vmin=0, vmax=1)
    a.set_xticks([])
    a.set_yticks([])
    a.set_title(f"{k}", fontsize=13)
fig.text(0.37, 0.93, "每一类的中心：训练图像的平均", ha="center", fontsize=14)
a = fig.add_axes([0.80, 0.12, 0.17, 0.72])
sd0 = np.array(D["sd0"]).reshape(28, 28)
rgb = plt.get_cmap("Blues")(sd0 / sd0.max())
rgb[sd0 == 0] = matplotlib.colors.to_rgba("#f4b6a0")  # the pixels that are 0 in every training image of a 0
a.imshow(rgb)
a.set_xticks([])
a.set_yticks([])
a.set_title(f"数字 0 每个像素散多开（蓝越深越开）\n浅红：{D['zero_px'][0]} 个像素永远是 0", fontsize=13)
save(fig, "F3")

# ---------------- F4: choosing reg_param by cross-validation (MNIST subset) ----------------
fig, ax = plt.subplots(figsize=(7.4, 3.8))
grid = D["grid"]
xs = [1e-5 if s == 0 else s for s in grid]  # 0 cannot sit on a log axis: draw it at the far left
ys = [D["curve"][str(s)] for s in grid]
ax.axvspan(0.01, 0.5, color=LIGHT, alpha=0.5, lw=0)
ax.set_title("灰色带：只搜 0.01、0.1、0.3、0.5，最好的 0.01 落在左端", fontsize=10, color="#333")
ax.annotate(f"全收成圆：{ys[-1]:.3f}", (xs[-1], ys[-1]), xytext=(-10, 0), textcoords="offset points",
            color="#555", fontsize=10, ha="right", va="center")
ax.plot(xs, ys, "o-", color=G, lw=2.2, ms=6)
nx = [v for v in D["narrow"] if str(v) not in D["curve"] and str(float(v)) not in D["curve"]]   # 0.3: only in the narrow search
ax.scatter(nx, [D["narrow_curve"][str(v)] for v in nx], s=36, color=G, zorder=5)
bi = grid.index(D["best"])
ax.scatter([xs[bi]], [ys[bi]], s=140, facecolor="none", edgecolor=G, lw=2.2, zorder=5)
ax.annotate(f"最好：{D['best']}，{D['best_cv']:.3f}", (xs[bi], ys[bi]), xytext=(0, 16), textcoords="offset points", ha="center",
            color=G, fontsize=11, weight="bold", arrowprops=dict(arrowstyle="-", color=G, lw=0.8))
ax.annotate(f"不收缩：{ys[0]:.3f}", (xs[0], ys[0]), xytext=(12, -4), textcoords="offset points", color=R, fontsize=10)
ax.set_xscale("log")
tx = xs + nx
ax.set_xticks(tx, ["0"] + ["" if v == 0.5 else str(v) for v in grid[1:]] + ["" for v in nx])   # 0.3 and 0.5 are named in the grey band
ax.minorticks_off()
ax.set(ylim=(0.4, 0.95), xlabel="reg_param（往“圆”收多少；对数刻度）", ylabel="交叉验证准确率")
save(fig, "F4")

# ---------------- F5: the whole chain in one picture ----------------
steps = [("直线不够", "一紧一散的两类，最好的直线也只比猜多数类好一点", B),
         ("描述每一类", "一维：钟形曲线；二维：椭圆（中心与散布）", G),
         ("比较", "在这一类里有多常见 × 这一类的比例，谁大判谁", G),
         ("两个候选", "共用散布：直线（LDA）；各有散布：弯线（QDA）", G),
         ("检验", "同一交叉验证下比较；更灵活不一定更好", B),
         ("估不出来时", "往“圆”收一点（reg_param），用交叉验证选，不落在两端", R)]
fig, ax = plt.subplots(figsize=(11.2 if EN else 8.2, 4.0))
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
save(fig, "F5")

# ---------------- F6: back to the opening data: three methods, three boundaries ----------------
T = N["two"]
x1, x2, lb = np.array(T["x1"]), np.array(T["x2"]), np.array(T["label"])
X0, X1, Y0, Y1 = -3.5, 5.5, -3.5, 4.5
gx, gy = np.meshgrid(np.linspace(X0, X1, 300), np.linspace(Y0, Y1, 300))
P = np.c_[gx.ravel(), gy.ravel()]


def post(S1, S0):
    out = []
    for mu, S, share in [(T["mu"]["1"], S1, T["n1"] / 360), (T["mu"]["0"], S0, T["n0"] / 360)]:
        S = np.array(S)
        diff = P - np.array(mu)
        quad = np.einsum("ij,jk,ik->i", diff, np.linalg.inv(S), diff)
        out.append(np.exp(-0.5 * quad) / (2 * np.pi * np.sqrt(np.linalg.det(S))) * share)
    return (out[0] / (out[0] + out[1])).reshape(gx.shape)


lrp = 1 / (1 + np.exp(-(T["lr"]["w1"] * gx + T["lr"]["w2"] * gy + T["lr"]["b"])))
panels = [("logistic regression", lrp, None), ("LDA：共用散布", post(T["pooled"], T["pooled"]), "pooled"),
          ("QDA：各有散布", post(T["cov"]["1"], T["cov"]["0"]), "own")]
fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4), sharey=True, gridspec_kw=dict(wspace=0.08))
for ax, (title, pp, kind), key in zip(axes, panels, ["lr", "lda", "qda"]):
    ax.contourf(gx, gy, pp, levels=[0.5, 1], colors=["#e3efe9"])
    ax.contour(gx, gy, pp, levels=[0.5], colors=[G], linewidths=2.4)
    classes(ax, x1, x2, lb, size=12)
    if kind == "pooled":
        for c in ("1", "0"):
            ellipse(ax, T["mu"][c], T["pooled"], 2, color=B if c == "1" else GREY, lw=1.4, ls="--")
    if kind == "own":
        for c in ("1", "0"):
            ellipse(ax, T["mu"][c], T["cov"][c], 2, color=B if c == "1" else GREY, lw=1.4, ls="--")
    ax.set(xlim=(X0, X1), ylim=(Y0, Y1), aspect="equal", xlabel="x₁")
    ax.set_title(f"{title}\n准确率 {T['acc'][key]:.3f}")
axes[0].set_ylabel("x₂")
save(fig, "F6")

# ---------------- slides only ----------------
if SLIDES:
    # S1: the opening data with the logistic line (static copy for the question page)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.contourf(gx, gy, lrp, levels=[0.5, 1], colors=["#e3efe9"])
    ax.contour(gx, gy, lrp, levels=[0.5], colors=[G], linewidths=2.6)
    classes(ax, x1, x2, lb, size=18, names=("类 0（240 个）", "类 1（120 个）"))
    ax.set(xlim=(X0, X1), ylim=(Y0, Y1), aspect="equal", xlabel="x₁", ylabel="x₂")
    ax.legend(frameon=False, loc="upper right")
    save(fig, "S1")
    # S2: each class drawn as an ellipse (own spread), no boundary yet
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    classes(ax, x1, x2, lb, size=18)
    for c in ("1", "0"):
        for k in (1, 2):
            ellipse(ax, T["mu"][c], T["cov"][c], k, color=B if c == "1" else GREY, lw=2.4 if k == 1 else 1.4)
        ax.scatter(*T["mu"][c], marker="+", s=200, color=B if c == "1" else "#333", lw=2.5, zorder=6)
    ax.set(xlim=(X0, X1), ylim=(Y0, Y1), aspect="equal", xlabel="x₁", ylabel="x₂")
    save(fig, "S2")
    # S4: the two histograms alone, before any curve (question page)
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.hist(rad[lab == 0], bins=bins, density=True, histtype="step", color=GREY, lw=2.0, label="良性（267 个）")
    ax.hist(rad[lab == 1], bins=bins, density=True, color=B, alpha=0.45, label="恶性（159 个）")
    ax.set(xlabel="细胞核平均半径", ylabel="密度", xlim=(5, 30))
    ax.legend(frameon=False, loc="upper right")
    save(fig, "S4")
    # S3: the bar chart of accuracies alone (question page)
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    ax.barh(names, vals, color=[LIGHT, G, B, B], height=0.55)
    for k, v in enumerate(vals):
        ax.text(v + 0.004, k, f"{v:.3f}", va="center", fontsize=16, weight="bold")
    ax.set(xlim=(0.55, 1.03), xlabel="准确率（训练集上的 5 折交叉验证）")
    ax.invert_yaxis()
    save(fig, "S3")
if SLIDES:
    # S5: three shapes to shrink towards (schematic, two pixels of the digits)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0), gridspec_kw=dict(wspace=0.35))
    tt = np.linspace(0, 2 * np.pi, 200)
    titles = [f"共用的散布：仍然是平的\n（{D['zero_all']} 个像素在所有图像里都是 0）", "只留对角线：仍然是平的\n（这一类里恒为 0 的像素）", "一个圆：每个方向都有厚度"]
    for k, ax in enumerate(axes):
        ax.plot([-1.6, 1.6], [0, 0], color=B, lw=3, ls="--")
        if k == 0:
            ax.plot([-1.2, 1.2], [0.0, 0.0], color=G, lw=4)
        if k == 1:
            ax.plot(1.3 * np.cos(tt), 0.0 * np.sin(tt), color=G, lw=4)
        if k == 2:
            ax.plot(np.cos(tt), np.sin(tt), color=G, lw=4)
        ax.set(xlim=(-2, 2), ylim=(-1.5, 1.5), aspect="equal", xticks=[], yticks=[], xlabel="像素 a", ylabel="像素 b")
        ax.set_title(titles[k], fontsize=13)
    fig.text(0.5, -0.04, "蓝色虚线：一类自己的散布，在“像素 b”方向上没有厚度；绿色：往它收的形状", ha="center", fontsize=12)
    save(fig, "S5")
    # S5q: the same three sketches before the answer: only the flat spread and the candidate names
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0), gridspec_kw=dict(wspace=0.35))
    for k, ax in enumerate(axes):
        ax.plot([-1.6, 1.6], [0, 0], color=B, lw=3, ls="--")
        ax.set(xlim=(-2, 2), ylim=(-1.5, 1.5), aspect="equal", xticks=[], yticks=[], xlabel="像素 a", ylabel="像素 b")
        ax.set_title(["候选 1：往共用的散布收", "候选 2：只留对角线", "候选 3：往一个圆收"][k], fontsize=14)
    fig.text(0.5, -0.04, "蓝色虚线：一类自己的散布，在“像素 b”方向上没有厚度", ha="center", fontsize=12)
    save(fig, "S5q")
    # S7: the two-feature data alone (question page about the tilt)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    classes(ax, x1, x2, lb, size=18, names=("类 0（240 个）", "类 1（120 个）"))
    ax.set(xlim=(X0, X1), ylim=(Y0, Y1), aspect="equal", xlabel="x₁", ylabel="x₂")
    ax.legend(frameon=False, loc="upper right")
    save(fig, "S7")
    # S6: the two candidates before any boundary (predict page)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), sharey=True)
    for ax, kind in zip(axes, ["pooled", "own"]):
        classes(ax, x1, x2, lb, size=10)
        for c in ("1", "0"):
            S = T["pooled"] if kind == "pooled" else T["cov"][c]
            ellipse(ax, T["mu"][c], S, 2, color=B if c == "1" else "#333", lw=2.6)
        ax.set(xlim=(X0, X1), ylim=(Y0, Y1), aspect="equal", xlabel="x₁")
        ax.set_title("两类共用一个 Σ" if kind == "pooled" else "每类有自己的 Σ")
    axes[0].set_ylabel("x₂")
    save(fig, "S6")
print("figures ok")
