"""Lesson 10 Note: static figures. Reads numbers.json written by build_numbers.py.
Run (from this folder):  python build_note_figures.py            -> Chinese figures here
                         python build_note_figures.py --lang en  -> English figures in ../note_en/
                         python build_note_figures.py --slides   -> English, projector-size text in ../slides/
Colours (fixed across the course notes): data blue, model green, error red.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text
from matplotlib.patches import Rectangle, FancyBboxPatch

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


def clean(ax):
    ax.set(xticks=[], yticks=[])
    for sp in ax.spines.values():
        sp.set_visible(False)


D, C = N["dense"], N["cnn"]
seven = np.array([w for w in N["widget_images"] if w["digit"] == 7][0]["img"])
perm = np.random.default_rng(0).permutation(784)

# ---------------- F1: shuffled pixels do not bother the fully connected network ----------------
fig = plt.figure(figsize=(10, 3.4))
ax1, ax2 = fig.add_axes([0.0, 0.05, 0.22, 0.8]), fig.add_axes([0.25, 0.05, 0.22, 0.8])
ax1.imshow(seven, cmap="Greys")
ax1.set_title("原来的图")
ax2.imshow(seven.reshape(784)[perm].reshape(28, 28), cmap="Greys")
ax2.set_title("像素打乱后")
for ax in (ax1, ax2):
    clean(ax)
ax3 = fig.add_axes([0.62, 0.18, 0.36, 0.62])
vals = [D["val"][-1], D["shuffled_val"][-1]]
ax3.barh(["原来的图", "像素打乱后"], vals, color=[G, R], height=0.5)
for k, v in enumerate(vals):
    ax3.text(v + 0.002, k, f"{v:.4f}", va="center", fontsize=10)
ax3.set(xlim=(0.9, 1.0), xlabel="全连接网络的验证准确率（5 轮）")
ax3.invert_yaxis()
save(fig, "F1")

if SLIDES:  # S1: the question page shows only the two images
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.6))
    axes[0].imshow(seven, cmap="Greys")
    axes[0].set_title("原来的图")
    axes[1].imshow(seven.reshape(784)[perm].reshape(28, 28), cmap="Greys")
    axes[1].set_title("像素打乱后")
    for ax in axes:
        clean(ax)
    save(fig, "S1")

# ---------------- F2: two candidates for "the same detector everywhere" ----------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2) if SLIDES else (10, 4.2))
rng = np.random.default_rng(1)
for ax, shared, title, count in [(axes[0], False, "每个位置单独一套 3×3 权重", f"{N['local_params']:,} 个权重"),
                                 (axes[1], True, "所有位置共用同一套 3×3 权重", f"{N['shared_params']:,} 个权重")]:
    ax.imshow(seven, cmap="Greys", alpha=0.35, extent=(0, 28, 28, 0))
    cols = ["#0072B2", "#00795A", "#C84B00", "#7B3294", "#E69F00"]
    for k, (x, y) in enumerate([(3, 4), (12, 5), (20, 9), (8, 17), (17, 21)]):
        c = cols[0] if shared else cols[k]
        ax.add_patch(Rectangle((x, y), 3, 3, fill=True, facecolor=c, alpha=0.55, edgecolor=c, lw=2))
    ax.set_title(title)
    ax.text(14, 31.5, f"16 个检测器：{count}", ha="center", fontsize=11, color=R if not shared else G)
    ax.set(xlim=(0, 28), ylim=(33, 0))
    clean(ax)
save(fig, "F2")

# ---------------- F3: max pooling ----------------
grid = np.array([[0, 1, 0, 2], [3, 0, 1, 0], [0, 0, 5, 1], [2, 1, 0, 0]])
pooled = grid.reshape(2, 2, 2, 2).max(axis=(1, 3))
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4) if SLIDES else (7.6, 3.2), gridspec_kw=dict(width_ratios=[2, 1.2], wspace=0.5))
for ax, g, title in [(axes[0], grid, "特征图的一角（4×4）"), (axes[1], pooled, "每 2×2 留最大（2×2）")]:
    n = g.shape[0]
    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor="white", edgecolor=GREY))
            ax.text(j + 0.5, i + 0.5, str(g[i, j]), ha="center", va="center", fontsize=13)
    ax.set(xlim=(0, n), ylim=(n, 0), aspect="equal", title=title)
    clean(ax)
for (i, j) in [(0, 0), (0, 2), (2, 0), (2, 2)]:
    axes[0].add_patch(Rectangle((j, i), 2, 2, fill=False, edgecolor=G, lw=3))
save(fig, "F3")

# ---------------- F4: the CNN, layer by layer ----------------
steps = [("输入", "28×28×1", "0"), ("卷积\n16 个 3×3", "26×26×16", f"{C['layer_params'][0]:,}"), ("最大池化\n2×2", "13×13×16", "0"),
         ("卷积\n32 个 3×3", "11×11×32", f"{C['layer_params'][2]:,}"), ("最大池化\n2×2", "5×5×32", "0"),
         ("展平", "800", "0"), ("10 个 softmax", "10", f"{C['layer_params'][5]:,}")]
fig, ax = plt.subplots(figsize=(17, 3.4) if SLIDES else (11, 3.0))
for k, (name, shape, params) in enumerate(steps):
    x = k * 1.6
    col = G if params != "0" else B
    ax.add_patch(FancyBboxPatch((x, 0), 1.4, 1.3, boxstyle="round,pad=0.03", facecolor="white", edgecolor=col, lw=2))
    ax.text(x + 0.7, 1.02, name, ha="center", va="center", fontsize=11.5, color=col, weight="bold", linespacing=1.0)
    ax.text(x + 0.7, 0.62, shape, ha="center", va="center", fontsize=11.5)
    ax.text(x + 0.7, 0.25, f"权重 {params}", ha="center", va="center", fontsize=11, color=GREY)
    if k < len(steps) - 1:
        ax.annotate("", (x + 1.58, 0.65), (x + 1.43, 0.65), arrowprops=dict(arrowstyle="->", color=GREY))
ax.text(5.5, -0.35, f"一共 {C['params']:,} 个权重（Lesson 9 的全连接网络：{D['params']:,}）", ha="center", fontsize=11)
ax.set(xlim=(-0.1, 11.1), ylim=(-0.6, 1.4))
ax.axis("off")
save(fig, "F4")

# ---------------- F5: confusion matrix of the CNN on the validation set ----------------
E = N["errors"]
cm = np.array(E["cm"])
fig, ax = plt.subplots(figsize=(8, 7.2) if SLIDES else (5.6, 5.0))
off = cm.copy()
np.fill_diagonal(off, 0)
ax.imshow(off, cmap="Reds", vmin=0, vmax=max(10, off.max()))
for i in range(10):
    for j in range(10):
        if cm[i, j]:
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7.5 if i == j else 9,
                    color=GREY if i == j else "black")
ax.set(xticks=range(10), yticks=range(10), xlabel="预测的数字", ylabel="真实的数字")
ax.set_title(f"CNN 在验证集上：错 {E['wrong']} 张（全连接 {E['dense_wrong']} 张）")
save(fig, "F5")

# ---------------- F6: ten images the CNN gets wrong ----------------
fig, axes = plt.subplots(2, 5, figsize=(14, 6.4) if SLIDES else (10, 4.6))
for ax, ex in zip(axes.ravel(), E["examples"]):
    ax.imshow(np.array(ex["img"]), cmap="Greys")
    ax.set_title(f"真 {ex['true']} → 判 {ex['pred']}", fontsize=10)
    clean(ax)
save(fig, "F6")

# ---------------- F7: first-layer filters and feature maps ----------------
F = N["filters"]
fig, axes = plt.subplots(2, 9, figsize=(17, 4.6) if SLIDES else (11, 2.9))
axes[0, 0].imshow(np.array(F["image"]), cmap="Greys")
axes[0, 0].set_title("输入", fontsize=10)
axes[1, 0].axis("off")
for k in range(8):
    w = np.array(F["w"][k])
    v = np.abs(w).max()
    axes[0, k + 1].imshow(w, cmap="RdBu_r", vmin=-v, vmax=v)
    axes[0, k + 1].set_title(f"滤波器 {k + 1}", fontsize=9)
    axes[1, k + 1].imshow(np.array(F["maps"][k]), cmap="Greys")
for ax in axes.ravel():
    clean(ax)
fig.text(0.5, 0.02, "上排：第一层的 3×3 滤波器（红正蓝负）；下排：它们在这张 7 上的特征图（越黑越“亮”）", ha="center", fontsize=10)
save(fig, "F7")

# ---------------- F8: the whole chain in one picture ----------------
steps = [("打乱也不怕", "全连接网络不看邻居；平移 2 像素却掉到 0.870", R),
         ("局部 + 共享", "一个 3×3 滤波器到处用；108,160 → 160 个权重", G),
         ("最大池化", "每 2×2 留最大；小的平移影响变小", G),
         ("CNN", "12,810 个权重；验证 0.987；平移 2 像素仍有 0.962", B),
         ("错误分析", "混淆矩阵与常见错误；改对 214 张，新错 44 张", B)]
fig, ax = plt.subplots(figsize=(11.6 if EN else 8.4, 3.4))
ax.axis("off")
ax.set_xlim(0, 10)
ax.set_ylim(0, len(steps))
heads = [ax.text(0, len(steps) - 0.5 - k, s[0], ha="center", va="center", weight="bold", color=s[2], fontsize=13)
         for k, s in enumerate(steps)]
fig.canvas.draw()
to_data = ax.transData.inverted()
BW = max(np.ptp(to_data.transform(t.get_window_extent())[:, 0]) for t in heads) + 0.5  # box width from the longest title
for k, (h, sub, col) in enumerate(steps):
    yy = len(steps) - 1 - k
    ax.add_patch(Rectangle((0.2, yy + 0.12), BW, 0.76, facecolor=col, alpha=0.14, edgecolor=col, lw=1.4))
    heads[k].set_x(0.2 + BW / 2)
    ax.text(0.5 + BW, yy + 0.5, sub, va="center", fontsize=12.5, color="#333")
    if k < len(steps) - 1:
        ax.annotate("", xy=(0.2 + BW / 2, yy - 0.12), xytext=(0.2 + BW / 2, yy + 0.12),
                    arrowprops=dict(arrowstyle="->", color="#666", lw=1.2))
save(fig, "F8")
print("figures ok")
