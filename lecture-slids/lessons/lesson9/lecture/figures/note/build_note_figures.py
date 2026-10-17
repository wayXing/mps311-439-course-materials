"""Lesson 9 Note: static figures. Reads numbers.json written by build_numbers.py.
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
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, FancyBboxPatch

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
O = "#E69F00"  # second class of the ring (outer), as in Lesson 4
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


Rg = N["ring"]
rx, ry, rl = np.array(Rg["x"]), np.array(Rg["y"]), np.array(Rg["label"])


def ring(ax, s=16):
    ax.scatter(rx[rl == 1], ry[rl == 1], s=s, color=B, label="内圈（类 1）")
    ax.scatter(rx[rl == 0], ry[rl == 0], s=s, color=O, label="外圈（类 0）")
    ax.set(aspect="equal", xlim=(-3.6, 3.6), ylim=(-3.6, 3.6), xticks=[-3, 0, 3], yticks=[-3, 0, 3], xlabel="x₁", ylabel="x₂")


# ---------------- F1: one straight line fails; a hand-made feature works ----------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
ring(axes[0])
axes[0].axhline(0, color=G, lw=2.5)
axes[0].set_title(f"一条直线：准确率 {Rg['acc_line']:.2f}")
axes[0].legend(loc="upper right", fontsize=9, frameon=False)
dist = np.sqrt(rx ** 2 + ry ** 2)
bins = np.linspace(0, 3.6, 30)
axes[1].hist(dist[rl == 1], bins=bins, color=B, alpha=0.8)
axes[1].hist(dist[rl == 0], bins=bins, color=O, alpha=0.8)
axes[1].axvline(1.7, color=G, lw=2.5)
axes[1].set(xlabel="人想出的特征：到中心的距离", ylabel="点数")
axes[1].set_title(f"先想到这个特征：准确率 {Rg['acc_dist']:.2f}")
save(fig, "F1")


# ---------------- F2: the network as a picture ----------------
def node(ax, x, y, txt, col, r=0.32):
    ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=col, lw=2.2))
    ax.text(x, y, txt, ha="center", va="center", fontsize=12, color=col)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color=GREY, lw=1.1, shrinkA=14, shrinkB=14))


fig, ax = plt.subplots(figsize=(9.6, 4.6))
ins, hid, out = [(0.3, 2.6), (0.3, 1.0)], [(4.3, 3.1), (4.3, 1.8), (4.3, 0.5)], (8.3, 1.8)
for a in ins:
    for h in hid:
        arrow(ax, a, h)
for h in hid:
    arrow(ax, h, out)
for (x, y), t in zip(ins, ["x₁", "x₂"]):
    node(ax, x, y, t, B)
for k, (x, y) in enumerate(hid):
    node(ax, x, y, f"h{'₁₂₃'[k]}", G)
node(ax, *out, "p", G)
ax.text(0.3, 4.0, "输入：\n两个坐标", ha="center", va="bottom", fontsize=11)
ax.text(4.3, 4.0, "中间：\n3 个小 logistic", ha="center", va="bottom", fontsize=11, color=G)
ax.text(8.3, 4.0, "输出：\n再一个 logistic", ha="center", va="bottom", fontsize=11, color=G)
ax.text(4.3, -0.2, "每个：一条直线 + sigmoid\n给出一个新特征", ha="center", va="top", fontsize=10, color=GREY)
ax.text(7.9, 1.3, "用 3 个新特征\n给出内圈的概率", ha="left", va="top", fontsize=10, color=GREY)
ax.set(xlim=(-0.8, 11.0), ylim=(-1.4, 5.0), aspect="equal")
ax.axis("off")
save(fig, "F2")

# ---------------- F3: two activation functions ----------------
z = np.linspace(-5, 5, 300)
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))
axes[0].plot(z, 1 / (1 + np.exp(-z)), color=G, lw=2.5)
axes[0].set(title="sigmoid：压到 0 和 1 之间", xlabel="z", ylim=(-0.1, 1.1))
axes[1].plot(z, np.maximum(0, z), color=G, lw=2.5)
axes[1].set(title="ReLU：负的变成 0", xlabel="z", ylim=(-0.5, 5.2))
for ax in axes:
    ax.axhline(0, color=LIGHT, lw=1, zorder=0)
    ax.axvline(0, color=LIGHT, lw=1, zorder=0)
save(fig, "F3")

# ---------------- F4: MNIST, validation accuracy ----------------
M = N["mnist"]
H = M["hist"]
names = ["logistic regression（同一验证集）", "网络，去掉中间的激活（5 轮）", "网络（5 轮）", "网络（20 轮）"]
vals = [M["lr_val"], M["linear_val5"][-1], M["relu_val5"][-1], H["val_accuracy"][-1]]
colours = [GREY, LIGHT, G, G]
fig, ax = plt.subplots(figsize=(7.8, 2.9))
ax.barh(names, vals, color=colours, height=0.55)
for k, v in enumerate(vals):
    ax.text(v + 0.002, k, f"{v:.4f}", va="center", fontsize=10)
ax.set(xlim=(0.88, 1.0), xlabel="验证集准确率（10,000 张）")
ax.invert_yaxis()
save(fig, "F4")

# ---------------- F5: training and validation curves, 20 epochs ----------------
ep = np.arange(1, 21)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
axes[0].plot(ep, H["loss"], "o-", color=B, ms=4, label="训练")
axes[0].plot(ep, H["val_loss"], "o-", color=R, ms=4, label="验证")
k = M["best_val_loss_epoch"]
axes[0].annotate(f"验证损失最低：第 {k} 轮", (k, M["best_val_loss"]), xytext=(2, 0.2), fontsize=10,
                 arrowprops=dict(arrowstyle="->", color=GREY))
axes[0].set(xlabel="轮（epoch）", ylabel="交叉熵损失", title="损失", xticks=[1, 5, 10, 15, 20])
axes[1].plot(ep, H["accuracy"], "o-", color=B, ms=4, label="训练")
axes[1].plot(ep, H["val_accuracy"], "o-", color=R, ms=4, label="验证")
axes[1].set(xlabel="轮（epoch）", ylabel="准确率", title="准确率", ylim=(0.92, 1.0), xticks=[1, 5, 10, 15, 20])
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
save(fig, "F5")

if SLIDES:  # S1: the answer page: the gap between training and validation loss
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    ax.plot(ep, H["loss"], "o-", color=B, ms=4, label="训练")
    ax.plot(ep, H["val_loss"], "o-", color=R, ms=4, label="验证")
    ax.fill_between(ep, H["loss"], H["val_loss"], where=ep >= k, color=R, alpha=0.12)
    ax.axvline(k, color=GREY, ls="--", lw=1)
    ax.text(k + 0.3, 0.23, f"验证损失最低：第 {k} 轮", fontsize=10, color=GREY)
    ax.text(14, 0.07, "差距越拉越大", fontsize=10, color=R, ha="center")
    ax.set(xlabel="轮（epoch）", ylabel="交叉熵损失", xticks=[1, 5, 10, 15, 20])
    ax.legend(frameon=False, loc="upper right")
    save(fig, "S1")

# ---------------- F6: what the first-layer weights look like ----------------
rows_, cols_ = 2, 5                            # two rows: tiles large enough to read
fig, axes = plt.subplots(rows_, cols_, figsize=(9, 4.0))
for k, ax in enumerate(axes.ravel()):
    w = np.array(M["weights"][k]).reshape(28, 28)
    v = np.abs(w).max()
    ax.imshow(w, cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_title(f"单元 {k + 1}", fontsize=12)
    ax.set(xticks=[], yticks=[])
    for sp in ax.spines.values():
        sp.set_visible(False)
sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=matplotlib.colors.Normalize(-1, 1))  # one colour bar for all units
cb = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, ticks=[-1, 0, 1])
cb.ax.set_yticklabels(["蓝：这里有墨水，\n单元的加权和变小", "0", "红：这里有墨水，\n单元的加权和变大"], fontsize=11)
save(fig, "F6")

# ---------------- F7: the whole chain in one picture ----------------
steps = [("特征要人想", "环：一条直线 50%；先想到“距离”才 99%", B),
         ("几个小 logistic", "每个一条直线；给一个新特征，再组合", G),
         ("中间要弯一下", "去掉激活；仍是一条直线（50.3%）", R),
         ("训练", "交叉熵 + 梯度下降；误差往回传，一小批一步", G),
         ("MNIST", "92.8% → 97.8%；验证损失第 4 轮最低", B)]
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
save(fig, "F7")
print("figures ok")
