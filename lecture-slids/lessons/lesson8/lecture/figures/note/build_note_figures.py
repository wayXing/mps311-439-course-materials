"""Lesson 8 Note: static figures. Reads numbers.json written by build_numbers.py.
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
from matplotlib.patches import Rectangle

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
YACC = "Majority-vote accuracy" if EN else "多数投票准确率"
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




Bl = N["blobs"]
bx, by = np.array(Bl["x"]), np.array(Bl["y"])

# ---------------- F1: points without labels ----------------
fig, ax = plt.subplots(figsize=(5.6, 4.6))
ax.scatter(bx, by, s=22, color="#888", alpha=0.8)
ax.set(aspect="equal", xlabel="x₁", ylabel="x₂", title="180 个点，没有标签")
save(fig, "F1")

# ---------------- F2: within-group spread against k (elbow) ----------------
ks = [int(k) for k in Bl["inertias"]]
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.plot(ks, [Bl["inertias"][str(k)] for k in ks], "o-", color=G, lw=2.2)
ax.annotate("k = 3 之后下降变慢", (3, Bl["inertias"]["3"]), xytext=(20, 40), textcoords="offset points", fontsize=11, color=G,
            arrowprops=dict(arrowstyle="->", color=G))
ax.set(xlabel="群的个数 k", ylabel="群内散布\n（距离平方之和）", xticks=ks)
save(fig, "F2")

# ---------------- F3: a dendrogram of a small sample ----------------
from scipy.cluster.hierarchy import dendrogram
Dd = N["dendro"]
Z = np.array(Dd["Z"])
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2), gridspec_kw=dict(width_ratios=[1, 1.6], wspace=0.25))
cols = [B, G, R]
sx, sy, sg = np.array(Dd["x"]), np.array(Dd["y"]), np.array(Dd["group"])
pos = np.c_[sx, sy] + 0.0
span = max(np.ptp(sx), np.ptp(sy))
for _ in range(800):                                   # push the numbers apart so none sits on another
    for i in range(len(sx)):
        for j in range(len(sx)):
            d = pos[i] - pos[j]
            if i != j and np.hypot(*d) < 0.11 * span:
                pos[i] += 0.004 * span * d / (np.hypot(*d) + 1e-9)
for k in range(len(sx)):
    axes[0].scatter([sx[k]], [sy[k]], color=cols[sg[k]], s=40)
    axes[0].plot([sx[k], pos[k][0]], [sy[k], pos[k][1]], color=cols[sg[k]], lw=0.6, alpha=0.6)
    axes[0].text(pos[k][0], pos[k][1], str(k), fontsize=10, color=cols[sg[k]], ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))  # the same number as its leaf
axes[0].set(aspect="equal", xlabel="x₁", ylabel="x₂", title="24 个点" if SLIDES else f"从三团里各取几个点（{Dd['n']} 个）")
dendrogram(Z, ax=axes[1], color_threshold=0, above_threshold_color="#555", leaf_font_size=10, leaf_rotation=90, labels=[str(k) for k in range(len(sx))])
for t in axes[1].get_xticklabels():  # colour each leaf like its point on the left
    t.set_color(cols[sg[int(t.get_text())]])
cut = (Z[-3, 2] + Z[-2, 2]) / 2
axes[1].axhline(cut, color=R, ls="--", lw=1.6)
axes[1].text(len(sx) * 10 * 0.98, cut * 1.04, "在这里横切：3 群", ha="right", color=R, fontsize=11)
axes[1].set(ylabel="合并高度（Ward 距离）" if SLIDES else "合并时两群的距离（Ward）", title="树状图" if SLIDES else "树状图：每次合并最近的两群")
save(fig, "F3")

# ---------------- F4: digits, cluster against true digit ----------------
Dg = N["digits"]
T = np.array(Dg["table"])
fig, ax = plt.subplots(figsize=(6.6, 5.6))
ax.imshow(T, cmap="Greens")
for i in range(10):
    for j in range(10):
        if T[i, j]:
            ax.text(j, i, T[i, j], ha="center", va="center", fontsize=9, color="white" if T[i, j] > 0.45 * T.max() else "#222")
ax.set_xticks(range(10))
ax.set_yticks(range(10), [f"群 {k + 1}" for k in range(10)])
ax.set(xlabel="真实的数字（聚类时没用）", title="K-means（k = 10）的计数表（空格为 0）")
save(fig, "F4")

# ---------------- F5: the centre of each cluster ----------------
fig, axes = plt.subplots(1, 10, figsize=(12.0, 1.9))
for k in range(10):
    axes[k].imshow(np.array(Dg["centres"][k]).reshape(28, 28), cmap="Greys", vmin=0, vmax=1)
    axes[k].set_xticks([])
    axes[k].set_yticks([])
    axes[k].set_title(f"群 {k + 1}\n多数：{Dg['maj_digit'][k]}", fontsize=12)
save(fig, "F5")

# ---------------- F6: digits, how the distance between groups is measured ----------------
H = Dg["hier"]
names = ["K-means", "层次聚类：Ward", "层次聚类：average", "层次聚类：complete", "层次聚类：最近两点（single）"]
vals = [Dg["majority_fit"], H["ward"]["majority"], H["average"]["majority"], H["complete"]["majority"], H["single"]["majority"]]
colours = [G, B, LIGHT, LIGHT, R]
if SLIDES:  # the slides only name single and Ward
    names, vals, colours = [names[0], names[1], names[4]], [vals[0], vals[1], vals[4]], [colours[0], colours[1], colours[4]]
fig, ax = plt.subplots(figsize=(7.6, 3.4))
ax.barh(names, vals, color=colours, height=0.55)
for k, v in enumerate(vals):
    ax.text(v + 0.01, k, f"{v:.3f}", va="center", fontsize=10)
ax.text(0.35, len(vals) - 1, f"一个群吞掉 {H['single']['largest']} 张图", va="center", fontsize=10, color=R)
ax.set(xlim=(0, 1), xlabel=YACC)
ax.invert_yaxis()
save(fig, "F6")

# ---------------- F7: the whole chain in one picture ----------------
steps = [("没有标签", "只有点：有没有自然的几群？", B),
         ("什么叫一群", "离同一个中心近；群内散布越小越好", G),
         ("K-means", "分给最近的中心 ↔ 中心挪到平均，交替到停", G),
         ("失败的形状", "环、长带：按中心切，切线是直的", R),
         ("层次聚类", "逐步合并最近的两群；“多近”怎么量会改变结果", G),
         ("检查与规模", "计数表、多数投票；数据太多先子采样", B)]
fig, ax = plt.subplots(figsize=(11.6 if EN else 8.4, 4.0))
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

if SLIDES:
    # S1: within-cluster sum of squares after each round, for the two starting points
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for key, col, name in [("A", G, "起点 A"), ("B", R, "起点 B")]:
        v = [st["inertia"] for st in Bl["runs"][key]["steps"]]
        ax.plot(range(1, len(v) + 1), v, "o-", color=col, lw=2.4, label=f"{name}：停在 {v[-1]:.0f}")
    ax.set(xlabel="第几轮", ylabel="群内散布（对数刻度）", yscale="log", xticks=[1, 2, 3, 4])
    ax.legend(frameon=False)
    save(fig, "S1")
    # S2: one grouping drawn with each point joined to its centre (the score question)
    from sklearn.cluster import KMeans as _KM
    _km = _KM(3, n_init=10, random_state=0).fit(np.c_[bx, by])
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    for k in range(len(bx)):
        cc = _km.cluster_centers_[_km.labels_[k]]
        ax.plot([bx[k], cc[0]], [by[k], cc[1]], color=[B, G, R][_km.labels_[k]], lw=0.6, alpha=0.5)
    ax.scatter(bx, by, s=14, c=[[B, G, R][l] for l in _km.labels_])
    ax.scatter(_km.cluster_centers_[:, 0], _km.cluster_centers_[:, 1], marker="+", s=400, color="#111", lw=3)
    ax.set(aspect="equal", xlabel="x₁", ylabel="x₂", title="一种分法：每个点连到自己的中心")
    save(fig, "S2")
    # S3: two ways to say how near two groups are
    rr = np.random.default_rng(3)
    g1 = rr.normal([0, 0], 0.45, (12, 2))
    g2 = rr.normal([3, 0.6], 0.45, (12, 2))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8))
    for ax in axes:
        ax.scatter(g1[:, 0], g1[:, 1], color=B, s=40)
        ax.scatter(g2[:, 0], g2[:, 1], color=R, s=40)
        ax.set(aspect="equal", xticks=[], yticks=[], xlim=(-1.5, 4.5), ylim=(-1.6, 2.2))
        for sp in ax.spines.values():
            sp.set_visible(False)
    dd = ((g1[:, None, :] - g2[None, :, :]) ** 2).sum(2)
    i, j = np.unravel_index(dd.argmin(), dd.shape)
    axes[0].plot([g1[i, 0], g2[j, 0]], [g1[i, 1], g2[j, 1]], color=G, lw=3)
    axes[0].set_title("最近的两点：这条线")
    allp = np.r_[g1, g2]
    for grp, col in [(g1, B), (g2, R)]:
        axes[1].scatter(*grp.mean(0), marker="+", s=300, color=col, lw=3)
    axes[1].scatter(*allp.mean(0), marker="+", s=400, color=G, lw=3)
    axes[1].set_title("Ward：合并后散布增加多少")
    axes[1].text(1.5, -1.5, "+ 蓝、红：两群各自的中心；+ 绿：合并后的中心", ha="center", fontsize=11, color="#444")
    save(fig, "S3")
print("figures ok")
