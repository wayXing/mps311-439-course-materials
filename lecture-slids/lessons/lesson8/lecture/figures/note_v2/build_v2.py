"""Lesson 8 v2: the new numbers and figures (the old ones stay in ../note/, built by build_note_figures.py).
Run (from this folder):  python build_v2.py               -> Chinese figures here, numbers_v2.json
                         python build_v2.py --lang en     -> English figures in ../note_en_v2/
                         python build_v2.py --slides      -> English, projector-size text in ../slides_v2/
New figures:
  F0  40 handwritten digits from the fitting part, no labels (the problem)
  S1  within-cluster sum of squares after each round, two starts (the solution)
  S2  one grouping, every point joined to its centre (the model)
  F7  one-page recap, nine steps
  F8  six clustering methods on three hand-made datasets (the extension)
Numbers for the Note's Advanced Study 2 (fewer colours in a photo) are written to numbers_v2.json.
"""
import json, os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text
from matplotlib.patches import Rectangle
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
SLIDES = "--slides" in sys.argv
EN = SLIDES or "en" in " ".join(sys.argv[1:]).split()
OUTDIR = os.path.join(HERE, "..", "slides_v2" if SLIDES else "note_en_v2") if EN else HERE
os.makedirs(OUTDIR, exist_ok=True)
N = json.load(open(os.path.join(HERE, "..", "note", "numbers.json")))
B, G, R = "#0072B2", "#00795A", "#C84B00"
plt.rcParams.update({
    "font.family": ["DejaVu Sans", "Hiragino Sans GB", "Arial Unicode MS"],
    "svg.fonttype": "path", "axes.unicode_minus": False, "mathtext.fontset": "dejavusans",
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11, "figure.dpi": 100,
})
if SLIDES:
    plt.rcParams.update({"font.size": 16, "axes.titlesize": 17, "axes.labelsize": 16,
                         "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14})
T = lambda zh, en: en if EN else zh


def save(fig, name):
    if SLIDES:
        for t in fig.findobj(matplotlib.text.Text):
            if t.get_text() and t.get_fontsize() < 14:
                t.set_fontsize(14)
    if EN:  # no Chinese may be left in an English figure
        for t in fig.findobj(matplotlib.text.Text):
            assert not any("一" <= ch <= "鿿" for ch in t.get_text()), (name, t.get_text())
    fig.savefig(os.path.join(OUTDIR, f"{name}.svg"), bbox_inches="tight")
    if os.environ.get("PNGDIR"):
        fig.savefig(os.path.join(os.environ["PNGDIR"], f"{name}{'_en' if EN and not SLIDES else ''}{'_s' if SLIDES else ''}.png"),
                    bbox_inches="tight", dpi=90)
    plt.close(fig)


def majority(true, groups):
    """Share of points whose group's most common true label is their own; noise (-1) counts as wrong."""
    ok = 0
    for g in np.unique(groups):
        if g >= 0:
            ok += np.bincount(true[groups == g]).max()
    return ok / len(true)


r = lambda v, k=3: round(float(v), k)
OUT = {}

# ---------------- F0: digits without labels ----------------
mn = np.load(os.path.join(HERE, "../../../../data/mnist10k.npz"))
MX, MY = mn["X"] / 255.0, mn["y"].astype(int)
Dtr, _, dtr, _ = train_test_split(MX, MY, test_size=0.25, random_state=0, stratify=MY)
Da, _, da, _ = train_test_split(Dtr, dtr, test_size=0.25, random_state=0, stratify=dtr)
fig, axes = plt.subplots(4, 10, figsize=(8.0, 3.4))
for k, ax in enumerate(axes.flat):
    ax.imshow(Da[k].reshape(28, 28), cmap="gray_r", vmin=0, vmax=1)
    ax.set(xticks=[], yticks=[])
    for sp in ax.spines.values():
        sp.set_visible(False)
fig.suptitle(T("拟合部分的前 40 张，一共 5625 张，没人写下是几", "The first 40 of 5625 images; nobody wrote down which digit"),
             y=0.99, fontsize=12)
save(fig, "F0")
OUT["F0_digits"] = da[:40].tolist()

# ---------------- S1, S2: the blobs ----------------
Bl = N["blobs"]
bx, by = np.array(Bl["x"]), np.array(Bl["y"])
fig, ax = plt.subplots(figsize=(6.6, 3.8))
for key, col, name in [("A", G, T("起点 A", "start A")), ("B", R, T("起点 B", "start B"))]:
    v = [st["inertia"] for st in Bl["runs"][key]["steps"]]
    ax.plot(range(1, len(v) + 1), v, "o-", color=col, lw=2.4, label=T(f"{name}：停在 {v[-1]:.0f}", f"{name}: stops at {v[-1]:.0f}"))
ax.set(xlabel=T("第几轮", "round"), ylabel=T("群内散布 W（对数刻度）", "within-cluster sum of squares W (log scale)"),
       yscale="log", xticks=[1, 2, 3, 4])
ax.legend(frameon=False)
save(fig, "S1")

km3 = KMeans(3, n_init=10, random_state=0).fit(np.c_[bx, by])
fig, ax = plt.subplots(figsize=(5.8, 4.8))
for k in range(len(bx)):
    cc = km3.cluster_centers_[km3.labels_[k]]
    ax.plot([bx[k], cc[0]], [by[k], cc[1]], color=[B, G, R][km3.labels_[k]], lw=0.6, alpha=0.5)
ax.scatter(bx, by, s=14, c=[[B, G, R][l] for l in km3.labels_])
ax.scatter(km3.cluster_centers_[:, 0], km3.cluster_centers_[:, 1], marker="+", s=400, color="#111", lw=3)
for j, (cx, cy) in enumerate(km3.cluster_centers_):
    ax.annotate(f"$\\mu_{j + 1}$", (cx, cy), xytext=(10, 8), textcoords="offset points", fontsize=14, weight="bold")
ax.set(aspect="equal", xlabel="x₁", ylabel="x₂", title=T("每个点连到自己的中心", "Every point joined to its own centre"))
save(fig, "S2")

# ---------------- F7: one-page recap, nine steps ----------------
steps = [(T("问题", "Problem"), T("5625 张手写数字没有标签：有没有自然的几群？", "5625 digit images, no labels: natural groups?"), B),
         (T("启发", "Insight"), T("动手交替：分给最近的中心 ↔ 中心挪到平均，分数一直降", "alternate by hand: nearest centre, then mean; the score falls"), B),
         (T("模型", "Model"), T("每群一个中心，每个点由它那群的中心代表", "one centre per group; each point is represented by its centre"), G),
         (T("损失", "Loss"), T("群内散布 W：每个点到自己中心的距离平方之和", "within-cluster sum of squares W: squared distance to own centre, added up"), G),
         (T("求解", "Solution"), T("固定一组求另一组：K-means，一定停，可能停错 → 多个起点", "fix one, solve the other: K-means; always stops, may stop badly"), G),
         (T("检验", "Test"), T("环、长带：切线是直的 → 层次聚类，single 沿着形状连", "rings, strips: straight cuts; hierarchical, single follows the shape"), R),
         (T("应用", "Application"), T("MNIST：计数表、多数投票 57.6%；Ward 59.7%，single 11.5%", "MNIST: count table, majority vote 57.6%; Ward 59.7%, single 11.5%"), B),
         (T("扩展", "Extension"), T("DBSCAN、高斯混合、谱聚类：一行调用", "DBSCAN, Gaussian mixture, spectral: one line each"), GREY := "#6b6b6b"),
         (T("局限", "Limitations"), T("k 要人给；“一群”的定义决定结果；尺度；没有对错", "k is given; the definition of a group decides; scale; no right answer"), GREY)]
fig, ax = plt.subplots(figsize=(11.6 if EN else 9.0, 5.6))
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

# ---------------- F8: six methods on three hand-made datasets ----------------
rng = np.random.default_rng(0)
Rx = None
sh = N["shapes"]
data = {T("三团", "three blobs"): (np.c_[bx, by], np.repeat([0, 1, 2], 60), 3),
        T("两个环", "two rings"): (np.c_[sh["rings"]["x"], sh["rings"]["y"]], np.array(sh["rings"]["truth"]), 2),
        T("两条带", "two strips"): (np.c_[sh["strips"]["x"], sh["strips"]["y"]], np.array(sh["strips"]["truth"]), 2)}
methods = [("KMeans", lambda k: KMeans(k, n_init=10, random_state=0)),
           ("Ward", lambda k: AgglomerativeClustering(k, linkage="ward")),
           ("single", lambda k: AgglomerativeClustering(k, linkage="single")),
           ("DBSCAN", lambda k: DBSCAN(eps=0.5, min_samples=5)),
           ("GaussianMixture", lambda k: GaussianMixture(k, random_state=0)),
           ("Spectral", lambda k: SpectralClustering(k, affinity="nearest_neighbors", n_neighbors=10, random_state=0))]
cols = [B, R, G, "#CC79A7", "#E69F00", "#56B4E9"]
fig, axes = plt.subplots(3, 6, figsize=(13.0, 6.6))
OUT["F8"] = {}
for i, (dname, (X, Y, k)) in enumerate(data.items()):
    OUT["F8"][dname] = {}
    for j, (mname, make) in enumerate(methods):
        lab = make(k).fit_predict(X)
        acc = majority(Y, lab)
        OUT["F8"][dname][mname] = dict(acc=r(acc), groups=int(len(set(lab) - {-1})), noise=int((lab == -1).sum()))
        ax = axes[i, j]
        c = ["#bbbbbb" if l < 0 else cols[l % len(cols)] for l in lab]
        ax.scatter(X[:, 0], X[:, 1], s=4, c=c)
        ax.set(aspect="equal", xticks=[], yticks=[])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ng = len(set(lab) - {-1})
        txt = f"{100 * acc:.0f}%" + (T(f"（{ng} 群）", f" ({ng} groups)") if ng != k else "")
        ax.set_title(f"{mname}\n{txt}" if i == 0 else txt, fontsize=11)
        if j == 0:
            ax.set_ylabel(dname, fontsize=12)
fig.suptitle(T("同样三份数据，六种方法；百分数是找回的比例（灰点：DBSCAN 当成噪声的点）",
               "The same three datasets, six methods; the percentage is the share recovered (grey: points DBSCAN calls noise)"),
             y=1.0, fontsize=12)
save(fig, "F8")

# ---------------- numbers for Advanced Study 2: fewer colours in a photo ----------------
if not EN:
    from sklearn.datasets import load_sample_image
    img = load_sample_image("china.jpg") / 255.0
    P = img.reshape(-1, 3)
    sample = P[np.random.default_rng(0).choice(len(P), 20000, replace=False)]
    quant = {}
    for k in [4, 16, 64]:
        kmc = KMeans(k, n_init=1, random_state=0).fit(sample)
        lab = kmc.predict(P)
        quant[str(k)] = r(((P - kmc.cluster_centers_[lab]) ** 2).sum(1).mean(), 4)
    OUT["quant"] = dict(shape=list(img.shape), pixels=int(len(P)), error=quant)
    json.dump(OUT, open(os.path.join(HERE, "numbers_v2.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in OUT.items() if k != "F0_digits"}, ensure_ascii=False, indent=1))
print("figures ok:", OUTDIR)
