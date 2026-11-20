"""Lesson 7 Note V7 extras.
(1) Within one image: the 28 rows are the samples, the 28 columns the features; PCA on the rows of that one image.
    Same four validation images as I2. Writes v7_extra.json (used by I3 and the Note).
(2) Extension: 2-D maps of the same training digits by PCA and by nonlinear methods (F16), and a Swiss roll (F15).
Run: uv run --no-project --python 3.12 --with scikit-learn,matplotlib,umap-learn python build_v7_extra.py
NOTE_LANG=en writes English F15, F16 to ../note_en_v2/ (v7_extra.json is then left as it is); without it, the Chinese version here, unchanged.
"""
import json, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding
from sklearn.datasets import make_swiss_roll

HERE = os.path.dirname(os.path.abspath(__file__))
LESSONS = os.path.abspath(os.path.join(HERE, "../../../.."))
EN = os.environ.get("NOTE_LANG") == "en"   # NOTE_LANG=en: English labels, written to ../note_en_v2/
OUT = os.path.join(os.path.dirname(HERE), "note_en_v2") if EN else HERE
os.makedirs(OUT, exist_ok=True)
T = lambda zh, en: en if EN else zh
plt.rcParams.update({"font.family": T(["Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"], ["Arial", "DejaVu Sans"]), "svg.fonttype": "path",
                     "axes.unicode_minus": False, "font.size": 11, "axes.titlesize": 12})
r = lambda v, k=3: float(round(float(v), k))

mn = np.load(os.path.join(LESSONS, "data/mnist10k.npz"))
MX, MY = mn["X"] / 255.0, mn["y"].astype(int)
Xtr, Xte, ytr, yte = train_test_split(MX, MY, test_size=0.25, random_state=0, stratify=MY)
Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.25, random_state=0, stratify=ytr)
idx = [5, 0, 1, 2]                      # the same four validation images as I2
assert yv[5] == 6

# ---- (1) one image, rows as samples ----
across = PCA().fit(Xa)
out = dict(imgs=[], labels=[int(yv[i]) for i in idx])
for i in idx:
    A = Xv[i].reshape(28, 28)
    p = PCA().fit(A)                    # 28 samples (rows), 28 features (columns)
    Z = p.transform(A)
    errs = {}
    for k in range(0, 29):
        rec = p.mean_ + Z[:, :k] @ p.components_[:k]
        errs[k] = r(((A - rec) ** 2).sum(), 2)
    out["imgs"].append(dict(mean=[r(v) for v in p.mean_], comps=[[r(v, 4) for v in row] for row in p.components_],
                            Z=[[r(v, 4) for v in row] for row in Z], err=errs, cum=[r(v) for v in np.cumsum(p.explained_variance_ratio_)]))
# the same image, across-images PCA, K scores
z6 = across.transform(Xv[[5]])[0]
acr = {k: r(((Xv[5] - (across.mean_ + z6[:k] @ across.components_[:k])) ** 2).sum(), 1) for k in (5, 10, 20, 50)}
out["across_err6"] = acr
if not EN:
    json.dump(out, open(os.path.join(HERE, "v7_extra.json"), "w"))
e6 = out["imgs"][0]["err"]
print("within-image error, the 6:", {k: e6[k] for k in (0, 1, 2, 3, 5, 10)}, " across:", acr)
print("cum share of the 6 rows:", out["imgs"][0]["cum"][:6])

# ---- (2a) Swiss roll ----
Xs, ts = make_swiss_roll(1200, noise=0.3, random_state=0)
fig = plt.figure(figsize=(13, 3.6))
ax = fig.add_subplot(1, 4, 1, projection="3d")
ax.scatter(Xs[:, 0], Xs[:, 1], Xs[:, 2], c=ts, cmap="viridis", s=5)
ax.set_title(T("三维的卷：颜色是沿卷的位置", "A 3-D roll: colour is the\nposition along the roll")); ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([]); ax.view_init(10, -70)
for j, (name, model) in enumerate([(T("PCA：只是压扁，卷还卷着", "PCA: only squashed flat,\nthe roll is still rolled up"), PCA(2)), (T("Isomap：摊平成一张纸", "Isomap: unrolled\ninto a flat sheet"), Isomap(n_neighbors=10, n_components=2)),
                                   (T("t-SNE：邻居留在一起，分成几段", "t-SNE: neighbours stay\ntogether, in several pieces"), TSNE(2, perplexity=30, random_state=0))]):
    E = model.fit_transform(Xs)
    ax = fig.add_subplot(1, 4, j + 2)
    ax.scatter(E[:, 0], E[:, 1], c=ts, cmap="viridis", s=5); ax.set_title(name); ax.set_xticks([]); ax.set_yticks([])
fig.savefig(os.path.join(OUT, "F15.svg"), bbox_inches="tight")
if os.environ.get("PNGDIR"):
    fig.savefig(os.path.join(os.environ["PNGDIR"], "F15.png"), bbox_inches="tight", dpi=80)
plt.close(fig)

# ---- (2b) the same 2,000 training digits, six ways ----
sel = np.random.default_rng(0).choice(len(Xa), 2000, replace=False)
X2, y2 = Xa[sel], ya[sel]
X50 = PCA(50, random_state=0).fit_transform(X2)
import umap
methods = [(T("PCA（线性）", "PCA (linear)"), lambda: PCA(2).fit_transform(X2)),
           ("Kernel PCA", lambda: KernelPCA(2, kernel="rbf", gamma=0.02).fit_transform(X2)),
           ("Isomap", lambda: Isomap(n_neighbors=10, n_components=2).fit_transform(X50)),
           ("Spectral embedding", lambda: SpectralEmbedding(2, n_neighbors=10, random_state=0).fit_transform(X50)),
           ("t-SNE", lambda: TSNE(2, perplexity=30, random_state=0).fit_transform(X50)),
           ("UMAP", lambda: umap.UMAP(n_components=2, random_state=0).fit_transform(X50))]
cols = plt.get_cmap("tab10")
fig, axs = plt.subplots(2, 3, figsize=(13, 8.4))
times = {}
for ax, (name, f) in zip(axs.ravel(), methods):
    t0 = time.time(); E = f(); times[name] = r(time.time() - t0, 1)
    for d in range(10):
        m = y2 == d
        ax.scatter(E[m, 0], E[m, 1], s=4, color=cols(d), alpha=0.7)
        cx, cy = np.median(E[m, 0]), np.median(E[m, 1])
        ax.text(cx, cy, str(d), fontsize=11, fontweight="bold", ha="center", va="center", color="k",
                bbox=dict(fc="white", ec=cols(d), alpha=0.8, boxstyle="round,pad=0.15"))
    ax.set_title(name); ax.set_xticks([]); ax.set_yticks([])
fig.savefig(os.path.join(OUT, "F16.svg"), bbox_inches="tight")
if os.environ.get("PNGDIR"):
    fig.savefig(os.path.join(os.environ["PNGDIR"], "F16.png"), bbox_inches="tight", dpi=80)
plt.close(fig)
print("seconds:", times)
