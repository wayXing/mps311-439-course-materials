"""Lesson 7 Note, Advanced Study 2: PCA denoising of handwritten digits.
Components learned from the clean training images; noise (sd 0.3) added to the validation images;
each noisy image is projected onto the first K components and rebuilt. Writes F17.svg and denoise.json.
Run: uv run --no-project --python 3.12 --with scikit-learn,matplotlib python build_denoise.py
NOTE_LANG=en writes the English F17 to ../note_en_v2/ (denoise.json is then left as it is).
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
LESSONS = os.path.abspath(os.path.join(HERE, "../../../.."))
EN = os.environ.get("NOTE_LANG") == "en"
OUT = os.path.join(os.path.dirname(HERE), "note_en_v2") if EN else HERE
os.makedirs(OUT, exist_ok=True)
T = lambda zh, en: en if EN else zh
B, R, GREY = "#0072B2", "#C84B00", "#6b6b6b"
plt.rcParams.update({"font.family": T(["Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"], ["Arial", "DejaVu Sans"]),
                     "svg.fonttype": "path", "axes.unicode_minus": False, "font.size": 11, "axes.titlesize": 12,
                     "axes.spines.top": False, "axes.spines.right": False})

mn = np.load(os.path.join(LESSONS, "data/mnist10k.npz"))
MX, MY = mn["X"] / 255.0, mn["y"].astype(int)
Xtr, Xte, ytr, yte = train_test_split(MX, MY, test_size=0.25, random_state=0, stratify=MY)
Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.25, random_state=0, stratify=ytr)

SD = 0.3
pca = PCA().fit(Xa)                                   # clean training images only
noisy = Xv + np.random.default_rng(0).normal(0, SD, Xv.shape)
Zn = pca.transform(noisy)


def rebuild(k):
    return pca.mean_ + Zn[:, :k] @ pca.components_[:k]


Ks = [0, 5, 10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 300, 400, 500, 600, 784]
err = {k: float(((rebuild(k) - Xv) ** 2).sum(1).mean()) for k in Ks}
best = min(err, key=err.get)
noise_err = float(((noisy - Xv) ** 2).sum(1).mean())
out = dict(sd=SD, noise_err=round(noise_err, 1), err={k: round(v, 1) for k, v in err.items()}, best_k=best)
print(out)
if not EN:
    json.dump(out, open(os.path.join(HERE, "denoise.json"), "w"))

# ---- figure: three validation images; clean / noisy / K = 10, best, 400; then the error curve ----
idx = [5, 0, 1]
show = [10, best, 400]
cols = [T("原图", "clean"), T("加噪声", "noisy")] + [f"K = {k}" for k in show]
fig = plt.figure(figsize=(12, 3.9))
gs = fig.add_gridspec(3, 7, width_ratios=[1, 1, 1, 1, 1, 0.35, 3.4], wspace=0.08, hspace=0.12)
for r_, i in enumerate(idx):
    rec = {k: rebuild(k)[i] for k in show}
    imgs = [Xv[i], noisy[i]] + [rec[k] for k in show]
    for c, im in enumerate(imgs):
        ax = fig.add_subplot(gs[r_, c])
        ax.imshow(np.clip(im, 0, 1).reshape(28, 28), cmap="Greys", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(c == 3); s.set_color(B); s.set_linewidth(2)
        if r_ == 0:
            ax.set_title(cols[c], fontsize=11, color=B if c == 3 else "black")
ax = fig.add_subplot(gs[:, 6])
ks = [k for k in Ks if k > 0]
ax.plot(ks, [err[k] for k in ks], "-o", color=B, ms=3.5)
ax.axhline(noise_err, color=R, ls="--", lw=1.2)
ax.text(330, noise_err + 1.5, T(f"不处理：{noise_err:.1f}", f"no denoising: {noise_err:.1f}"), color=R, ha="center", va="bottom", fontsize=10)
ax.annotate(T(f"K = {best}：{err[best]:.1f}", f"K = {best}: {err[best]:.1f}"), (best, err[best]), (best + 120, err[best] - 4),
            fontsize=10, color=B, arrowprops=dict(arrowstyle="-", color=B, lw=0.8))
ax.set_xlabel(T("留几个主成分 K", "components kept K"))
ax.set_ylabel(T("和原图差多少（平方和）", "error against the clean image"))
ax.set_ylim(0, noise_err * 1.12)
fig.savefig(os.path.join(OUT, "F17.svg"), bbox_inches="tight")
if os.environ.get("PNGDIR"):
    fig.savefig(os.path.join(os.environ["PNGDIR"], "F17.png"), bbox_inches="tight", dpi=90)
