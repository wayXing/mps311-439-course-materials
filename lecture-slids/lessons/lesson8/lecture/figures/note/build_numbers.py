"""Lesson 8 Note: every number in the note, the figures, the widgets and the lab checks.
Run (from this folder):  python build_numbers.py   -> numbers.json (and prints what the note quotes)
Rules fixed in the design card before writing the note:
  blobs   hand-made: three groups of 60 points, centres (0,0), (4,0.5), (2,3.5), SD 0.6, default_rng(0)
  rings   hand-made: inner ring r ~ N(1, 0.12), 120 points; outer ring r ~ N(3, 0.15), 200 points; default_rng(0)
  strips  hand-made: two horizontal strips of 150 points, x ~ U(-4, 4), y ~ N(0, 0.25) and N(1.6, 0.25); same rng
  digits  the course MNIST subset (lessons/data/mnist10k.npz, pixels / 255), the Lesson 6 fitting part (5625 images); PCA with 50 components fitted on it (as in Assignment 2)
  K-means KMeans(n_clusters, n_init=10, random_state=0) unless a seed study says otherwise
"""
import json, os, warnings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import linkage

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUTJ = os.path.join(HERE, "numbers.json")
r = lambda v, k=3: round(float(v), k)
N = {}


def majority(true, groups):
    """Share of points whose group's most common true label is their own label."""
    ok = 0
    for g in np.unique(groups):
        ok += np.bincount(true[groups == g]).max()
    return ok / len(true)


# ---------- three blobs ----------
rng = np.random.default_rng(0)
Bx = np.r_[rng.normal([0, 0], 0.6, (60, 2)), rng.normal([4, 0.5], 0.6, (60, 2)), rng.normal([2, 3.5], 0.6, (60, 2))]
By = np.r_[np.zeros(60, int), np.ones(60, int), np.full(60, 2)]
starts = {"A": np.array([[-1.0, 3.5], [0.5, 4.2], [5.5, 3.8]]),   # converges to the three blobs
          "B": np.array([[-1.0, -1.2], [0.2, -1.6], [1.2, -1.2]])}  # gets stuck: two centres share one blob
runs_2d = {}
for key, start in starts.items():
    c = start.copy()
    steps = []
    for it in range(10):
        d = ((Bx[:, None, :] - c[None, :, :]) ** 2).sum(2)
        lab = d.argmin(1)
        steps.append(dict(centres=[[r(v) for v in row] for row in c], inertia=r(d.min(1).sum(), 1)))
        newc = np.array([Bx[lab == k].mean(0) if (lab == k).any() else c[k] for k in range(3)])
        if np.allclose(newc, c):
            break
        c = newc
    runs_2d[key] = dict(start=start.tolist(), steps=steps, final=steps[-1]["inertia"], n=len(steps))
km3 = KMeans(3, n_init=10, random_state=0).fit(Bx)
inertias = {str(k): r(KMeans(k, n_init=10, random_state=0).fit(Bx).inertia_, 1) for k in range(1, 8)}
N["blobs"] = dict(x=[r(v) for v in Bx[:, 0]], y=[r(v) for v in Bx[:, 1]], runs=runs_2d, final_inertia=r(km3.inertia_, 1), recovered=r(majority(By, km3.labels_)), inertias=inertias)

# ---------- two rings and two strips: three methods ----------
rng = np.random.default_rng(0)
a1, a2 = rng.uniform(0, 2 * np.pi, 120), rng.uniform(0, 2 * np.pi, 200)
r1, r2 = rng.normal(1, 0.12, 120), rng.normal(3, 0.15, 200)
Rx = np.r_[np.c_[r1 * np.cos(a1), r1 * np.sin(a1)], np.c_[r2 * np.cos(a2), r2 * np.sin(a2)]]
Ry = np.r_[np.zeros(120, int), np.ones(200, int)]
Sx = np.r_[np.c_[rng.uniform(-4, 4, 150), rng.normal(0, 0.25, 150)], np.c_[rng.uniform(-4, 4, 150), rng.normal(1.6, 0.25, 150)]]
Sy = np.r_[np.zeros(150, int), np.ones(150, int)]
shapes = {}
for name, X, Y in [("rings", Rx, Ry), ("strips", Sx, Sy)]:
    res = dict(x=[r(v) for v in X[:, 0]], y=[r(v) for v in X[:, 1]], truth=Y.tolist())
    for meth in ["kmeans", "single", "ward"]:
        model = KMeans(2, n_init=10, random_state=0) if meth == "kmeans" else AgglomerativeClustering(2, linkage=meth)
        lab = model.fit_predict(X)
        res[meth] = dict(labels=lab.tolist(), acc=r(majority(Y, lab)))
    shapes[name] = res
N["shapes"] = shapes

# ---------- a small sample of the blobs for the dendrogram ----------
pick = np.r_[np.arange(0, 60, 8), np.arange(60, 120, 8), np.arange(120, 180, 8)]
small = Bx[pick]
Z = linkage(small, method="ward")
N["dendro"] = dict(x=[r(v) for v in small[:, 0]], y=[r(v) for v in small[:, 1]], group=By[pick].tolist(),
                   Z=[[r(v, 4) for v in row] for row in Z], n=len(pick))

# ---------- digits: PCA(50) then clustering ----------
_mn = np.load(os.path.join(HERE, "../../../../data/mnist10k.npz"))  # the course MNIST subset: 10,000 images of 28 x 28
MX, MY = _mn["X"] / 255.0, _mn["y"].astype(int)  # pixels scaled to 0 to 1; digits 0 to 9
Dtr, Dte, dtr, dte = train_test_split(MX, MY, test_size=0.25, random_state=0, stratify=MY)
Da, Dv, da, dv = train_test_split(Dtr, dtr, test_size=0.25, random_state=0, stratify=dtr)
pca = PCA(50, random_state=0).fit(Da)
Za, Zv = pca.transform(Da), pca.transform(Dv)
km = KMeans(10, n_init=10, random_state=0).fit(Za)
table = np.zeros((10, 10), int)
for g, t in zip(km.labels_, da):
    table[g, t] += 1
order = np.argsort(table.argmax(1) * 1000 - table.max(1))  # sort clusters by their main digit
table = table[order]
centres_img = pca.inverse_transform(km.cluster_centers_[order])
maj_digit = table.argmax(1)
inv = np.empty(10, int)
inv[order] = np.arange(10)
val_pred = maj_digit[inv[km.predict(Zv)]]
seeds = []
for s in range(5):
    k1 = KMeans(10, n_init=1, random_state=s).fit(Za)
    seeds.append(dict(seed=s, inertia=r(k1.inertia_, 1), majority=r(majority(da, k1.labels_))))
hier = {}
for link in ["ward", "single", "average", "complete"]:
    lab = AgglomerativeClustering(10, linkage=link).fit_predict(Za)
    hier[link] = dict(majority=r(majority(da, lab)), largest=int(np.bincount(lab).max()))
lab_ward = AgglomerativeClustering(10, linkage="ward").fit_predict(Za)
inertia_k = {str(k): r(KMeans(k, n_init=10, random_state=0).fit(Za).inertia_, 0) for k in [2, 5, 8, 10, 12, 15, 20]}
N["digits"] = dict(fit=len(da), val=len(dv), pca_cum=r(pca.explained_variance_ratio_.sum()), table=table.tolist(),
                   maj_digit=maj_digit.tolist(), majority_fit=r(majority(da, km.labels_)), majority_val=r((val_pred == dv).mean()),
                   inertia=r(km.inertia_, 1), seeds=seeds, hier=hier, centres=[[r(v) for v in c] for c in centres_img],
                   missing=[int(d) for d in range(10) if d not in maj_digit], doubled=[int(d) for d in range(10) if (maj_digit == d).sum() > 1],
                   inertia_k=inertia_k)
# memory of hierarchical clustering: one number (8 bytes) per pair of points
pairs = lambda n: n * (n - 1) / 2
N["memory"] = dict(n_digits=len(da), gb_digits=r(pairs(len(da)) * 8 / 1e9, 4), gb_mnist=r(pairs(50000) * 8 / 1e9, 1),
                   gb_5000=r(pairs(5000) * 8 / 1e9, 2))
# a random half of the fitting images: does hierarchical clustering still find the digits?
sub = np.random.default_rng(0).choice(len(da), 500, replace=False)
lab_sub = AgglomerativeClustering(10, linkage="ward").fit_predict(Za[sub])
N["subsample"] = dict(n=500, majority=r(majority(da[sub], lab_sub)))

# ---------- Advanced: K-means in numpy, ten starts ----------
best = None
runs = []
for s in range(10):
    rr = np.random.default_rng(s)
    cc = Za[rr.choice(len(Za), 10, replace=False)]
    for it in range(100):
        lab = ((Za[:, None, :] - cc[None, :, :]) ** 2).sum(2).argmin(1)
        new = np.array([Za[lab == k].mean(0) if (lab == k).any() else cc[k] for k in range(10)])
        if np.allclose(new, cc):
            break
        cc = new
    inert = float(((Za - cc[lab]) ** 2).sum())
    runs.append(r(inert, 1))
N["adv"] = dict(runs=runs, best=min(runs), worst=max(runs), sklearn=r(km.inertia_, 1))

json.dump(N, open(OUTJ, "w"), ensure_ascii=False, indent=1)
show = {"blobs": {k: v for k, v in N["blobs"].items() if k not in ("x", "y", "runs")},
        "shapes": {k: {m: v[m]["acc"] for m in ("kmeans", "single", "ward")} for k, v in N["shapes"].items()},
        "digits": {k: v for k, v in N["digits"].items() if k not in ("centres",)}, "memory": N["memory"], "subsample": N["subsample"], "adv": N["adv"]}
show["blobs"]["runs"] = {k: [s["inertia"] for s in v["steps"]] for k, v in N["blobs"]["runs"].items()}
print(json.dumps(show, ensure_ascii=False, indent=1))
