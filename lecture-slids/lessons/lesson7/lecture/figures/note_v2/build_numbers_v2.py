"""Lesson 7 Note V2: every number in the note, the figures and the widgets.
Run (from this folder):  python build_numbers_v2.py   -> numbers_v2.json (and prints what the note quotes)
Fixed in design_card_v2.md:
  arm     synthetic height and arm span, 200 people, default_rng(0):
          size ~ N(0, 7^2); height = 170 + size + N(0, 1.5^2); arm = 170 + size + N(0, 2^2)  (cm)
          variances and covariances divide by n - 1 (as scikit-learn's explained_variance_)
  ring    200 points on a circle of radius 5 plus N(0, 0.3^2) noise, same generator after arm
  cars    the Auto MPG file used in the linear-regression lessons (392 cars), five measurements
  digits  the course MNIST subset (10,000 images, pixels / 255); 2,500 kept as the test set,
          the other 7,500 split 3:1 into 5,625 training (fitting) images and 1,875 validation images.
          PCA is fitted on the 5,625 training images only.
"""
import json, os, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import sklearn

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
LESSONS = os.path.join(HERE, "../../../..")
r = lambda v, k=3: round(float(v), k)
N = {"sklearn": sklearn.__version__}

# ---------- height and arm span ----------
rng = np.random.default_rng(0)
n = 200
size = rng.normal(0, 7, n)
h = 170 + size + rng.normal(0, 1.5, n)
a = 170 + size + rng.normal(0, 2.0, n)
X = np.c_[h, a]
m = X.mean(0)
Xc = X - m
S = Xc.T @ Xc / (n - 1)
tot = float(np.trace(S))
vals, vecs = np.linalg.eigh(S)
vals, vecs = vals[::-1], vecs[:, ::-1]
if vecs[0, 0] < 0:
    vecs[:, 0] *= -1
if vecs[1, 1] < 0:
    vecs[:, 1] *= -1
angles = list(range(0, 180))
spread, error = [], []
for t in angles:
    u = np.array([np.cos(np.radians(t)), np.sin(np.radians(t))])
    v = float(u @ S @ u)
    z = Xc @ u
    err = float(((Xc - np.outer(z, u)) ** 2).sum() / (n - 1))
    assert abs(v + err - tot) < 1e-9  # Pythagoras, point by point
    spread.append(r(v, 2))
    error.append(r(err, 2))
theta = float(0.5 * np.degrees(np.arctan2(2 * S[0, 1], S[0, 0] - S[1, 1])))
u1 = vecs[:, 0]
assert abs(np.degrees(np.arctan2(u1[1], u1[0])) - theta) < 1e-6
z1, z2 = Xc @ vecs[:, 0], Xc @ vecs[:, 1]
pca = PCA().fit(X)
assert np.allclose(pca.explained_variance_, vals)
u45 = np.array([1, 1]) / np.sqrt(2)
N["arm"] = dict(h=[r(v, 2) for v in h], a=[r(v, 2) for v in a], mean=[r(v, 1) for v in m],
                corr=r(np.corrcoef(h, a)[0, 1], 2), S=[[r(v, 2) for v in row] for row in S], total=r(tot, 2),
                angles=angles, spread=spread, error=error, theta=r(theta, 1),
                lam=[r(v, 2) for v in vals], share=[r(v / tot, 3) for v in vals], u1=[r(v, 3) for v in u1], u2=[r(v, 3) for v in vecs[:, 1]],
                share_height=r(S[0, 0] / tot, 3), share_arm=r(S[1, 1] / tot, 3), share_45=r(u45 @ S @ u45 / tot, 3),
                corr_z1z2=r(np.corrcoef(z1, z2)[0, 1], 3), theta_min=r(theta + 90, 1),
                tan2=r(2 * S[0, 1] / (S[0, 0] - S[1, 1]), 2))

# ---------- a ring ----------
t = rng.uniform(0, 2 * np.pi, n)
Rg = np.c_[np.cos(t), np.sin(t)] * 5 + rng.normal(0, 0.3, (n, 2))
pr = PCA().fit(Rg)
N["ring"] = dict(x=[r(v, 2) for v in Rg[:, 0]], y=[r(v, 2) for v in Rg[:, 1]], share=[r(v, 3) for v in pr.explained_variance_ratio_],
                 u1=[r(v, 3) for v in pr.components_[0]])

# ---------- cars ----------
cars = pd.read_csv(os.path.join(LESSONS, "lesson2/lab/auto_mpg.csv")).dropna()
cols = ["cylinders", "displacement", "horsepower", "weight", "acceleration"]
C = cars[cols].astype(float)
corr = C.corr()
raw = PCA().fit(C.values)
sc = StandardScaler().fit(C.values)
Zs = sc.transform(C.values)
ps = PCA().fit(Zs)
T = ps.transform(Zs)
w1 = ps.components_[0] * (1 if ps.components_[0][3] > 0 else -1)  # sign: heavier car = larger score
T1 = Zs @ w1
w2 = ps.components_[1] * (1 if ps.components_[1][4] > 0 else -1)
rw1 = raw.components_[0] * (1 if raw.components_[0][3] > 0 else -1)
N["cars"] = dict(n=len(C), cols=cols, mean=[r(v, 1) for v in C.mean()], std=[r(v, 1) for v in C.std()],
                 corr=[[r(v, 2) for v in row] for row in corr.values],
                 corr_min4=r(corr.values[:4, :4][np.triu_indices(4, 1)].min(), 2), corr_max4=r(corr.values[:4, :4][np.triu_indices(4, 1)].max(), 2),
                 raw_w1=[r(v, 3) for v in rw1], raw_share=[r(v, 4) for v in raw.explained_variance_ratio_],
                 w1=[r(v, 2) for v in w1], w2=[r(v, 2) for v in w2], share=[r(v, 3) for v in ps.explained_variance_ratio_],
                 cum2=r(ps.explained_variance_ratio_[:2].sum(), 3),
                 pc1=[r(v, 3) for v in T1], mpg=[r(v, 1) for v in cars.mpg],
                 corr_pc1_mpg=r(np.corrcoef(T1, cars.mpg)[0, 1], 2), corr_hp_mpg=r(np.corrcoef(cars.horsepower, cars.mpg)[0, 1], 2),
                 corr_weight_mpg=r(np.corrcoef(cars.weight, cars.mpg)[0, 1], 2))

# ---------- digits ----------
mn = np.load(os.path.join(LESSONS, "data/mnist10k.npz"))
MX, MY = mn["X"] / 255.0, mn["y"].astype(int)
Xtr, Xte, ytr, yte = train_test_split(MX, MY, test_size=0.25, random_state=0, stratify=MY)
Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.25, random_state=0, stratify=ytr)
assert (len(ya), len(yv), len(yte)) == (5625, 1875, 2500)
full = PCA().fit(Xa)
evr, cum = full.explained_variance_ratio_, np.cumsum(full.explained_variance_ratio_)
KW = 150
img = 5  # a 6 from the validation set
assert yv[img] == 6
Zv = full.transform(Xv[[img]])[0]
def rebuild(k):
    return full.mean_ + Zv[:k] @ full.components_[:k]
ks_img = [1, 2, 5, 10, 20, 50, 100]
rec_err = {str(k): r(((Xv[img] - rebuild(k)) ** 2).sum(), 1) for k in ks_img}
# average reconstruction error over the validation set
rec_val = {}
Zall = full.transform(Xv)
for k in [10, 50, 100]:
    R_ = full.mean_ + Zall[:, :k] @ full.components_[:k]
    rec_val[str(k)] = r(((Xv - R_) ** 2).sum(1).mean(), 1)
p2 = PCA(n_components=2).fit(Xa)
Z2 = p2.transform(Xa)
# extended reading: classify after PCA, choose K on the validation set
ext = {}
for k in [2, 10, 20, 40, 50, 100]:
    p = PCA(n_components=k).fit(Xa)
    A, V = p.transform(Xa), p.transform(Xv)
    row = dict(lr=r(LogisticRegression(max_iter=3000).fit(A, ya).score(V, yv)))
    try:
        row["qda"] = r(QuadraticDiscriminantAnalysis().fit(A, ya).score(V, yv))
    except Exception as e:  # scikit-learn >= 1.8 refuses a covariance matrix it cannot invert
        row["qda"] = None
    ext[str(k)] = row
try:
    qda784 = r(QuadraticDiscriminantAnalysis().fit(Xa, ya).score(Xv, yv))
except Exception as e:
    qda784 = "error: " + type(e).__name__
lr784 = r(LogisticRegression(max_iter=3000).fit(Xa, ya).score(Xv, yv))
best = max([k for k in ext if ext[k]["qda"] is not None], key=lambda k: ext[k]["qda"])
pb = PCA(n_components=int(best)).fit(Xa)
test_qda = r(QuadraticDiscriminantAnalysis().fit(pb.transform(Xa), ya).score(pb.transform(Xte), yte))
# Advanced: covariance eigenvectors are the components; SVD gives the same
S784 = np.cov(Xa.T)
ev, evec = np.linalg.eigh(S784)
ev, evec = ev[::-1], evec[:, ::-1]
diff = max(min(np.abs(evec[:, k] - full.components_[k]).max(), np.abs(evec[:, k] + full.components_[k]).max()) for k in range(5))
U, sv, Vt = np.linalg.svd(Xa - Xa.mean(0), full_matrices=False)
zero_pixels = int((Xa.var(0) == 0).sum())
# Advanced: training reconstruction error with K = 50 equals the sum of the dropped eigenvalues
Za50 = full.transform(Xa)[:, :50]
rec50 = full.mean_ + Za50 @ full.components_[:50]
train_err50 = float(((Xa - rec50) ** 2).sum() / (len(Xa) - 1))
rest50 = float(full.explained_variance_[50:].sum())
assert abs(train_err50 - rest50) < 1e-6 * rest50
total_var = float(np.trace(S784))
N["digits"] = dict(fit=len(ya), val=len(yv), test=len(yte), evr=[r(v, 5) for v in evr[:200]], cum=[r(v, 5) for v in cum[:KW]],
                   cum_all=[r(v, 4) for v in cum[:300]],
                   k90=int(np.argmax(cum >= 0.9) + 1), k95=int(np.argmax(cum >= 0.95) + 1),
                   cum_at={str(k): r(cum[k - 1], 3) for k in [1, 2, 5, 10, 20, 50, 100]},
                   mean=[r(v) for v in full.mean_], comps6=[[r(v) for v in c] for c in full.components_[:6]],
                   comps=[[r(v, 3) for v in c] for c in full.components_[:KW]],
                   img_label=int(yv[img]), original=[r(v) for v in Xv[img]], z=[r(v, 3) for v in Zv[:KW]],
                   z_first=[r(v, 2) for v in Zv[:3]],
                   extra_imgs=[[r(v) for v in Xv[i]] for i in (0, 1, 2)], extra_labels=[int(yv[i]) for i in (0, 1, 2)],
                   extra_z=[[r(v, 3) for v in full.transform(Xv[[i]])[0][:KW]] for i in (0, 1, 2)],
                   recs={str(k): [r(v) for v in rebuild(k)] for k in ks_img}, rec_err=rec_err, rec_val=rec_val,
                   pc2_x=[r(v, 2) for v in Z2[:, 0]], pc2_y=[r(v, 2) for v in Z2[:, 1]], pc2_label=ya.tolist(),
                   pc2_share=r(p2.explained_variance_ratio_.sum(), 3),
                   ext=ext, qda784=qda784, lr784=lr784, best_k=int(best), test_qda=test_qda,
                   eig_diff=float(f"{diff:.1e}"), svd_var_first=r(sv[0] ** 2 / (len(Xa) - 1), 4), pca_var_first=r(full.explained_variance_[0], 4),
                   zero_pixels=zero_pixels, n_zero_eig=int((ev < 1e-10).sum()),
                   rest50=r(rest50, 2), train_err50=r(train_err50, 2), total_var=r(total_var, 2), top50=r(full.explained_variance_[:50].sum(), 2))


# ---- V7 additions: the two least-squares lines, and a 3-D cloud near a plane ----
Sa = np.cov(np.c_[h, a].T)
N["arm"]["reg_h2a"] = r(np.degrees(np.arctan(Sa[0, 1] / Sa[0, 0])), 1)   # arm span fitted from height (vertical distances)
N["arm"]["reg_a2h"] = r(np.degrees(np.arctan(Sa[1, 1] / Sa[0, 1])), 1)   # height fitted from arm span (horizontal distances)
rg3 = np.random.default_rng(7)
n3 = 150
Z3 = np.c_[rg3.normal(0, 4, n3), rg3.normal(0, 2, n3), rg3.normal(0, 0.45, n3)]
Q, _ = np.linalg.qr(rg3.normal(size=(3, 3)))
P3 = Z3 @ Q.T
P3 -= P3.mean(0)
ev3, V3 = np.linalg.eigh(np.cov(P3.T))
ev3, V3 = ev3[::-1], V3[:, ::-1]
N["plane3d"] = dict(x=[[r(v, 3) for v in row] for row in P3], lam=[r(v, 3) for v in ev3], share=[r(v / ev3.sum(), 3) for v in ev3],
                    v=[[r(v, 4) for v in V3[:, j]] for j in range(3)], cum2=r((ev3[0] + ev3[1]) / ev3.sum(), 3))

json.dump(N, open(os.path.join(HERE, "numbers_v2.json"), "w"), ensure_ascii=False)
skip = {"x", "v", "h", "a", "angles", "spread", "error", "x", "y", "pc1", "mpg", "evr", "cum", "cum_all", "mean", "comps6", "comps",
        "original", "z", "extra_imgs", "extra_z", "recs", "pc2_x", "pc2_y", "pc2_label", "corr"}
print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk not in skip} if isinstance(v, dict) else v) for k, v in N.items()},
                 ensure_ascii=False, indent=1))
print("arm spread at 0/30/45/60/90/135:", [N["arm"]["spread"][t] for t in (0, 30, 45, 60, 90, 135)])
print("evr first 6:", N["digits"]["evr"][:6])
