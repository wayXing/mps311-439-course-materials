"""Lesson 7 Note: every number in the note, the figures, the widgets and the lab checks.
Run (from this folder):  python build_numbers.py   -> numbers.json (and prints what the note quotes)
Rules fixed in the design card before writing the note:
  cloud    hand-made 2-D cloud, 200 points, N((0,0), [[3,1.6],[1.6,1.5]]), default_rng(0); spreads with ddof=0
  digits   the Lesson 4 split (pixels / 16, test 450); the 1347 training images split 3:1 again
           (Lesson 6): 1010 to fit, 337 to validate. PCA is fitted on the 1010 fitting images only.
  lumps    load_breast_cancer, the Lesson 4 split, standardised on the training set; PCA(2) on the training set
"""
import json, os, warnings
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline  # build script only: scaling and PCA refitted inside every fold
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUTJ = os.path.join(HERE, "numbers.json")
r = lambda v, k=3: round(float(v), k)
KF = KFold(5, shuffle=True, random_state=0)
N = {}

# ---------- a 2-D cloud: turning a line trades spread against reconstruction error ----------
rng = np.random.default_rng(0)
C = rng.multivariate_normal([0, 0], [[3, 1.6], [1.6, 1.5]], 200)
Cc = C - C.mean(0)
total = float((Cc ** 2).sum(1).mean())
angles = list(range(0, 180, 1))
spread, error = [], []
for a in angles:
    u = np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])
    z = Cc @ u
    spread.append(r((z ** 2).mean()))
    error.append(r(((Cc - np.outer(z, u)) ** 2).sum(1).mean()))
pc = PCA().fit(C)
pc_angle = float(np.degrees(np.arctan2(pc.components_[0, 1], pc.components_[0, 0]))) % 180
N["cloud"] = dict(x=[r(v, 5) for v in C[:, 0]], y=[r(v, 5) for v in C[:, 1]], angles=angles, spread=spread, error=error,
                  total=r(total), pc_angle=r(pc_angle, 1), evr=[r(v) for v in pc.explained_variance_ratio_],
                  best_angle=int(angles[int(np.argmax(spread))]), max_spread=max(spread), min_error=min(error),
                  x_spread=spread[0], y_spread=spread[90])
assert abs(spread[0] + error[0] - total) < 0.01
z_pc1 = Cc @ np.array([np.cos(np.radians(pc_angle)), np.sin(np.radians(pc_angle))])
second = {}
for a in (76.0, 90.0, 121.0):  # candidates for a second direction (whole degrees, as on the slider)
    z2 = Cc @ np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])
    second[str(round(a))] = dict(corr=r(np.corrcoef(z_pc1, z2)[0, 1], 2), spread=r((z2 ** 2).mean(), 2))
N["cloud_second"] = second

# ---------- digits ----------
_mn = np.load(os.path.join(HERE, "../../../../data/mnist10k.npz"))  # the course MNIST subset: 10,000 images of 28 x 28
MX, MY = _mn["X"] / 255.0, _mn["y"].astype(int)  # pixels scaled to 0 to 1; digits 0 to 9
Dtr, Dte, dtr, dte = train_test_split(MX, MY, test_size=0.25, random_state=0, stratify=MY)
Da, Dv, da, dv = train_test_split(Dtr, dtr, test_size=0.25, random_state=0, stratify=dtr)
assert (len(da), len(dv), len(dte)) == (5625, 1875, 2500)
full = PCA().fit(Da)
KW = 150  # the widgets get the first 150 components (784 x 784 would make the page far too large)
evr = full.explained_variance_ratio_
cum = np.cumsum(evr)
var = Da.var(0)
order = np.argsort(var)[::-1]
ks = [2, 5, 10, 20, 30, 40, 50, 100, 784]
table = {}
for k in ks:
    p = PCA(n_components=k, random_state=0).fit(Da)
    Za, Zv = p.transform(Da), p.transform(Dv)
    rec = p.inverse_transform(Zv)
    row = dict(cum=r(cum[k - 1]), rec_err=r(((Dv - rec) ** 2).sum(1).mean()))
    for name, model in [("lr", LogisticRegression(max_iter=2000)), ("lda", LinearDiscriminantAnalysis()),
                        ("qda", QuadraticDiscriminantAnalysis()), ("tree", DecisionTreeClassifier(max_depth=15, random_state=0))]:
        row[name] = r(model.fit(Za, da).score(Zv, dv))
    table[str(k)] = row
pix = {}
for k in [2, 10, 20]:
    keep = order[:k]
    back = np.tile(Da.mean(0), (len(Dv), 1))
    back[:, keep] = Dv[:, keep]  # keep k pixels, fill the others with their average
    pix[str(k)] = dict(share=r(var[keep].sum() / var.sum()), rec_err=r(((Dv - back) ** 2).sum(1).mean()),
                       qda=r(QuadraticDiscriminantAnalysis().fit(Da[:, keep], da).score(Dv[:, keep], dv)),
                       lr=r(LogisticRegression(max_iter=2000).fit(Da[:, keep], da).score(Dv[:, keep], dv)))
# one validation image rebuilt from k components, and the same image from k pixels
img = 5
recs = {}
for k in [2, 5, 10, 20, 50, 100]:
    p = PCA(n_components=k, random_state=0).fit(Da)
    recs[str(k)] = [r(v) for v in p.inverse_transform(p.transform(Dv[[img]]))[0]]
pix_recs = {}
for k in [10, 20]:
    back = Da.mean(0).copy()
    back[order[:k]] = Dv[img, order[:k]]
    pix_recs[str(k)] = [r(v) for v in back]
p2 = PCA(n_components=2, random_state=0).fit(Da)
Z2 = p2.transform(Da)
# the same rule as Lesson 6: fit on the fitting part only; what if we fitted on fitting + validation?
p_leak = PCA(n_components=10, random_state=0).fit(np.r_[Da, Dv])
leak_acc = r(QuadraticDiscriminantAnalysis().fit(p_leak.transform(Da), da).score(p_leak.transform(Dv), dv))
# test set once: QDA on 30 components chosen on validation
best_k = max([10, 20, 30, 40, 50, 100], key=lambda k: table[str(k)]["qda"])
pb = PCA(n_components=best_k, random_state=0).fit(Da)
test_qda = r(QuadraticDiscriminantAnalysis().fit(pb.transform(Da), da).score(pb.transform(Dte), dte))
pix_cum = np.cumsum(var[order]) / var.sum()
N["digits"] = dict(pix_order=[int(i) for i in order], pix_cum=[r(v, 4) for v in pix_cum], fit=len(da), val=len(dv), test=len(dte), evr=[r(v, 5) for v in evr], cum=[r(v, 5) for v in cum],
                   k90=int(np.argmax(cum >= 0.9) + 1), k95=int(np.argmax(cum >= 0.95) + 1), table=table, pixels=pix,
                   mean=[r(v) for v in Da.mean(0)], comps=[[r(v) for v in c] for c in full.components_[:6]],
                   img=img, img_label=int(dv[img]), original=[r(v) for v in Dv[img]], recs=recs, pix_recs=pix_recs,
                   pc2_x=[r(v) for v in Z2[:, 0]], pc2_y=[r(v) for v in Z2[:, 1]], pc2_label=da.tolist(),
                   pc2_evr=[r(v) for v in p2.explained_variance_ratio_], leak_qda10=leak_acc,
                   all_comps=[[r(v, 3) for v in c] for c in full.components_[:KW]], img_z=[r(v, 3) for v in full.transform(Dv[[img]])[0][:KW]],
                   extra_imgs=[[r(v) for v in Dv[i]] for i in (0, 1, 2)], extra_z=[[r(v, 3) for v in full.transform(Dv[[i]])[0][:KW]] for i in (0, 1, 2)],
                   extra_labels=[int(dv[i]) for i in (0, 1, 2)],
                   best_k=best_k, test_qda=test_qda, qda_reg_l5=0.861, qda_raw_val=table["784"]["qda"],
                   rec_err_img={str(k): r(((Dv[img] - np.array(recs[str(k)])) ** 2).sum(), 2) for k in recs},
                   pix_rec_err_img={str(k): r(((Dv[img] - np.array(pix_recs[str(k)])) ** 2).sum(), 2) for k in pix_recs})

# ---------- lumps, 30 features down to 2 ----------
d = load_breast_cancer()
y = (d.target == 0).astype(int)
Xtr, Xte, ytr, yte = train_test_split(d.data, y, test_size=0.25, random_state=0, stratify=y)
sc = StandardScaler().fit(Xtr)
Ztr = sc.transform(Xtr)
pl = PCA(n_components=2).fit(Ztr)
L2 = pl.transform(Ztr)
w1 = pl.components_[0]
top1 = np.argsort(np.abs(w1))[::-1][:4]
N["lumps"] = dict(evr=[r(v) for v in pl.explained_variance_ratio_], x=[r(v, 2) for v in L2[:, 0]], y=[r(v, 2) for v in L2[:, 1]],
                  label=ytr.tolist(), lr2_cv=r(cross_val_score(make_pipeline(StandardScaler(), PCA(n_components=2), LogisticRegression()), Xtr, ytr, cv=KF).mean()),
                  lr30_cv=r(cross_val_score(make_pipeline(StandardScaler(), LogisticRegression()), Xtr, ytr, cv=KF).mean()),
                  pc1_top=[[d.feature_names[i], r(w1[i])] for i in top1], pc1_all_positive=bool((w1 > 0).all()))

# ---------- Advanced Study: the eigenvectors of the covariance matrix are the components ----------
S = np.cov(Da.T)
vals, vecs = np.linalg.eigh(S)
idx = np.argsort(vals)[::-1]
vals, vecs = vals[idx], vecs[:, idx]
diff = [r(min(np.abs(vecs[:, k] - full.components_[k]).max(), np.abs(vecs[:, k] + full.components_[k]).max()), 8) for k in range(5)]
U, sv, Vt = np.linalg.svd(Da - Da.mean(0), full_matrices=False)
N["adv"] = dict(eig_diff=max(diff), eig_ratio_first=r(vals[0] / vals.sum(), 4), svd_var_first=r(sv[0] ** 2 / (len(Da) - 1), 4),
                pca_var_first=r(full.explained_variance_[0], 4))

json.dump(N, open(OUTJ, "w"), ensure_ascii=False, indent=1)
show = {"cloud": {k: v for k, v in N["cloud"].items() if k not in ("x", "y", "angles", "spread", "error")},
        "digits": {k: v for k, v in N["digits"].items() if k not in ("pix_order", "pix_cum", "all_comps", "img_z", "extra_imgs", "extra_z", "evr", "cum", "mean", "comps", "original", "recs", "pix_recs", "pc2_x", "pc2_y", "pc2_label")},
        "lumps": {k: v for k, v in N["lumps"].items() if k not in ("x", "y", "label")}, "adv": N["adv"]}
show["digits"]["evr5"] = N["digits"]["evr"][:5]
print(json.dumps(show, ensure_ascii=False, indent=1))
