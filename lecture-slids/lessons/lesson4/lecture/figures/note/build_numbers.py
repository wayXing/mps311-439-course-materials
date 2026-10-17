"""Lesson 4 Note: every number in the note, the figures, the widgets and the lab checks.
Run (from this folder):  python build_numbers.py   -> numbers.json (and prints what the note quotes)
Rules fixed in the design card before writing the note:
  data         load_breast_cancer, 1 = malignant (the class we look for)
  split        train_test_split(test_size=0.25, random_state=0, stratify=y)  -> 426 / 143
  model        LogisticRegression() with sklearn defaults (C = 1)
  one feature  mean radius, raw units;  two features  mean radius + mean texture, standardised
  30 features  StandardScaler fitted once on the training set (no Pipeline); per-fold scaling is checked
  CV           KFold(5, shuffle=True, random_state=0) inside the training set; cross_val_predict
  opening      test tumour 47 (radius 12.8, malignant); closing also uses test tumour 73
  digits       the course MNIST subset (lessons/data/mnist10k.npz, pixels / 255), same split rule; LogisticRegression(max_iter=1000, C) with C chosen by the same 5-fold CV
"""
import json, os, warnings
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold, cross_val_predict, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUTJ = os.path.join(HERE, "numbers.json")
r = lambda v, k=2: round(float(v), k)

d = load_breast_cancer()
y = (d.target == 0).astype(int)
FN = list(d.feature_names)
Xtr, Xte, ytr, yte = train_test_split(d.data, y, test_size=0.25, random_state=0, stratify=y)
assert (len(ytr), ytr.sum(), len(yte), yte.sum()) == (426, 159, 143, 53)
KF = KFold(5, shuffle=True, random_state=0)
RAD, TEX = FN.index("mean radius"), FN.index("mean texture")
N = {"data": dict(n=len(y), malignant=int(y.sum()), n_features=len(FN), train=len(ytr), train_mal=int(ytr.sum()),
                  test=len(yte), test_mal=int(yte.sum()), share_mal=r(y.mean(), 3))}


def counts(true, pred):
    tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
    return dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), acc=r((tp + tn) / len(true), 3),
                precision=r(tp / (tp + fp), 3) if tp + fp else 0.0, recall=r(tp / (tp + fn), 3))


# ---------- step 1: a straight line on 0/1 labels ----------
rad = Xtr[:, RAD]
lin = LinearRegression().fit(rad[:, None], ytr)
plin = lin.predict(rad[:, None])
big = 40.0
rad_a, y_a = np.r_[rad, np.full(20, big)], np.r_[ytr, np.ones(20, int)]
lin_a = LinearRegression().fit(rad_a[:, None], y_a)
log1 = LogisticRegression().fit(rad[:, None], ytr)
log1_a = LogisticRegression().fit(rad_a[:, None], y_a)
N["line"] = dict(pred_min=r(plin.min()), pred_max=r(plin.max()), share_outside=r(((plin < 0) | (plin > 1)).mean(), 2),
                 w=r(lin.coef_[0], 4), b=r(lin.intercept_, 3),
                 boundary=r((0.5 - lin.intercept_) / lin.coef_[0]), boundary_added=r((0.5 - lin_a.intercept_) / lin_a.coef_[0]),
                 added_radius=big, added_n=20, radius_min=r(rad.min()), radius_max=r(rad.max()))
N["one"] = dict(r=[r(v, 2) for v in rad], y=ytr.tolist())   # training radius and labels, for the widgets
N["logit1"] = dict(w=r(log1.coef_[0, 0], 3), b=r(log1.intercept_[0], 3),
                   boundary=r(-log1.intercept_[0] / log1.coef_[0, 0]), boundary_added=r(-log1_a.intercept_[0] / log1_a.coef_[0, 0]))

# ---------- opening tumour ----------
sc = StandardScaler().fit(Xtr)
Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
m30 = LogisticRegression().fit(Ztr, ytr)
p30_test = m30.predict_proba(Zte)[:, 1]
T = 47
N["tumour"] = dict(index=T, radius=r(Xte[T, RAD], 1), label=int(yte[T]),
                   p1=r(log1.predict_proba(Xte[[T]][:, [RAD]])[0, 1], 3), p30=r(p30_test[T], 3))
assert N["tumour"]["label"] == 1
T2 = 73  # a second, smaller malignant test lump: radius alone misses it, 30 features find it
N["tumour2"] = dict(index=T2, radius=r(Xte[T2, RAD], 1), label=int(yte[T2]),
                    p1=r(log1.predict_proba(Xte[[T2]][:, [RAD]])[0, 1], 3), p30=r(p30_test[T2], 3))
assert N["tumour2"]["label"] == 1 and N["tumour2"]["p1"] < 0.1 <= N["tumour2"]["p30"]
assert N["tumour"]["p1"] >= 0.1 and N["tumour"]["p30"] < 0.3

# ---------- step 2: two standardised features, for the boundary widget ----------
sc2 = StandardScaler().fit(Xtr[:, [RAD, TEX]])
Z2 = sc2.transform(Xtr[:, [RAD, TEX]])
m2 = LogisticRegression().fit(Z2, ytr)
N["two"] = dict(w1=r(m2.coef_[0, 0]), w2=r(m2.coef_[0, 1]), b=r(m2.intercept_[0]),
                z1=[r(v, 3) for v in Z2[:, 0]], z2=[r(v, 3) for v in Z2[:, 1]], label=ytr.tolist())

# ---------- step 3: squared error vs cross-entropy for a true 1 ----------
N["loss_table"] = [dict(p=p, sq=r((1 - p) ** 2, 3), ce=r(-np.log(p), 2)) for p in [0.9, 0.5, 0.1, 0.01, 0.0001]]

# ---------- step 4: cross-entropy along one weight (standardised radius), and gradient descent ----------
z1 = (rad - rad.mean()) / rad.std()
mz = LogisticRegression(penalty=None).fit(z1[:, None], ytr)
b_opt = float(mz.intercept_[0])


def ce(w, b=b_opt):
    p = 1 / (1 + np.exp(-(w * z1 + b)))
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(ytr * np.log(p) + (1 - ytr) * np.log(1 - p)))


wgrid = np.linspace(-1, 7, 161)
w, path = 0.0, []
for step in range(10):
    p = 1 / (1 + np.exp(-(w * z1 + b_opt)))
    path.append(dict(w=r(w, 3), loss=r(ce(w), 4)))
    w = w - 5.0 * float(np.mean((p - ytr) * z1))
N["gd"] = dict(w_opt=r(mz.coef_[0, 0], 3), b=r(b_opt, 3), lr=5.0, grid=[r(v, 3) for v in wgrid],
               loss=[r(ce(v), 4) for v in wgrid], path=path)

# ---------- step 5: baseline and confusion matrix, cross-validated predictions on the training set ----------
N["baseline"] = dict(acc=r(1 - ytr.mean(), 3), test_acc=r(1 - yte.mean(), 3))
pred1_cv = cross_val_predict(LogisticRegression(), Xtr[:, [RAD]], ytr, cv=KF)
N["cv1"] = counts(ytr, pred1_cv)
prob30_cv = cross_val_predict(LogisticRegression(), Ztr, ytr, cv=KF, method="predict_proba")[:, 1]
N["cv30"] = counts(ytr, (prob30_cv >= 0.5).astype(int))
# unstandardised 30 features: does the default solver complain?
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    LogisticRegression().fit(Xtr, ytr)
N["unscaled_warning"] = any("converge" in str(c.message).lower() for c in caught)

# ---------- step 6: thresholds on cross-validated probabilities; the test set once ----------
N["thresholds"] = {str(t): counts(ytr, (prob30_cv >= t).astype(int)) for t in [0.5, 0.3, 0.1]}
prob_fold = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression()), Xtr, ytr, cv=KF, method="predict_proba")[:, 1]
for t in [0.5, 0.3, 0.1]:  # per-fold scaling gives the same counts as scaling once
    assert counts(ytr, (prob_fold >= t).astype(int)) == N["thresholds"][str(t)], t
N["chosen"] = 0.1
N["test"] = {str(t): counts(yte, (p30_test >= t).astype(int)) for t in [0.5, 0.3, 0.1]}
sweep = np.round(np.arange(0.05, 0.96, 0.05), 2)
N["sweep"] = [dict(t=float(t), fn=int(((prob30_cv < t) & (ytr == 1)).sum()), fp=int(((prob30_cv >= t) & (ytr == 0)).sum())) for t in sweep]
fpr, tpr, thr = roc_curve(ytr, prob30_cv)
N["roc"] = dict(fpr=[r(v, 4) for v in fpr], tpr=[r(v, 4) for v in tpr], auc=r(roc_auc_score(ytr, prob30_cv), 3),
                points={str(t): dict(fpr=r(N["thresholds"][str(t)]["fp"] / (len(ytr) - ytr.sum()), 4),
                                     tpr=r(N["thresholds"][str(t)]["tp"] / ytr.sum(), 4)) for t in [0.5, 0.3, 0.1]})
N["hist"] = dict(mal=[r(v, 4) for v in prob30_cv[ytr == 1]], ben=[r(v, 4) for v in prob30_cv[ytr == 0]])

# ---------- extension: rare disease arithmetic ----------
N["rare"] = dict(prevalence=0.01, n=10000, sick=100, acc_all_healthy=0.99)

# ---------- extension: handwritten digits (multiclass), the course MNIST subset ----------
_mn = np.load(os.path.join(HERE, "../../../../data/mnist10k.npz"))  # 10,000 MNIST images of 28 x 28
MX, MY = _mn["X"] / 255.0, _mn["y"].astype(int)  # pixels scaled to 0 to 1; digits 0 to 9
Dtr, Dte, dtr, dte = train_test_split(MX, MY, test_size=0.25, random_state=0, stratify=MY)
C_grid = [1.0, 0.3, 0.1, 0.03, 0.01]  # how strongly to regularise (Lesson 3): smaller C is stronger
C_cv = {str(c): r(cross_val_score(LogisticRegression(max_iter=1000, C=c), Dtr, dtr, cv=KF).mean(), 4) for c in C_grid}
C_best = max(C_grid, key=lambda c: C_cv[str(c)])
assert C_best not in (C_grid[0], C_grid[-1])
md = LogisticRegression(max_iter=1000, C=C_best).fit(Dtr, dtr)
md1 = LogisticRegression(max_iter=1000).fit(Dtr, dtr)  # the default C = 1, for comparison
dp = md.predict(Dte)
cm = confusion_matrix(dte, dp)
off = cm.copy()
np.fill_diagonal(off, 0)
a, b = np.unravel_index(off.argmax(), off.shape)
counts_tr = np.bincount(dtr)
N["digits"] = dict(n=len(MY), train=len(dtr), test=len(dte), acc=r(accuracy_score(dte, dp), 3), acc_C1=r(md1.score(Dte, dte), 3),
                   C_cv=C_cv, C_best=C_best,
                   majority_acc=r((dte == np.argmax(counts_tr)).mean(), 3), confused_true=int(a), confused_pred=int(b),
                   confused_n=int(off[a, b]), cm=cm.tolist(), coef=[[r(v, 3) for v in row] for row in md.coef_],
                   coef_C1=[[r(v, 3) for v in row] for row in md1.coef_])

# ---------- extension: a ring that a straight line cannot separate ----------
rng = np.random.default_rng(0)
ang = rng.uniform(0, 2 * np.pi, 300)
rr = np.r_[rng.normal(1.0, 0.25, 150), rng.normal(2.4, 0.3, 150)]
ring_X = np.c_[rr * np.cos(ang), rr * np.sin(ang)]
ring_y = np.r_[np.ones(150, int), np.zeros(150, int)]
acc_line = accuracy_score(ring_y, LogisticRegression().fit(ring_X, ring_y).predict(ring_X))
dist = np.sqrt((ring_X ** 2).sum(1))[:, None]
acc_dist = accuracy_score(ring_y, LogisticRegression().fit(dist, ring_y).predict(dist))
N["ring"] = dict(x=[r(v, 3) for v in ring_X[:, 0]], y=[r(v, 3) for v in ring_X[:, 1]], label=ring_y.tolist(),
                 acc_line=r(acc_line, 2), acc_dist=r(acc_dist, 2))

# ---------- Advanced Study 2: hand-written gradient descent vs sklearn (same default penalty) ----------
n_tr = len(ytr)
w_gd, b_gd = np.zeros(Ztr.shape[1]), 0.0
for step in range(5000):
    p = 1 / (1 + np.exp(-(Ztr @ w_gd + b_gd)))
    w_gd = w_gd - 0.5 * (Ztr.T @ (p - ytr) / n_tr + w_gd / n_tr)
    b_gd = b_gd - 0.5 * np.mean(p - ytr)
nopen = LogisticRegression(C=1e6, max_iter=10000).fit(Ztr, ytr)
N["as2"] = dict(diff=r(np.abs(w_gd - m30.coef_[0]).max(), 3), max_w=r(np.abs(m30.coef_).max()),
                max_w_nopenalty=round(float(np.abs(nopen.coef_).max())), train_acc_nopenalty=r(nopen.score(Ztr, ytr), 3))

json.dump(N, open(OUTJ, "w"), ensure_ascii=False, indent=1)
show = {k: v for k, v in N.items() if k not in ("gd", "roc", "hist", "digits", "ring", "sweep", "two", "one")}
show["gd"] = {k: v for k, v in N["gd"].items() if k not in ("grid", "loss")}
show["roc"] = {k: v for k, v in N["roc"].items() if k in ("auc", "points")}
show["digits"] = {k: v for k, v in N["digits"].items() if k not in ("cm", "coef")}
show["ring"] = {k: v for k, v in N["ring"].items() if k.startswith("acc")}
print(json.dumps(show, ensure_ascii=False, indent=1))
