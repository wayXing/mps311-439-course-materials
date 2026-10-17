"""Lesson 5 Note: every number in the note, the figures, the widgets and the lab checks.
Run (from this folder):  python build_numbers.py   -> numbers.json (and prints what the note quotes)
Rules fixed in the design card before writing the note:
  breast cancer  load_breast_cancer, 1 = malignant; train_test_split(test_size=0.25, random_state=0, stratify=y)
                 (the same split as Lesson 4); 30 features standardised once on the training set
  CV             KFold(5, shuffle=True, random_state=0) inside the training set (as in Lesson 4)
  one feature    mean radius; the opening lump of Lesson 4 (test lump 47, radius 12.8)
  2-D data       hand-made: class 1 (120 points) N((0,0), [[.35,.15],[.15,.35]]), class 0 (240 points)
                 N((1,.6), [[2.2,-.6],[-.6,1.6]]), drawn with default_rng(0), class 1 first
  digits         the course MNIST subset (lessons/data/mnist10k.npz, pixels / 255), same split rule; QDA reg_param chosen by the same CV
"""
import json, os, warnings
import numpy as np
from scipy.stats import norm
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
OUTJ = os.path.join(HERE, "numbers.json")
r = lambda v, k=3: round(float(v), k)
KF = KFold(5, shuffle=True, random_state=0)
N = {}


def quiet_cv(model, X, y):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        s = cross_val_score(model, X, y, cv=KF).mean()
    return float(s), any("full rank" in str(c.message) for c in caught)


# ---------- breast cancer, the same split as Lesson 4 ----------
d = load_breast_cancer()
y = (d.target == 0).astype(int)
Xtr, Xte, ytr, yte = train_test_split(d.data, y, test_size=0.25, random_state=0, stratify=y)
assert (len(ytr), ytr.sum(), len(yte), yte.sum()) == (426, 159, 143, 53)
rad = Xtr[:, 0]

# ---------- one feature: a bell curve for each class ----------
one = {}
for c in (1, 0):
    one[str(c)] = dict(mean=r(rad[ytr == c].mean(), 2), sd=r(rad[ytr == c].std(ddof=1), 2), n=int((ytr == c).sum()),
                       share=r((ytr == c).mean(), 3))
x0 = float(Xte[47, 0])  # the opening lump of Lesson 4 (radius 12.77, shown as 12.8)
h1 = norm.pdf(x0, rad[ytr == 1].mean(), rad[ytr == 1].std(ddof=1))
h0 = norm.pdf(x0, rad[ytr == 0].mean(), rad[ytr == 0].std(ddof=1))
s1, s0 = h1 * (ytr == 1).mean(), h0 * (ytr == 0).mean()
q1 = QuadraticDiscriminantAnalysis().fit(rad[:, None], ytr)
l1 = LinearDiscriminantAnalysis().fit(rad[:, None], ytr)
g = np.linspace(0, 40, 40001)
pq = q1.predict_proba(g[:, None])[:, 1]
cross = g[np.where(np.diff((pq >= 0.5).astype(int)))[0]]
pl = l1.predict_proba(g[:, None])[:, 1]
cross_l = g[np.where(np.diff((pl >= 0.5).astype(int)))[0]]
lr1 = LogisticRegression().fit(rad[:, None], ytr)
N["one"] = dict(cls=one, x=r(x0, 2), h1=r(h1, 5), h0=r(h0, 5), s1=r(s1, 4), s0=r(s0, 4), post=r(s1 / (s1 + s0)),
                qda_post=r(q1.predict_proba([[x0]])[0, 1]), lda_post=r(l1.predict_proba([[x0]])[0, 1]),
                lr_post=r(lr1.predict_proba([[x0]])[0, 1]), qda_cuts=[r(v, 2) for v in cross], lda_cut=r(cross_l[0], 2),
                radius_min=r(rad.min(), 2), r=[r(v, 2) for v in rad], y=ytr.tolist())
assert abs(N["one"]["post"] - N["one"]["qda_post"]) < 0.002

# ---------- hand-made 2-D data ----------
rng = np.random.default_rng(0)
C1, C0 = np.array([[0.35, 0.15], [0.15, 0.35]]), np.array([[2.2, -0.6], [-0.6, 1.6]])
X1 = rng.multivariate_normal([0, 0], C1, 120)
X0 = rng.multivariate_normal([1.0, 0.6], C0, 240)
X2, y2 = np.r_[X1, X0], np.r_[np.ones(120, int), np.zeros(240, int)]
m_lr, m_lda, m_qda = LogisticRegression().fit(X2, y2), LinearDiscriminantAnalysis().fit(X2, y2), QuadraticDiscriminantAnalysis().fit(X2, y2)
mu = {str(c): X2[y2 == c].mean(0) for c in (1, 0)}
cov = {str(c): np.cov(X2[y2 == c].T) for c in (1, 0)}
pooled = ((119 * cov["1"]) + (239 * cov["0"])) / 358


def gauss_post(P, covs, shrink=0.0):
    """Probability of class 1 from two Gaussians and the class shares (what the widgets compute in JS)."""
    out = []
    for c in ("1", "0"):
        S = (1 - shrink) * covs[c] + shrink * np.eye(2)
        diff = P - mu[c]
        quad = np.einsum("ij,jk,ik->i", diff, np.linalg.inv(S), diff)
        dens = np.exp(-0.5 * quad) / (2 * np.pi * np.sqrt(np.linalg.det(S)))
        out.append(dens * (120 if c == "1" else 240) / 360)
    return out[0] / (out[0] + out[1])


# the JS formulas must give the same answers as sklearn
assert np.abs(gauss_post(X2, cov) - m_qda.predict_proba(X2)[:, 1]).max() < 1e-6
assert np.abs(gauss_post(X2, {"1": pooled, "0": pooled}) - m_lda.predict_proba(X2)[:, 1]).max() < 1e-6
for s in (0.3, 1.0):
    ref = QuadraticDiscriminantAnalysis(reg_param=s).fit(X2, y2).predict_proba(X2)[:, 1]
    assert np.abs(gauss_post(X2, cov, s) - ref).max() < 1e-6, s
acc_train = lambda m: r(m.score(X2, y2))
shrink_acc = {str(s): r(((gauss_post(X2, cov, s) >= 0.5) == y2).mean()) for s in (0.0, 0.3, 0.6, 1.0)}
N["two"] = dict(x1=[r(v, 3) for v in X2[:, 0]], x2=[r(v, 3) for v in X2[:, 1]], label=y2.tolist(),
                n1=120, n0=240, baseline=r(240 / 360),
                mu={c: [r(v, 4) for v in mu[c]] for c in mu}, cov={c: [[r(v, 4) for v in row] for row in cov[c]] for c in cov},
                pooled=[[r(v, 4) for v in row] for row in pooled],
                lr=dict(w1=r(m_lr.coef_[0, 0], 4), w2=r(m_lr.coef_[0, 1], 4), b=r(m_lr.intercept_[0], 4)),
                acc=dict(lr=acc_train(m_lr), lda=acc_train(m_lda), qda=acc_train(m_qda)),
                cv=dict(lr=r(quiet_cv(LogisticRegression(), X2, y2)[0]), lda=r(quiet_cv(LinearDiscriminantAnalysis(), X2, y2)[0]),
                        qda=r(quiet_cv(QuadraticDiscriminantAnalysis(), X2, y2)[0])),
                shrink_acc=shrink_acc)
pt = np.array([[1.5, -1.5]])  # a point the widget starts from
N["two"]["start"] = dict(x=1.5, y=-1.5, post=r(gauss_post(pt, cov)[0]))

# ---------- breast cancer, 30 features: LR, LDA, QDA under the same cross-validation ----------
sc = StandardScaler().fit(Xtr)
Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
bc = {}
for name, model in [("lr", LogisticRegression()), ("lda", LinearDiscriminantAnalysis()), ("qda", QuadraticDiscriminantAnalysis())]:
    acc, warned = quiet_cv(model, Ztr, ytr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = cross_val_predict(model, Ztr, ytr, cv=KF)
    cm = confusion_matrix(ytr, pred, labels=[1, 0])
    bc[name] = dict(acc=r(acc), warned=warned, missed=int(cm[0, 1]), false_alarms=int(cm[1, 0]))
assert abs(quiet_cv(LinearDiscriminantAnalysis(), Xtr, ytr)[0] - bc["lda"]["acc"]) < 0.001  # scaling does not change LDA
dd = 30
bc["counts"] = dict(lr=dd + 1, lda=2 * dd + dd * (dd + 1) // 2 + 1, qda=2 * dd + 2 * (dd * (dd + 1) // 2) + 1, cov_one=dd * (dd + 1) // 2)
bc["baseline"] = r(1 - ytr.mean())
bc_grid = [0.0, 0.001, 0.01, 0.1, 0.3, 0.5, 0.9]
bc["qda_reg"] = {str(s): r(quiet_cv(QuadraticDiscriminantAnalysis(reg_param=s), Ztr, ytr)[0], 4) for s in bc_grid}
N["bc"] = bc

# ---------- MNIST subset: QDA in 784 dimensions ----------
_mn = np.load(os.path.join(HERE, "../../../../data/mnist10k.npz"))  # the course MNIST subset: 10,000 images of 28 x 28
MX, MY = _mn["X"] / 255.0, _mn["y"].astype(int)  # pixels scaled to 0 to 1; digits 0 to 9
Dtr, Dte, dtr, dte = train_test_split(MX, MY, test_size=0.25, random_state=0, stratify=MY)
zero_px = [int((Dtr[dtr == k].std(0) == 0).sum()) for k in range(10)]
zero_all = int((Dtr.std(0) == 0).sum())  # pixels that are 0 in every training image of every class
per_class = [int((dtr == k).sum()) for k in range(10)]
qda_default, warned = quiet_cv(QuadraticDiscriminantAnalysis(), Dtr, dtr)
grid = [0.0, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]
curve = {str(s): r(quiet_cv(QuadraticDiscriminantAnalysis(reg_param=s), Dtr, dtr)[0], 4) for s in grid}
best = max(grid, key=lambda s: curve[str(s)])
assert best not in (grid[0], grid[-1])
narrow = [0.01, 0.1, 0.3, 0.5]
narrow_curve = {str(s): r(quiet_cv(QuadraticDiscriminantAnalysis(reg_param=s), Dtr, dtr)[0], 4) for s in narrow}
narrow_best = max(narrow, key=lambda s: narrow_curve[str(s)])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    q_best = QuadraticDiscriminantAnalysis(reg_param=best).fit(Dtr, dtr)
    lda_d = LinearDiscriminantAnalysis().fit(Dtr, dtr)
    lr_d = LogisticRegression(max_iter=1000, C=0.1).fit(Dtr, dtr)   # C = 0.1, as chosen in Lesson 4
N["digits"] = dict(n=len(MY), train=len(dtr), test=len(dte), per_class_min=min(per_class), per_class_max=max(per_class),
                   zero_px=zero_px, zero_all=zero_all, zero_px_min=min(zero_px), zero_px_max=max(zero_px),
                   cov_one=784 * 785 // 2, qda_default_cv=r(qda_default), qda_warned=warned,
                   grid=grid, curve=curve, best=best, best_cv=curve[str(best)],
                   narrow=narrow, narrow_curve=narrow_curve, narrow_best=narrow_best,
                   lda_cv=r(quiet_cv(LinearDiscriminantAnalysis(), Dtr, dtr)[0]),
                   lr_cv=r(quiet_cv(LogisticRegression(max_iter=1000, C=0.1), Dtr, dtr)[0]),
                   test_qda=r(q_best.score(Dte, dte)), test_lda=r(lda_d.score(Dte, dte)), test_lr=r(lr_d.score(Dte, dte)),
                   means=[[r(v, 3) for v in Dtr[dtr == k].mean(0)] for k in range(10)],
                   sd0=[r(v, 3) for v in Dtr[dtr == 0].std(0)])

# ---------- Advanced Study 2: a Gaussian classifier in numpy, against sklearn's QDA (breast cancer, shrink 0.1) ----------
shrink = 0.1
scores = []
for c in (0, 1):
    Xc = Ztr[ytr == c]
    S = (1 - shrink) * np.cov(Xc.T) + shrink * np.eye(30)
    diff = Zte - Xc.mean(0)
    quad = np.einsum("ij,jk,ik->i", diff, np.linalg.inv(S), diff)
    scores.append(-0.5 * quad - 0.5 * np.linalg.slogdet(S)[1] + np.log((ytr == c).mean()))
mine = np.argmax(np.c_[scores[0], scores[1]], axis=1)
ref = QuadraticDiscriminantAnalysis(reg_param=shrink).fit(Ztr, ytr).predict(Zte)
N["as2"] = dict(shrink=shrink, differ=int((mine != ref).sum()), test_acc=r((mine == yte).mean()), n_test=len(yte))

json.dump(N, open(OUTJ, "w"), ensure_ascii=False, indent=1)
show = {k: v for k, v in N.items() if k not in ("two", "digits", "one")}
show["one"] = {k: v for k, v in N["one"].items() if k not in ("r", "y")}
show["two"] = {k: v for k, v in N["two"].items() if k not in ("x1", "x2", "label")}
show["digits"] = {k: v for k, v in N["digits"].items() if k not in ("means", "sd0")}
print(json.dumps(show, ensure_ascii=False, indent=1))
