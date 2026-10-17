"""Lesson 6 Note: every number in the note, the figures, the widgets and the lab checks.
Run (from this folder):  python build_numbers.py   -> numbers.json (and prints what the note quotes)
Rules fixed in the design card before writing the note:
  ring           the ring of Lesson 4 (default_rng(0), 150 inner = class 1, 150 outer = class 0)
  example node   hand-made, 1 feature: 80 grey on [0, 10] and 20 blue on [4, 6], default_rng(1)
  breast cancer  load_breast_cancer, 1 = malignant; the Lesson 4 split (test 143, training 426);
                 the training part split again 3:1 (stratify, random_state=0): 319 to fit, 107 to validate
  2-D trees      mean radius and mean texture, raw units; DecisionTreeClassifier(max_depth=d, random_state=0)
  CV             KFold(5, shuffle=True, random_state=0) on the 426 training lumps (as in Lessons 4 and 5)
  digits         the Lesson 4 split (pixels / 16); the 1347 training images split 3:1 again
"""
import json, os, time, warnings
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUTJ = os.path.join(HERE, "numbers.json")
r = lambda v, k=3: round(float(v), k)
KF = KFold(5, shuffle=True, random_state=0)
N = {}


def gini(labels):
    if len(labels) == 0:
        return 0.0
    p = labels.mean()
    return 2 * p * (1 - p)


def err(labels):
    if len(labels) == 0:
        return 0.0
    p = labels.mean()
    return min(p, 1 - p)


# ---------- the ring of Lesson 4 ----------
rng = np.random.default_rng(0)
ang = rng.uniform(0, 2 * np.pi, 300)
rr = np.r_[rng.normal(1.0, 0.25, 150), rng.normal(2.4, 0.3, 150)]
RX = np.c_[rr * np.cos(ang), rr * np.sin(ang)]
RY = np.r_[np.ones(150, int), np.zeros(150, int)]
box = (np.abs(RX[:, 0]) < 1.6) & (np.abs(RX[:, 1]) < 1.6)
ring_trees = {}
for d in [1, 2, 3, 4, 5]:
    t = DecisionTreeClassifier(max_depth=d, random_state=0).fit(RX, RY)
    ring_trees[str(d)] = dict(acc=r(t.score(RX, RY)), leaves=int(t.get_n_leaves()))
gx = np.linspace(-3.5, 3.5, 71)
GG = np.array([[a, b] for b in gx for a in gx])
t4 = DecisionTreeClassifier(max_depth=4, random_state=0).fit(RX, RY)
N["ring"] = dict(x=[r(v) for v in RX[:, 0]], y=[r(v) for v in RX[:, 1]], label=RY.tolist(),
                 acc_line=r(LogisticRegression().fit(RX, RY).score(RX, RY), 2), acc_box=r(((box).astype(int) == RY).mean()),
                 trees=ring_trees, grid4="".join(str(int(v)) for v in t4.predict(GG)), grid=[r(v, 2) for v in gx])

# ---------- one node, one feature: error rate cannot see progress, Gini can ----------
r1 = np.random.default_rng(1)
xg, xb = r1.uniform(0, 10, 80), r1.uniform(4, 6, 20)
NX = np.r_[xg, xb]
NY = np.r_[np.zeros(80, int), np.ones(20, int)]
cuts = np.round(np.arange(0.25, 10.0, 0.25), 2)
e_curve, g_curve = [], []
for c in cuts:
    L, Rr = NY[NX <= c], NY[NX > c]
    e_curve.append(r((len(L) * err(L) + len(Rr) * err(Rr)) / len(NY)))
    g_curve.append(r((len(L) * gini(L) + len(Rr) * gini(Rr)) / len(NY)))
best = int(np.argmin(g_curve))
c_best = float(cuts[best])
L, Rr = NY[NX <= c_best], NY[NX > c_best]
N["node"] = dict(x=[r(v, 2) for v in NX], label=NY.tolist(), cuts=cuts.tolist(), err=e_curve, gini=g_curve,
                 parent_err=r(err(NY)), parent_gini=r(gini(NY)), best_cut=c_best, best_gini=g_curve[best],
                 left=[int((L == 1).sum()), int((L == 0).sum())], right=[int((Rr == 1).sum()), int((Rr == 0).sum())],
                 err_min=min(e_curve), err_max_inside=max(e_curve[4:-4]))
assert N["node"]["err_min"] == N["node"]["parent_err"] == 0.2  # the error rate never moves

# ---------- breast cancer ----------
d = load_breast_cancer()
y = (d.target == 0).astype(int)
Xtr, Xte, ytr, yte = train_test_split(d.data, y, test_size=0.25, random_state=0, stratify=y)
Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.25, random_state=0, stratify=ytr)
assert (len(ya), len(yv), ya.sum(), yv.sum()) == (319, 107, 119, 40)
two = [0, 1]  # mean radius, mean texture
depths2 = list(range(1, 11))
gr_x = np.linspace(6, 29, 70)
gr_y = np.linspace(9, 40, 70)
G2 = np.array([[a, b] for b in gr_y for a in gr_x])
curve2, grids = {}, {}
for dp in depths2:
    t = DecisionTreeClassifier(max_depth=dp, random_state=0).fit(Xa[:, two], ya)
    curve2[str(dp)] = dict(train=r(t.score(Xa[:, two], ya)), val=r(t.score(Xv[:, two], yv)), leaves=int(t.get_n_leaves()))
    grids[str(dp)] = "".join(str(int(v)) for v in t.predict(G2))
full2 = DecisionTreeClassifier(random_state=0).fit(Xa[:, two], ya)
stump = DecisionTreeClassifier(max_depth=1, random_state=0).fit(Xa[:, two], ya)
f0, th0 = int(stump.tree_.feature[0]), float(stump.tree_.threshold[0])
left = ya[Xa[:, f0] <= th0]
right = ya[Xa[:, f0] > th0]
N["bc2"] = dict(train_x=[r(v, 2) for v in Xa[:, 0]], train_y=[r(v, 2) for v in Xa[:, 1]], train_label=ya.tolist(),
                val_x=[r(v, 2) for v in Xv[:, 0]], val_y=[r(v, 2) for v in Xv[:, 1]], val_label=yv.tolist(),
                curve=curve2, grids=grids, gx=[r(v, 2) for v in gr_x], gy=[r(v, 2) for v in gr_y],
                full_depth=int(full2.get_depth()), full_leaves=int(full2.get_n_leaves()),
                full_train=r(full2.score(Xa[:, two], ya)), full_val=r(full2.score(Xv[:, two], yv)),
                root_feature=d.feature_names[f0], root_threshold=r(th0, 2),
                root_left=[int(left.sum()), int(len(left) - left.sum())], root_right=[int(right.sum()), int(len(right) - right.sum())],
                root_gini_parent=r(gini(ya)), root_gini_left=r(gini(left)), root_gini_right=r(gini(right)),
                root_gini_split=r((len(left) * gini(left) + len(right) * gini(right)) / len(ya)))
# how much the "best depth" depends on which 107 lumps land in the validation set
picks = []
for s in range(5):
    a_, v_, b_, w_ = train_test_split(Xtr[:, two], ytr, test_size=0.25, random_state=s, stratify=ytr)
    vals = [DecisionTreeClassifier(max_depth=dp, random_state=0).fit(a_, b_).score(v_, w_) for dp in depths2]
    picks.append(dict(seed=s, best=int(depths2[int(np.argmax(vals))]), val=[r(v) for v in vals]))
cv2 = {str(dp): r(cross_val_score(DecisionTreeClassifier(max_depth=dp, random_state=0), Xtr[:, two], ytr, cv=KF).mean()) for dp in depths2}
N["bc2"]["picks"] = picks
N["bc2"]["cv"] = cv2
N["bc2"]["cv_best"] = int(max(depths2, key=lambda dp: cv2[str(dp)]))

# ---------- breast cancer, 30 features, the same CV as Lessons 4 and 5 ----------
cv30 = {str(dp): r(cross_val_score(DecisionTreeClassifier(max_depth=dp, random_state=0), Xtr, ytr, cv=KF).mean()) for dp in depths2}
best30 = int(max(depths2, key=lambda dp: cv30[str(dp)]))
t30 = DecisionTreeClassifier(max_depth=best30, random_state=0).fit(Xtr, ytr)
imp = np.argsort(t30.feature_importances_)[::-1][:5]
N["bc30"] = dict(cv=cv30, best=best30, best_cv=cv30[str(best30)], natural_depth=int(DecisionTreeClassifier(random_state=0).fit(Xtr, ytr).get_depth()),
                 lr=0.979, lda=0.960, qda=0.958, top=[[d.feature_names[i], r(t30.feature_importances_[i])] for i in imp])

# the lab: 30 features, fit on 319, validate on 107
vc30 = {}
for dp in depths2:
    t = DecisionTreeClassifier(max_depth=dp, random_state=0).fit(Xa, ya)
    vc30[str(dp)] = dict(train=r(t.score(Xa, ya)), val=r(t.score(Xv, yv)))
vbest30 = int(max(depths2, key=lambda dp: vc30[str(dp)]["val"]))
tv = DecisionTreeClassifier(max_depth=vbest30, random_state=0).fit(Xa, ya)
t2 = DecisionTreeClassifier(max_depth=2, random_state=0).fit(Xa, ya)
impv = np.argsort(tv.feature_importances_)[::-1][:5]
N["bc30"].update(val_curve=vc30, val_best=vbest30, val_best_acc=vc30[str(vbest30)]["val"], test_acc=r(tv.score(Xte, yte)),
                 root2=[d.feature_names[int(t2.tree_.feature[0])], r(t2.tree_.threshold[0], 4)],
                 top_val=[[d.feature_names[i], r(tv.feature_importances_[i])] for i in impv])

# ---------- digits: choose the depth on a validation set ----------
_mn = np.load(os.path.join(HERE, "../../../../data/mnist10k.npz"))  # the course MNIST subset: 10,000 images of 28 x 28
MX, MY = _mn["X"] / 255.0, _mn["y"].astype(int)  # pixels scaled to 0 to 1; digits 0 to 9
Dtr, Dte, dtr, dte = train_test_split(MX, MY, test_size=0.25, random_state=0, stratify=MY)
Da, Dv, da, dv = train_test_split(Dtr, dtr, test_size=0.25, random_state=0, stratify=dtr)
ddepths = list(range(1, 21))
dcurve = {}
for dp in ddepths:
    t = DecisionTreeClassifier(max_depth=dp, random_state=0).fit(Da, da)
    dcurve[str(dp)] = dict(train=r(t.score(Da, da)), val=r(t.score(Dv, dv)))
dbest = int(max(ddepths, key=lambda dp: dcurve[str(dp)]["val"]))
assert dbest not in (ddepths[0], ddepths[-1])
tbest = DecisionTreeClassifier(max_depth=dbest, random_state=0).fit(Da, da)
t0 = time.perf_counter()
for dp in ddepths:
    DecisionTreeClassifier(max_depth=dp, random_state=0).fit(Da, da)
t_val = time.perf_counter() - t0
t0 = time.perf_counter()
for dp in ddepths:
    cross_val_score(DecisionTreeClassifier(max_depth=dp, random_state=0), Dtr, dtr, cv=KF)
t_cv = time.perf_counter() - t0
dcv = {str(dp): r(cross_val_score(DecisionTreeClassifier(max_depth=dp, random_state=0), Dtr, dtr, cv=KF).mean()) for dp in ddepths}
N["digits_cv"] = dict(curve=dcv, best=int(max(ddepths, key=lambda dp: dcv[str(dp)])))
N["digits"] = dict(fit=len(da), val=len(dv), test=len(dte), curve=dcurve, best=dbest, best_val=dcurve[str(dbest)]["val"],
                   natural_depth=int(DecisionTreeClassifier(random_state=0).fit(Da, da).get_depth()),
                   test_acc=r(tbest.score(Dte, dte)), fits_val=len(ddepths), fits_cv=5 * len(ddepths),
                   time_ratio=r(t_cv / t_val, 1))

# ---------- Advanced Study 2: how much does the tree change with another sample of lumps? ----------
roots = []
rb = np.random.default_rng(0)
for k in range(10):
    idx = rb.integers(0, len(ytr), len(ytr))  # draw 426 lumps with replacement
    t = DecisionTreeClassifier(max_depth=3, random_state=0).fit(Xtr[idx], ytr[idx])
    roots.append([d.feature_names[int(t.tree_.feature[0])], r(t.tree_.threshold[0], 3)])
N["as2"] = dict(roots=roots, distinct=len(set(x[0] for x in roots)))

json.dump(N, open(OUTJ, "w"), ensure_ascii=False, indent=1)
show = {
    "ring": {k: v for k, v in N["ring"].items() if k not in ("x", "y", "label", "grid4", "grid")},
    "node": {k: v for k, v in N["node"].items() if k not in ("x", "label", "cuts", "err", "gini")},
    "bc2": {k: v for k, v in N["bc2"].items() if k not in ("train_x", "train_y", "train_label", "val_x", "val_y", "val_label", "grids", "gx", "gy")},
    "bc30": N["bc30"], "digits": N["digits"], "as2": N["as2"],
}
print(json.dumps(show, ensure_ascii=False, indent=1))
