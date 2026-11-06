"""Lesson 6 Note: numbers for the four-cluster data and the regression tree (widgets I1x, I4).
Run (from this folder): python build_numbers_v2.py -> numbers_v2.json
  XOR            hand-made, four clusters of 40, sd 0.6, centres (1,1),(4,4) class 0 and (1,4),(4,1) class 1, default_rng(0)
  XOR (machine)  the same but cluster sizes 60, 30, 30, 40 (a perfectly balanced XOR gives greedy no first cut)
  breast cancer  as in V1 (build_numbers.py): Lesson 4 split, 319 to fit / 107 to validate, KFold(5, shuffle, 0)
  Auto MPG       lesson2/lab/auto_mpg.csv, the Lesson 2 split (test_size=0.2, random_state=42): 313 training cars
"""
import json, os, math, warnings
import numpy as np, pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import BaggingClassifier, GradientBoostingRegressor
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
r = lambda v, k=3: round(float(v), k)
KF = KFold(5, shuffle=True, random_state=0)
N = {}

def xor(sizes, seed=0, sd=0.6):
    rng = np.random.default_rng(seed)
    c = [[1, 1], [4, 4], [1, 4], [4, 1]]; lab = [0, 0, 1, 1]
    X = np.vstack([rng.normal(ci, sd, (m, 2)) for ci, m in zip(c, sizes)])
    y = np.concatenate([[l] * m for l, m in zip(lab, sizes)])
    return X, y

X, y = xor((40, 40, 40, 40))
def vote(mask):  # accuracy if each region predicts its majority
    ok = 0
    for m in mask:
        if m.sum(): ok += max(y[m].sum(), m.sum() - y[m].sum())
    return ok / len(y)
cuts = np.round(np.arange(0.0, 5.01, 0.1), 2)
one = [vote([X[:, 0] <= a, X[:, 0] > a]) for a in cuts]
two = vote([(X[:, 0] <= 2.5) & (X[:, 1] <= 2.5), (X[:, 0] <= 2.5) & (X[:, 1] > 2.5), (X[:, 0] > 2.5) & (X[:, 1] <= 2.5), (X[:, 0] > 2.5) & (X[:, 1] > 2.5)])
sym = {d: r(DecisionTreeClassifier(max_depth=d, random_state=0).fit(X, y).score(X, y)) for d in (1, 2, 3)}
Xm, ym = xor((60, 30, 30, 40))
asym = {d: r(DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xm, ym).score(Xm, ym)) for d in (1, 2, 3)}
N["xor"] = dict(x=[r(v, 2) for v in X[:, 0]], y=[r(v, 2) for v in X[:, 1]], label=y.tolist(),
                logit=r(LogisticRegression().fit(X, y).score(X, y)), one_cut_best=r(max(one)), one_cut_at_25=r(vote([X[:, 0] <= 2.5, X[:, 0] > 2.5])),
                two_cuts=r(two), greedy_sym=sym, greedy_asym=asym)

# Gini = 2 x squared error of predicting the leaf share p; entropy = cross-entropy of predicting p
d = load_breast_cancer(); yb = (d.target == 0).astype(int)
Xtr, Xte, ytr, yte = train_test_split(d.data, yb, test_size=0.25, random_state=0, stratify=yb)
p = 35 / 228
N["leaf"] = dict(p=r(p), mse=r(p * (1 - p)), gini=r(2 * p * (1 - p)), entropy=r(-(p * math.log(p) + (1 - p) * math.log(1 - p))),
                 parent_p=r(119 / 319))
# how many trees: 319 lumps, 2 features -> distinct cut points
Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.25, random_state=0, stratify=ytr)
ncut = int(sum(len(np.unique(Xa[:, j])) - 1 for j in (0, 1)))
N["count"] = dict(cuts=ncut, depth2=f"{float(ncut) ** 3:.1e}", depth3=f"{float(ncut) ** 7:.1e}")

# regression tree: horsepower -> mpg on the 313 training cars of Lesson 2
cars = pd.read_csv(os.path.join(HERE, "../../../../lesson2/lab/auto_mpg.csv"))
Xc, yc = cars[["horsepower"]].values, cars["mpg"].values
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.2, random_state=42)
assert len(yc_tr) == 313
rm = lambda m, A, b: r(-cross_val_score(m, A, b, cv=KF, scoring="neg_root_mean_squared_error").mean(), 2)
grid = np.linspace(40, 235, 196)
reg = {}
for dp in list(range(1, 11)) + [None]:
    t = DecisionTreeRegressor(max_depth=dp, random_state=0).fit(Xc_tr, yc_tr)
    reg[str(dp)] = dict(train=r(np.sqrt(np.mean((t.predict(Xc_tr) - yc_tr) ** 2)), 2), cv=rm(DecisionTreeRegressor(max_depth=dp, random_state=0), Xc_tr, yc_tr),
                        leaves=int(t.get_n_leaves()), curve=[r(v, 2) for v in t.predict(grid[:, None])])
best = min([k for k in reg if k != "None"], key=lambda k: reg[k]["cv"])
N["reg"] = dict(x=[int(v) for v in Xc_tr[:, 0]], y=[r(v, 1) for v in yc_tr], grid=[r(v, 1) for v in grid], depths=reg, best=int(best),
                linear_cv=rm(LinearRegression(), Xc_tr, yc_tr), natural_depth=int(DecisionTreeRegressor(random_state=0).fit(Xc_tr, yc_tr).get_depth()))

# Advanced: bagging on 30-feature lumps; boosting by hand on all car features
single = r(cross_val_score(DecisionTreeClassifier(random_state=0), Xtr, ytr, cv=KF).mean())
bag = r(cross_val_score(BaggingClassifier(DecisionTreeClassifier(random_state=0), n_estimators=25, random_state=0), Xtr, ytr, cv=KF).mean())
A = cars.drop(columns="mpg").values; A_tr, A_te, b_tr, b_te = train_test_split(A, cars["mpg"].values, test_size=0.2, random_state=42)
pred = np.zeros(len(b_tr)); rounds = []
for k in range(3):
    t = DecisionTreeRegressor(max_depth=2, random_state=0).fit(A_tr, b_tr - pred)
    pred = pred + t.predict(A_tr); rounds.append(r(np.sqrt(np.mean((b_tr - pred) ** 2)), 2))
N["adv"] = dict(single_full=single, bagging25=bag, boost_rounds=rounds,
                cars_cv=dict(linear=rm(LinearRegression(), A_tr, b_tr), tree5=rm(DecisionTreeRegressor(max_depth=5, random_state=0), A_tr, b_tr),
                             gbr=rm(GradientBoostingRegressor(random_state=0), A_tr, b_tr)))
json.dump(N, open(os.path.join(HERE, "numbers_v2.json"), "w"), ensure_ascii=False)
show = {k: v for k, v in N.items() if k not in ("xor", "reg")}
print(json.dumps(show, ensure_ascii=False, indent=1))
print("xor", {k: v for k, v in N["xor"].items() if k not in ("x", "y", "label")})
print("reg best", N["reg"]["best"], "linear", N["reg"]["linear_cv"], "natural", N["reg"]["natural_depth"], {k: (v["train"], v["cv"], v["leaves"]) for k, v in reg.items()})
