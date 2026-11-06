"""Lesson 6 Note: Titanic (two classes) first, then iris (three classes), then Auto MPG (regression).
Run (from this folder): python build_numbers_v4.py -> numbers_v4.json
Titanic: seaborn-data titanic.csv; columns survived, pclass, sex, age, sibsp, parch, fare; rows without age dropped (714);
         sex = 1 for female; stratified 3:1 split, random_state=0 (535 train / 179 test); KFold(5, shuffle, 0).
Iris: load_iris, stratified 3:1 split, random_state=0. Breast cancer: the Lesson 4 split, only for the comparison.
"""
import json, os, warnings
import numpy as np, pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import BaggingClassifier
from sklearn.inspection import permutation_importance
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
r = lambda v, k=3: round(float(v), k)
KF = KFold(5, shuffle=True, random_state=0)
URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
N = {}
ship = pd.read_csv(URL)
ship = ship[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare"]].dropna()
ship["sex"] = (ship["sex"] == "female").astype(int)
X = ship.drop(columns="survived"); y = ship["survived"]; cols = list(X.columns)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
cv = lambda m, XX=Xtr, yy=ytr: r(cross_val_score(m, XX, yy, cv=KF).mean())
N["data"] = dict(rows=len(ship), raw=891, train=len(ytr), test=len(yte), surv=int(ytr.sum()), p=r(ytr.mean()),
                 surv_rate_female=r(ytr[Xtr.sex == 1].mean()), surv_rate_male=r(ytr[Xtr.sex == 0].mean()))

def tree_nodes(t, Xn):
    T = t.tree_; out = []
    for i in range(T.node_count):
        n = int(T.n_node_samples[i]); s = int(round(T.value[i][0][1] * n))
        out.append(dict(i=i, n=n, surv=s, p=r(s / n), gini=r(T.impurity[i]), nG=r(n * T.impurity[i], 1),
                        feat=(Xn[T.feature[i]] if T.feature[i] >= 0 else None), thr=r(T.threshold[i], 2),
                        left=int(T.children_left[i]), right=int(T.children_right[i])))
    return out
# depth-2 tree (F1) and importance by hand
t2 = DecisionTreeClassifier(max_depth=2, random_state=0).fit(Xtr, ytr)
nodes = tree_nodes(t2, cols); dec = []
for nd in nodes:
    if nd["left"] >= 0:
        L, R = nodes[nd["left"]], nodes[nd["right"]]
        T = t2.tree_
        dec.append(dict(node=nd["i"], feat=nd["feat"], thr=nd["thr"], before=r(nd["n"] * T.impurity[nd["i"]], 1),
                        left=r(L["n"] * T.impurity[L["i"]], 1), right=r(R["n"] * T.impurity[R["i"]], 1),
                        drop=r(nd["n"] * T.impurity[nd["i"]] - L["n"] * T.impurity[L["i"]] - R["n"] * T.impurity[R["i"]], 1)))
N["t2"] = dict(nodes=nodes, dec=dec, imp={c: r(v) for c, v in zip(cols, t2.feature_importances_)},
               weighted_root=r((nodes[1]["n"] * nodes[1]["gini"] + nodes[nodes[0]["right"]]["n"] * nodes[nodes[0]["right"]]["gini"]) / nodes[0]["n"]))
# candidate cuts
cands = {c: int(len(np.unique(Xtr[c])) - 1) for c in cols}
tot = sum(cands.values())
N["cands"] = dict(per=cands, total=tot, d2=f"{tot**3:.1e}", d3=f"{tot**7:.1e}")
# depth sweep: train and CV, and five validation splits
sweep = []
for d in range(1, 13):
    t = DecisionTreeClassifier(max_depth=d, random_state=0)
    sweep.append(dict(d=d, train=r(t.fit(Xtr, ytr).score(Xtr, ytr)), cv=cv(t), leaves=int(t.get_n_leaves())))
full = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr)
N["sweep"] = dict(rows=sweep, full_depth=int(full.get_depth()), full_leaves=int(full.get_n_leaves()),
                  full_train=r(full.score(Xtr, ytr)), full_cv=cv(DecisionTreeClassifier(random_state=0)))
vals = []
for seed in range(5):
    Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.25, random_state=seed, stratify=ytr)
    acc = [r(DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xa, ya).score(Xv, yv)) for d in range(1, 13)]
    vals.append(dict(seed=seed, acc=acc, best=int(np.argmax(acc)) + 1))
N["valsplits"] = dict(n_fit=len(ya), n_val=len(yv), splits=vals)
best = max(sweep, key=lambda s: s["cv"])["d"]
t4 = DecisionTreeClassifier(max_depth=best, random_state=0).fit(Xtr, ytr)
N["best"] = dict(depth=best, cv=cv(DecisionTreeClassifier(max_depth=best, random_state=0)), test=r(t4.score(Xte, yte)),
                 leaves=int(t4.get_n_leaves()), imp={c: r(v) for c, v in zip(cols, t4.feature_importances_)},
                 logit=cv(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))))
# a passenger read along a depth-3 tree: the first third-class woman in the test set
t3 = DecisionTreeClassifier(max_depth=3, random_state=0).fit(Xtr, ytr)
cand = Xte[(Xte.sex == 1) & (Xte.pclass == 3)]
row = cand.iloc[[0]]; path = t3.decision_path(row).indices
steps = [dict(feat=cols[t3.tree_.feature[n]], thr=r(t3.tree_.threshold[n], 2), val=r(row.iloc[0, t3.tree_.feature[n]], 2))
         for n in path if t3.tree_.feature[n] >= 0]
leaf = path[-1]; ln = int(t3.tree_.n_node_samples[leaf])
w3 = Xtr[(Xtr.sex == 1) & (Xtr.pclass == 3)]; w3y = ytr[w3.index]
cheap = w3.fare <= 20.8
N["path"] = dict(passenger=row.iloc[0].to_dict(), survived=int(yte[row.index[0]]), steps=steps, leaf_n=ln,
                 leaf_surv=int(round(t3.tree_.value[leaf][0][1] * ln)), proba=r(t3.predict_proba(row)[0, 1]),
                 cv3=cv(DecisionTreeClassifier(max_depth=3, random_state=0)),
                 w3_cheap=[int(cheap.sum()), r(w3y[cheap].mean())], w3_dear=[int((~cheap).sum()), r(w3y[~cheap].mean())],
                 family_cheap=r((w3.sibsp + w3.parch)[cheap].mean(), 1), family_dear=r((w3.sibsp + w3.parch)[~cheap].mean(), 1))
N["t3"] = dict(nodes=tree_nodes(t3, cols), path=[int(n) for n in path])
# for the Slides: weighted Gini of a cut on age, for 30 men drawn at random (the loss is a staircase)
rs = np.random.default_rng(1); pick = rs.choice(np.where(Xtr.sex.values == 0)[0], 30, replace=False)
ag = Xtr.age.values[pick]; yy = ytr.values[pick]; grid = np.linspace(0, 75, 1501); wg = []
for c in grid:
    L, R = yy[ag <= c], yy[ag > c]
    g = lambda v: 2 * v.mean() * (1 - v.mean()) if len(v) else 0.0
    wg.append(r((len(L) * g(L) + len(R) * g(R)) / len(yy), 4))
N["stair"] = dict(grid=[r(v, 2) for v in grid], wgini=wg, ages=[r(v, 1) for v in ag], labels=[int(v) for v in yy])
# the small investigation: drop one feature at a time
base = cross_val_score(DecisionTreeClassifier(max_depth=best, random_state=0), Xtr, ytr, cv=KF)
inv = {}
for c in ["fare", "pclass", "sex"]:
    k = [x for x in cols if x != c]
    sc = cross_val_score(DecisionTreeClassifier(max_depth=best, random_state=0), Xtr[k], ytr, cv=KF)
    tt = DecisionTreeClassifier(max_depth=best, random_state=0).fit(Xtr[k], ytr)
    inv[c] = dict(cv=r(sc.mean()), diff=[r(v) for v in sc - base], imp={x: r(v) for x, v in zip(k, tt.feature_importances_)})
N["inv"] = dict(base_folds=[r(v) for v in base], drops=inv, corr_pclass_fare=r(Xtr.pclass.corr(Xtr.fare), 2),
                fare_by_class={int(k): r(v, 1) for k, v in Xtr.groupby("pclass").fare.median().items()})
# Advanced: hand best cut on age among the men; stability; bagging; permutation importance
men = Xtr.sex == 0; age = Xtr.age[men].values; ym = ytr[men].values
vv = np.unique(age); cuts = (vv[:-1] + vv[1:]) / 2; bs, bc = 1, None
for c in cuts:
    L, R = ym[age <= c], ym[age > c]
    s = (len(L) * 2 * L.mean() * (1 - L.mean()) + len(R) * 2 * R.mean() * (1 - R.mean())) / len(ym)
    if s < bs: bs, bc = s, c
N["adv1"] = dict(n_men=int(men.sum()), best_cut=r(bc, 2), score=r(bs), before=r(2 * ym.mean() * (1 - ym.mean())))
rng = np.random.default_rng(0); stab = []
for k in range(10):
    idx = rng.integers(0, len(ytr), len(ytr))
    t = DecisionTreeClassifier(max_depth=3, random_state=0).fit(Xtr.iloc[idx], ytr.iloc[idx]); T = t.tree_
    stab.append([cols[T.feature[0]], cols[T.feature[T.children_left[0]]], r(T.threshold[T.children_left[0]], 1),
                 cols[T.feature[T.children_right[0]]], r(T.threshold[T.children_right[0]], 1)])
N["adv2"] = stab
N["adv3"] = dict(single=cv(DecisionTreeClassifier(random_state=0)),
                 bag=cv(BaggingClassifier(DecisionTreeClassifier(random_state=0), n_estimators=25, random_state=0)))
Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.25, random_state=0, stratify=ytr)
tf = DecisionTreeClassifier(max_depth=best, random_state=0).fit(Xa, ya)
pi = permutation_importance(tf, Xv, yv, n_repeats=20, random_state=0)
N["adv5"] = dict(perm={c: r(v) for c, v in zip(cols, pi.importances_mean)}, gini={c: r(v) for c, v in zip(cols, tf.feature_importances_)})
# iris
ir = load_iris(); Xi, yi = ir.data, ir.target
Xi_tr, Xi_te, yi_tr, yi_te = train_test_split(Xi, yi, test_size=0.25, random_state=0, stratify=yi)
ti = DecisionTreeClassifier(max_depth=2, random_state=0).fit(Xi_tr, yi_tr); Ti = ti.tree_
cnt = lambda i: [int(round(v * Ti.n_node_samples[i])) for v in Ti.value[i][0]]
inodes = [dict(i=i, counts=cnt(i), gini=r(Ti.impurity[i]), feat=(ir.feature_names[Ti.feature[i]] if Ti.feature[i] >= 0 else None),
               thr=r(Ti.threshold[i], 2)) for i in range(Ti.node_count)]
N["iris"] = dict(n=len(yi_tr), nodes=inodes, imp=[r(v) for v in ti.feature_importances_],
                 cv=[cv(DecisionTreeClassifier(max_depth=k, random_state=0), Xi_tr, yi_tr) for k in range(1, 6)],
                 logit=cv(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), Xi_tr, yi_tr),
                 corr_len_wid=r(np.corrcoef(Xi_tr[:, 2], Xi_tr[:, 3])[0, 1]),
                 test=r(ti.score(Xi_te, yi_te)), stump_len=cv(DecisionTreeClassifier(max_depth=2, random_state=0), Xi_tr[:, [2]], yi_tr),
                 stump_wid=cv(DecisionTreeClassifier(max_depth=2, random_state=0), Xi_tr[:, [3]], yi_tr))
# breast cancer comparison only
d = load_breast_cancer(); Xb, yb = d.data, 1 - d.target
Xb_tr, _, yb_tr, _ = train_test_split(Xb, yb, test_size=0.25, random_state=0, stratify=yb)
N["bc"] = dict(tree5=cv(DecisionTreeClassifier(max_depth=5, random_state=0), Xb_tr, yb_tr),
               logit=cv(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)), Xb_tr, yb_tr))
json.dump(N, open(os.path.join(HERE, "numbers_v4.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(N, ensure_ascii=False))
