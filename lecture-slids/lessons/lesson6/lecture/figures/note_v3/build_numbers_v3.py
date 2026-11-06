"""Lesson 6 Note V3 (2026-11-05): numbers that are new in V3 (reading a tree, multi-class, regression on all features).
Run (from this folder): python build_numbers_v3.py -> numbers_v3.json
Splits as in V1/V2: breast cancer = Lesson 4 split (426 train / 143 test), 319 fit / 107 validate; KFold(5, shuffle, 0).
Iris = stratified 3:1 split, random_state=0. Auto MPG = Lesson 2 split (test_size=0.2, random_state=42).
"""
import json, os, warnings
import numpy as np, pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.inspection import permutation_importance
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
r = lambda v, k=3: round(float(v), k)
KF = KFold(5, shuffle=True, random_state=0)
N = {}

d = load_breast_cancer(); X = d.data; y = 1 - d.target; names = list(d.feature_names)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
Xf, Xv, yf, yv = train_test_split(Xtr, ytr, test_size=0.25, random_state=0, stratify=ytr)

# A, B: the F1 tree (two features, 319 lumps): leaves and importance by hand
t = DecisionTreeClassifier(max_depth=2, random_state=0).fit(Xf[:, [0, 1]], yf); T = t.tree_
nodes = []
for i in range(T.node_count):
    n = int(T.n_node_samples[i]); mal = int(round(T.value[i][0][1] * n))
    nodes.append(dict(i=i, n=n, mal=mal, p=r(mal / n), gini=r(T.impurity[i]), nG=r(n * T.impurity[i], 1),
                      feat=int(T.feature[i]), thr=r(T.threshold[i], 2), left=int(T.children_left[i]), right=int(T.children_right[i])))
dec = []
for nd in nodes:
    if nd["left"] >= 0:
        L, R = nodes[nd["left"]], nodes[nd["right"]]
        dec.append(dict(node=nd["i"], feat=nd["feat"], before=r(nd["n"] * T.impurity[nd["i"]], 1),
                        left=r(L["n"] * T.impurity[L["i"]], 1), right=r(R["n"] * T.impurity[R["i"]], 1),
                        drop=r(nd["n"] * T.impurity[nd["i"]] - L["n"] * T.impurity[L["i"]] - R["n"] * T.impurity[R["i"]], 1)))
N["f1"] = dict(nodes=nodes, dec=dec, imp=[r(v) for v in t.feature_importances_])

# C: iris, three classes
ir = load_iris(); Xi, yi = ir.data, ir.target
Xi_tr, Xi_te, yi_tr, yi_te = train_test_split(Xi, yi, test_size=0.25, random_state=0, stratify=yi)
ti = DecisionTreeClassifier(max_depth=2, random_state=0).fit(Xi_tr, yi_tr); Ti = ti.tree_
def counts(i): return [int(round(v * Ti.n_node_samples[i])) for v in Ti.value[i][0]]
N["iris"] = dict(n=len(yi_tr), root=counts(0), root_gini=r(Ti.impurity[0]),
                 cut=[ir.feature_names[Ti.feature[0]], r(Ti.threshold[0], 2)],
                 left=counts(Ti.children_left[0]), left_gini=r(Ti.impurity[Ti.children_left[0]]),
                 right=counts(Ti.children_right[0]), right_gini=r(Ti.impurity[Ti.children_right[0]]),
                 imp=[r(v) for v in ti.feature_importances_],
                 corr_len_wid=r(np.corrcoef(Xi_tr[:, 2], Xi_tr[:, 3])[0, 1]),
                 cv=[r(cross_val_score(DecisionTreeClassifier(max_depth=k, random_state=0), Xi_tr, yi_tr, cv=KF).mean()) for k in range(1, 6)])

# D: one lump read along a depth-3 tree on 30 features
t3 = DecisionTreeClassifier(max_depth=3, random_state=0).fit(Xtr, ytr)
i = int(np.where(yte == 1)[0][0]); path = t3.decision_path(Xte[i:i + 1]).indices
steps = [dict(feat=names[t3.tree_.feature[n]], thr=r(t3.tree_.threshold[n], 3), val=r(Xte[i, t3.tree_.feature[n]], 3))
         for n in path if t3.tree_.feature[n] >= 0]
leaf = path[-1]; ln = int(t3.tree_.n_node_samples[leaf])
N["path"] = dict(lump=i, steps=steps, leaf_n=ln, leaf_mal=int(round(t3.tree_.value[leaf][0][1] * ln)),
                 proba=r(t3.predict_proba(Xte[i:i + 1])[0, 1]),
                 cv3=r(cross_val_score(DecisionTreeClassifier(max_depth=3, random_state=0), Xtr, ytr, cv=KF).mean()))

# E: 30 features, depth 5: importance and the small investigation
t5 = DecisionTreeClassifier(max_depth=5, random_state=0).fit(Xtr, ytr)
imp = pd.Series(t5.feature_importances_, names).sort_values(ascending=False)
keep = [j for j in range(30) if names[j] != "mean concave points"]
t5b = DecisionTreeClassifier(max_depth=5, random_state=0).fit(Xtr[:, keep], ytr)
impb = pd.Series(t5b.feature_importances_, [names[j] for j in keep]).sort_values(ascending=False)
mcp = names.index("mean concave points"); wp = names.index("worst perimeter")
N["imp30"] = dict(top=[[k, r(v)] for k, v in imp.head(5).items()], nonzero=int((imp > 0).sum()),
                  leaves=int(t5.get_n_leaves()), perimeter=r(imp["mean perimeter"]), radius=r(imp["mean radius"]),
                  corr_rad_per=r(np.corrcoef(Xtr[:, 0], Xtr[:, 2])[0, 1]),
                  cv_all=r(cross_val_score(DecisionTreeClassifier(max_depth=5, random_state=0), Xtr, ytr, cv=KF).mean()),
                  cv_drop=r(cross_val_score(DecisionTreeClassifier(max_depth=5, random_state=0), Xtr[:, keep], ytr, cv=KF).mean()),
                  top_drop=[[k, r(v)] for k, v in impb.head(5).items()],
                  corr_mcp_wp=r(np.corrcoef(Xtr[:, mcp], Xtr[:, wp])[0, 1]),
                  mcp_mean=[r(Xtr[ytr == 0, mcp], 4) if False else r(Xtr[ytr == 0, mcp].mean(), 3), r(Xtr[ytr == 1, mcp].mean(), 3)])
# G: permutation importance (Advanced): fit on 319, shuffle on the 107 validation lumps
tf = DecisionTreeClassifier(max_depth=5, random_state=0).fit(Xf, yf)
pi = permutation_importance(tf, Xv, yv, n_repeats=20, random_state=0)
pis = pd.Series(pi.importances_mean, names).sort_values(ascending=False)
gis = pd.Series(tf.feature_importances_, names).sort_values(ascending=False)
N["perm"] = dict(perm_top=[[k, r(v)] for k, v in pis.head(5).items()], gini_top=[[k, r(v)] for k, v in gis.head(5).items()],
                 val_acc=r(tf.score(Xv, yv)))

# F: Auto MPG, all six features
cars = pd.read_csv(os.path.join(HERE, "..", "..", "..", "..", "lesson2", "lab", "auto_mpg.csv"))
Xa = cars.drop(columns="mpg"); ya = cars["mpg"]
Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(Xa, ya, test_size=0.2, random_state=42)
rm = lambda m: r(-cross_val_score(m, Xa_tr, ya_tr, cv=KF, scoring="neg_root_mean_squared_error").mean(), 2)
cvd = [rm(DecisionTreeRegressor(max_depth=k, random_state=0)) for k in range(1, 11)]
ta2 = DecisionTreeRegressor(max_depth=2, random_state=0).fit(Xa_tr, ya_tr); A = ta2.tree_
leaves = [dict(n=int(A.n_node_samples[k]), mpg=r(A.value[k][0][0], 1)) for k in range(A.node_count) if A.children_left[k] < 0]
splits = [[list(Xa.columns)[A.feature[k]], r(A.threshold[k], 1)] for k in range(A.node_count) if A.children_left[k] >= 0]
ta5 = DecisionTreeRegressor(max_depth=5, random_state=0).fit(Xa_tr, ya_tr)
C = Xa_tr.corr()
N["mpg"] = dict(cv_depth=cvd, best=int(np.argmin(cvd)) + 1, linear=rm(LinearRegression()), leaves=leaves, splits=splits,
                imp={k: r(v) for k, v in pd.Series(ta5.feature_importances_, Xa.columns).sort_values(ascending=False).items()},
                corr_disp_weight=r(C.loc["displacement", "weight"], 2), corr_disp_hp=r(C.loc["displacement", "horsepower"], 2),
                corr_disp_cyl=r(C.loc["displacement", "cylinders"], 2))

# H: when does a tree win: XOR (unequal clusters, as V2) with cross-validation
rng = np.random.default_rng(0); c = [[1, 1], [4, 4], [1, 4], [4, 1]]; lab = [0, 0, 1, 1]; sizes = (60, 30, 30, 40)
Xx = np.vstack([rng.normal(ci, 0.6, (m, 2)) for ci, m in zip(c, sizes)]); yx = np.concatenate([[l] * m for l, m in zip(lab, sizes)])
N["xor_cv"] = dict(logit=r(cross_val_score(LogisticRegression(), Xx, yx, cv=KF).mean()),
                   tree2=r(cross_val_score(DecisionTreeClassifier(max_depth=2, random_state=0), Xx, yx, cv=KF).mean()),
                   tree3=r(cross_val_score(DecisionTreeClassifier(max_depth=3, random_state=0), Xx, yx, cv=KF).mean()))
json.dump(N, open(os.path.join(HERE, "numbers_v3.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(N, ensure_ascii=False, indent=1))
