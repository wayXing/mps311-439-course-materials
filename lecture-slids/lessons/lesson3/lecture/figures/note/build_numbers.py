"""Lesson 3 Note: every number in the note, the figures and the widgets.
Run (from this folder):  python build_numbers.py           -> numbers.json
                         python build_numbers.py --fast    (skips the slow repeated-sample experiment if cached)
Rules fixed before looking at results:
  split        train_test_split(test_size=0.2, random_state=42)  (same as Lesson 2)
  ten cars     first 10 rows of the training set (Lesson 2 classroom cars)
  opening car  the test car with the largest horsepower
  CV           KFold(5) (= cv=5), inside the 313 training cars only
  poly design  PolynomialFeatures -> StandardScaler -> model, always in one Pipeline
"""
import json, os, sys, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV, RidgeCV, lasso_path
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
FAST = "--fast" in sys.argv
OUTJ = os.path.join(HERE, "numbers.json")
old = json.load(open(OUTJ)) if os.path.exists(OUTJ) else {}

df = pd.read_csv(os.path.join(HERE, "auto_mpg.csv"))
FEAT6 = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"]
X, y = df[FEAT6], df["mpg"].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
assert len(Xtr) == 313 and len(Xte) == 79
HP = ["horsepower"]
rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
r = lambda v, k=3: round(float(v), k)
KF = KFold(5)  # same folds as cv=5 in the note's code (training rows are already shuffled by the split)
N = {}

def poly(d, model):
    return make_pipeline(PolynomialFeatures(d, include_bias=False), StandardScaler(), model)

def cv_rmse(m, Xa, ya):
    s = -cross_val_score(m, Xa, ya, cv=KF, scoring="neg_root_mean_squared_error")
    return float(s.mean()), float(s.std(ddof=1) / np.sqrt(len(s)))

# ---------- opening car ----------
i0 = Xte.horsepower.idxmax()
car = Xte.loc[i0]
lin = LinearRegression().fit(Xtr[HP], ytr)
quad = poly(2, LinearRegression()).fit(Xtr[HP], ytr)
q_raw = LinearRegression().fit(np.c_[Xtr.horsepower, Xtr.horsepower ** 2], ytr)
N["car"] = dict(hp=float(car.horsepower), wt=float(car.weight), year=int(car.model_year), cyl=int(car.cylinders),
                mpg=float(df.loc[i0, "mpg"]), pred_line=r(lin.predict(car[HP].to_frame().T)[0], 2),
                pred_quad=r(quad.predict(car[HP].to_frame().T)[0], 2))
N["line"] = dict(w=r(lin.coef_[0], 4), b=r(lin.intercept_, 2),
                 train=r(rmse(ytr, lin.predict(Xtr[HP])), 2), test=r(rmse(yte, lin.predict(Xte[HP])), 2))
N["quad"] = dict(b=r(q_raw.intercept_, 2), w1=r(q_raw.coef_[0], 4), w2=float(f"{q_raw.coef_[1]:.3g}"),
                 train=r(rmse(ytr, quad.predict(Xtr[HP])), 2), test=r(rmse(yte, quad.predict(Xte[HP])), 2))
# grouped mean residuals (five equal-count groups by hp), line vs quadratic
grp = pd.qcut(Xtr.horsepower, 5, labels=False)
N["resid_groups"] = dict(
    hp_mid=[r(Xtr.horsepower[grp == g].median(), 0) for g in range(5)],
    line=[r((ytr - lin.predict(Xtr[HP]))[grp.values == g].mean(), 2) for g in range(5)],
    quad=[r((ytr - quad.predict(Xtr[HP]))[grp.values == g].mean(), 2) for g in range(5)])

# ---------- ten cars, degree 1..7 ----------
X10, y10 = Xtr[HP].iloc[:10], ytr[:10]
N["ten"] = dict(hp=X10.horsepower.tolist(), mpg=[float(v) for v in y10],
                distinct_hp=int(X10.horsepower.nunique()))
base10 = float(y10.mean())
N["ten"]["baseline_test"] = r(rmse(yte, np.full(len(yte), base10)), 2)
grid = np.linspace(40, 235, 196)
deg_rows, curves = [], {}
for d in range(1, 8):
    m = poly(d, LinearRegression()).fit(X10, y10)
    co = m[-1].coef_
    deg_rows.append(dict(d=d, train=r(rmse(y10, m.predict(X10)), 2), test=r(rmse(yte, m.predict(Xte[HP])), 2),
                         maxcoef=float(f"{np.abs(co).max():.3g}"), coefs=[float(f"{v:.4g}") for v in co]))
    curves[d] = [r(v, 3) for v in m.predict(pd.DataFrame({"horsepower": grid}))]
N["ten"]["degrees"] = deg_rows

# ---------- Ridge on the ten cars, degree 7 ----------
D_R = 7
lams = [0.0] + [float(v) for v in np.logspace(-4, 3, 57)]
ridge_rows, rcurves = [], []
for lam in lams:
    m = poly(D_R, Ridge(alpha=lam) if lam > 0 else LinearRegression()).fit(X10, y10)
    ridge_rows.append(dict(lam=lam, train=r(rmse(y10, m.predict(X10)), 2), test=r(rmse(yte, m.predict(Xte[HP])), 2),
                           norm=float(f"{np.linalg.norm(m[-1].coef_):.4g}"), coefs=[float(f"{v:.4g}") for v in m[-1].coef_]))
    rcurves.append([r(v, 3) for v in m.predict(pd.DataFrame({"horsepower": grid}))])
def at(lam):
    return min(ridge_rows, key=lambda q: abs(np.log10(max(q["lam"], 1e-9)) - np.log10(lam)))
N["ridge_ten"] = dict(degree=D_R, lam0=ridge_rows[0], picks={str(l): at(l) for l in [0.001, 0.01, 0.1, 1, 10, 100, 1000]})
# Lasso on the same ten cars, degree 7: how many of the 7 coefficients stay non-zero
N["lasso_ten"] = {}
for lam in [0.001, 0.01, 0.1, 0.5, 1, 2, 5]:
    m = poly(D_R, Lasso(alpha=lam, max_iter=500000)).fit(X10, y10)
    N["lasso_ten"][str(lam)] = dict(nz=int((np.abs(m[-1].coef_) > 1e-8).sum()), test=r(rmse(yte, m.predict(Xte[HP])), 2))
N["ridge_ten"]["min_nonzero"] = int(min((np.abs(np.array(q["coefs"])) > 1e-8).sum() for q in ridge_rows[1:]))  # Ridge never sets a coefficient to zero
# p = 1 shrinkage on the ten cars (standardised hp): w(lam) = w_ols * n/(n+lam)
z = (X10.horsepower - X10.horsepower.mean()) / X10.horsepower.std(ddof=0)
w1 = float(np.sum(z * (y10 - y10.mean())) / np.sum(z ** 2))
N["p1"] = dict(n=10, w_ols=r(w1, 3), w_lam10=r(w1 * 10 / 20, 3), sxx=r(np.sum(z ** 2), 3),
               check_sklearn=r(Ridge(alpha=10).fit(z.values.reshape(-1, 1), y10).coef_[0], 3))

# ---------- 313 cars: degree by CV ----------
cv_deg = []
for d in range(1, 11):
    m = poly(d, LinearRegression())
    mu, se = cv_rmse(m, Xtr[HP], ytr)
    m.fit(Xtr[HP], ytr)
    cv_deg.append(dict(d=d, cv=r(mu), se=r(se), train=r(rmse(ytr, m.predict(Xtr[HP]))), test=r(rmse(yte, m.predict(Xte[HP])))))
N["cv_degree"] = cv_deg
best = min(cv_deg, key=lambda q: q["cv"])
simple = min(q["d"] for q in cv_deg if q["cv"] <= best["cv"] + best["se"])
N["cv_degree_choice"] = dict(best_d=best["d"], best_cv=best["cv"], best_se=best["se"], one_se_d=simple)
# a single 80/20 validation split inside the 313, five different seeds, to show its wobble
wob = []
for s in range(5):
    a, b, ya, yb = train_test_split(Xtr[HP], ytr, test_size=0.2, random_state=s)
    row = [r(rmse(yb, poly(d, LinearRegression()).fit(a, ya).predict(b)), 2) for d in (1, 2, 6)]
    wob.append(row)
N["single_val"] = dict(degrees=[1, 2, 6], seeds=wob)

# ---------- 27 features: Ridge / Lasso by CV ----------
P2 = lambda m: poly(2, m)
nfeat = int(PolynomialFeatures(2, include_bias=False).fit(Xtr).n_output_features_)
ols27 = P2(LinearRegression())
ols_cv, ols_se = cv_rmse(ols27, Xtr, ytr)
ols27.fit(Xtr, ytr)
RG = np.logspace(-3, 2, 26)
rcv = [dict(lam=float(l), **dict(zip(["cv", "se"], map(lambda v: r(v), cv_rmse(P2(Ridge(alpha=l)), Xtr, ytr))))) for l in RG]
rb = min(rcv, key=lambda q: q["cv"])
rfit = P2(Ridge(alpha=rb["lam"])).fit(Xtr, ytr)
# the note's short code uses RidgeCV(cv=5, scoring=RMSE); it must pick the same lambda as the grid above
rcv_short = P2(RidgeCV(alphas=RG, cv=5, scoring="neg_root_mean_squared_error")).fit(Xtr, ytr)
assert abs(rcv_short[-1].alpha_ - rb["lam"]) < 1e-12, (rcv_short[-1].alpha_, rb["lam"])
test_short = r(rmse(yte, rcv_short.predict(Xte)))
LG = np.logspace(-4, 1, 26)
lcv = []
for l in LG:
    m = P2(Lasso(alpha=l, max_iter=200000))
    mu, se = cv_rmse(m, Xtr, ytr)
    nz = int((np.abs(m.fit(Xtr, ytr)[-1].coef_) > 1e-8).sum())
    lcv.append(dict(lam=float(l), cv=r(mu), se=r(se), nz=nz))
lb = min(lcv, key=lambda q: q["cv"])
lfit = P2(Lasso(alpha=lb["lam"], max_iter=200000)).fit(Xtr, ytr)
names = lfit[0].get_feature_names_out(FEAT6)
N["p27"] = dict(nfeat=nfeat,
                ols=dict(cv=r(ols_cv), se=r(ols_se), train=r(rmse(ytr, ols27.predict(Xtr))), test=r(rmse(yte, ols27.predict(Xte))),
                         maxcoef=r(np.abs(ols27[-1].coef_).max(), 1)),
                ridge=dict(lam=rb["lam"], cv=rb["cv"], se=rb["se"], train=r(rmse(ytr, rfit.predict(Xtr))), test=r(rmse(yte, rfit.predict(Xte))),
                           maxcoef=r(np.abs(rfit[-1].coef_).max(), 1)),
                lasso=dict(lam=lb["lam"], cv=lb["cv"], se=lb["se"], nz=lb["nz"], train=r(rmse(ytr, lfit.predict(Xtr))), test=r(rmse(yte, lfit.predict(Xte))),
                           kept=[n for n, c in zip(names, lfit[-1].coef_) if abs(c) > 1e-8]),
                ridge_curve=rcv, lasso_curve=lcv)
N["p27"]["ridge"]["test_short_code"] = test_short
for l in [1e-4, 1e-3, 1e-2, 0.1, 1, 10]:
    m = P2(Lasso(alpha=l, max_iter=200000)).fit(Xtr, ytr)
    N["p27"].setdefault("lasso_nz_at", {})[str(l)] = int((np.abs(m[-1].coef_) > 1e-8).sum())
# lasso path for the figure
Z27 = StandardScaler().fit_transform(PolynomialFeatures(2, include_bias=False).fit_transform(Xtr))
alphas_p, coefs_p, _ = lasso_path(Z27, ytr - ytr.mean(), alphas=np.logspace(-4, 1, 80), max_iter=200000)
N["lasso_path"] = dict(alphas=[float(a) for a in alphas_p], coefs=[[r(v, 4) for v in row] for row in coefs_p], names=list(names))
# condition numbers (Advanced Study 1)
ev = np.linalg.eigvalsh(Z27.T @ Z27)
N["cond27"] = dict(xtx=float(f"{ev.max() / ev.min():.3g}"),
                   ridge={str(l): float(f"{(ev.max() + l) / (ev.min() + l):.3g}") for l in [0.01, 0.1, 1, 10]})

# ---------- hp coefficient stability (six raw features, standardised), 500 resamples ----------
rc = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 61), cv=KF)).fit(Xtr, ytr)
lam6 = float(rc[-1].alpha_)
lc = make_pipeline(StandardScaler(), LassoCV(alphas=np.logspace(-4, 0, 41), cv=KF, max_iter=100000)).fit(Xtr, ytr)
lam6L = float(lc[-1].alpha_)
rng = np.random.default_rng(0)
j = FEAT6.index("horsepower")
res = {"ols": [], "ridge": []}
zero = np.zeros(6)
for _ in range(500):
    ii = rng.integers(0, 313, 313)
    Xb, yb = Xtr.iloc[ii], ytr[ii]
    res["ols"].append(make_pipeline(StandardScaler(), LinearRegression()).fit(Xb, yb)[-1].coef_[j])
    res["ridge"].append(make_pipeline(StandardScaler(), Ridge(alpha=lam6)).fit(Xb, yb)[-1].coef_[j])
    zero += np.abs(make_pipeline(StandardScaler(), Lasso(alpha=lam6L, max_iter=100000)).fit(Xb, yb)[-1].coef_) < 1e-8
N["boot"] = dict(lam_ridge=r(lam6, 2), lam_lasso=r(lam6L, 4),
                 ols=[r(v, 2) for v in np.percentile(res["ols"], [2.5, 97.5])],
                 ridge=[r(v, 2) for v in np.percentile(res["ridge"], [2.5, 97.5])],
                 lasso_zero_share={f: r(zero[k] / 500, 2) for k, f in enumerate(FEAT6)},
                 corr=r(np.corrcoef(Xtr.T.values)[FEAT6.index("cylinders"), FEAT6.index("displacement")], 2))

# ---------- training-set size experiment (slow) ----------
if FAST and "size" in old:
    N["size"] = old["size"]
else:
    rng = np.random.default_rng(1)
    out = {}
    for n in [40, 80, 313]:
        reps = 60 if n < 313 else 1
        acc = {"ols": [], "ridge": [], "lasso": [], "lasso_nz": []}
        for t in range(reps):
            ii = rng.choice(313, n, replace=False) if n < 313 else np.arange(313)
            kf = KFold(5, shuffle=True, random_state=t)
            Xa, ya = Xtr.iloc[ii], ytr[ii]
            acc["ols"].append(rmse(yte, P2(LinearRegression()).fit(Xa, ya).predict(Xte)))
            acc["ridge"].append(rmse(yte, P2(RidgeCV(alphas=RG, cv=kf)).fit(Xa, ya).predict(Xte)))
            lm = P2(LassoCV(alphas=LG, cv=kf, max_iter=50000)).fit(Xa, ya)
            acc["lasso"].append(rmse(yte, lm.predict(Xte)))
            acc["lasso_nz"].append(int((np.abs(lm[-1].coef_) > 1e-8).sum()))
        out[str(n)] = {k: dict(median=r(np.median(v), 2), q25=r(np.percentile(v, 25), 2), q75=r(np.percentile(v, 75), 2))
                       for k, v in acc.items()}
        out[str(n)]["reps"] = reps
    N["size"] = out

# ---------- feature-engineering examples beyond the anchor ----------
from sklearn.datasets import fetch_california_housing
bike = pd.read_csv(os.path.join(HERE, "bike_hourly.csv"))
P = 2 * np.pi * bike.hour / 24
bx = {"raw": bike[["hour"]].values, "sincos": np.c_[np.sin(P), np.cos(P)],
      "sincos2": np.c_[np.sin(P), np.cos(P), np.sin(2 * P), np.cos(2 * P)]}
bres = {}
for k, Xb in bx.items():
    a_, b_, ya_, yb_ = train_test_split(Xb, bike["count"].values, test_size=0.2, random_state=42)
    m_ = LinearRegression().fit(a_, ya_)
    bres[k] = r(m_.score(b_, yb_), 2)
    bres[k + "_rmse"] = r(rmse(yb_, m_.predict(b_)), 1)
hourly = bike.groupby("hour")["count"].mean()
fits = {}
for k, Xb in bx.items():
    m = LinearRegression().fit(Xb, bike["count"].values)
    Pg = 2 * np.pi * np.arange(24) / 24
    g = {"raw": np.arange(24).reshape(-1, 1), "sincos": np.c_[np.sin(Pg), np.cos(Pg)],
         "sincos2": np.c_[np.sin(Pg), np.cos(Pg), np.sin(2 * Pg), np.cos(2 * Pg)]}[k]
    fits[k] = [r(v, 1) for v in m.predict(g)]
N["bike"] = dict(n=int(len(bike)), r2=bres, hourly_mean=[r(v, 1) for v in hourly.values], fits=fits)
ca = fetch_california_housing(as_frame=True).frame
N["calif"] = dict(n=int(len(ca)), corr_bedrooms=r(np.corrcoef(ca.AveBedrms, ca.MedHouseVal)[0, 1], 2),
                  corr_rooms=r(np.corrcoef(ca.AveRooms, ca.MedHouseVal)[0, 1], 2),
                  corr_ratio=r(np.corrcoef(ca.AveBedrms / ca.AveRooms, ca.MedHouseVal)[0, 1], 2))
wtr, wte = Xtr[["weight"]], Xte[["weight"]]
m_mpg = LinearRegression().fit(wtr, ytr)
m_gpm = LinearRegression().fit(wtr, 100 / ytr)
N["gp100"] = dict(corr_mpg=r(np.corrcoef(Xtr.weight, ytr)[0, 1], 2), corr_gp100=r(np.corrcoef(Xtr.weight, 100 / ytr)[0, 1], 2),
                  test_mpg=r(rmse(yte, m_mpg.predict(wte)), 2), test_via_gp100=r(rmse(yte, 100 / m_gpm.predict(wte)), 2))
# scales of the features a reader meets in section 3 (training cars)
N["scales"] = dict(hp=[r(Xtr.horsepower.min(), 0), r(Xtr.horsepower.max(), 0)],
                   hp2=[r(Xtr.horsepower.min() ** 2, 0), r(Xtr.horsepower.max() ** 2, 0)],
                   wt=[r(Xtr.weight.min(), 0), r(Xtr.weight.max(), 0)],
                   hp7_max=float(f"{Xtr.horsepower.max() ** 7:.2g}"))
# raw (unstandardised) degree-2 coefficients vs standardised, to show units drive size
q_std = poly(2, LinearRegression()).fit(Xtr[HP], ytr)[-1].coef_
N["quad"]["std_coefs"] = [r(v, 2) for v in q_std]

# ---------- widget data ----------
N["widget"] = dict(grid=[float(g) for g in grid], deg_curves=curves, lams=lams, ridge_curves=rcurves,
                   ridge_rows=ridge_rows, test_hp=Xte.horsepower.tolist(), test_mpg=[float(v) for v in yte])
json.dump(N, open(OUTJ, "w"), ensure_ascii=False, indent=1)

# ---------- print what the note quotes ----------
show = {k: v for k, v in N.items() if k not in ("widget", "lasso_path")}
show["p27"] = {k: v for k, v in N["p27"].items() if k not in ("ridge_curve", "lasso_curve")}
print(json.dumps(show, ensure_ascii=False, indent=1))
