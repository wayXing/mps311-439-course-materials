"""Lesson 2 Note: static figures + data for the interactive widgets.
Run:  python build_note_figures.py  data/auto_mpg.csv  out_dir
Colours (fixed across the note): data blue, model green, error red.
"""
import sys, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

CSV, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
B, G, R = "#0072B2", "#00795A", "#C84B00"
GREY, LIGHT = "#6b6b6b", "#d9d9d9"
plt.rcParams.update({
    "font.family": ["DejaVu Sans"], "svg.fonttype": "path",
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11, "figure.dpi": 100,
})

df = pd.read_csv(CSV)
FEAT6 = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"]
X = df[FEAT6]; y = df["mpg"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
assert len(Xtr) == 313 and len(Xte) == 79
base = ytr.mean()
rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
print("baseline test RMSE", rmse(yte, base), "mean", base)

def fit(cols):
    m = LinearRegression().fit(Xtr[cols], ytr)
    return m, rmse(ytr, m.predict(Xtr[cols])), rmse(yte, m.predict(Xte[cols]))
M = {1: ["horsepower"], 2: ["horsepower", "weight"],
     3: ["cylinders", "displacement", "horsepower", "weight"], 4: FEAT6}
models = {k: fit(c) for k, c in M.items()}
for k, (m, a, b) in models.items():
    print("model", k, "hp coef", round(m.coef_[M[k].index("horsepower")], 4), "train", round(a, 2), "test", round(b, 2))

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.svg", bbox_inches="tight")
    plt.close(fig)

# three named cars
three = pd.DataFrame({"hp": [130., 165., 150.], "mpg": [18., 15., 18.]})
xs, ys = three.hp.values, three.mpg.values
w3 = np.sum((xs - xs.mean()) * (ys - ys.mean())) / np.sum((xs - xs.mean()) ** 2)
b3 = ys.mean() - w3 * xs.mean()
mse3 = np.mean((ys - (b3 + w3 * xs)) ** 2)
print("three cars", w3, b3, mse3, np.sqrt(mse3))
mseA = np.mean((ys - (40 - 0.15 * xs)) ** 2); mseB = np.mean((ys - (40 - 0.16 * xs)) ** 2)
print("A", mseA, np.sqrt(mseA), "B", mseB, np.sqrt(mseB))
test_rmse3 = rmse(yte, b3 + w3 * Xte["horsepower"])
print("three-car line on test", test_rmse3)

# ---------------- F2: data table ----------------
rows = df.head(4)
cols = FEAT6 + ["mpg"]
heads = ["cyl", "disp", "hp", "wt", "acc", "year", "mpg"]
fig, ax = plt.subplots(figsize=(7.4, 2.5)); ax.axis("off")
ax.set_xlim(0, 8); ax.set_ylim(0, 5.2)
ax.add_patch(Rectangle((0.9, 0.35), 5.7, 3.85, color=B, alpha=0.08, lw=0))
ax.add_patch(Rectangle((6.65, 0.35), 1.2, 3.85, color=B, alpha=0.20, lw=0))
ax.text(3.75, 4.9, "Feature x (used to predict)", ha="center", color=B, weight="bold")
ax.text(7.25, 4.9, "Target y", ha="center", color=B, weight="bold")
ax.plot([0.95, 6.55], [4.62, 4.62], color=B, lw=1.5); ax.plot([6.7, 7.8], [4.62, 4.62], color=B, lw=1.5)
xpos = [1.3, 2.35, 3.4, 4.45, 5.5, 6.2, 7.25]
ax.text(0.45, 3.85, "Obs.", ha="center", color=GREY, weight="bold")
for xp, h in zip(xpos, heads): ax.text(xp, 3.85, h, ha="center", weight="bold")
for i, (_, r) in enumerate(rows.iterrows()):
    yy = 3.05 - i * 0.75
    ax.text(0.45, yy, str(i + 1), ha="center", color=GREY)
    vals = [f"{int(r.cylinders)}", f"{r.displacement:.0f}", f"{r.horsepower:.0f}", f"{int(r.weight)}",
            f"{r.acceleration:.1f}", f"{int(r.model_year)}", f"{r.mpg:.0f}"]
    for xp, v in zip(xpos, vals): ax.text(xp, yy, v, ha="center")
ax.text(4.0, 0.0, "Each row is one observation (one car)", ha="center", color=GREY, fontsize=10)
save(fig, "F2")

# ---------------- F4: A vs B residuals ----------------
fig, axs = plt.subplots(1, 2, figsize=(8, 3.3), sharey=True)
xl = np.array([120, 172])
for ax, (name, b, w, mse) in zip(axs, [("Line A", 40, -0.15, mseA), ("Line B", 40, -0.16, mseB)]):
    ax.plot(xl, b + w * xl, color=G, lw=2.2)
    for xi, yi in zip(xs, ys):
        ax.plot([xi, xi], [yi, b + w * xi], color=R, lw=2.2)
    ax.scatter(xs, ys, color=B, s=60, zorder=3)
    ax.set_title(f"{name}: MSE {mse:.2f}, RMSE {np.sqrt(mse):.2f} mpg")
    ax.set_xlabel("Horsepower (hp)"); ax.set_xlim(120, 172); ax.set_ylim(11, 22)
axs[0].set_ylabel("mpg")
axs[1].text(121, 11.6, "Red segments are residuals", color=R, fontsize=10)
fig.tight_layout(); save(fig, "F4")

# ---------------- F5: J(b,w) contours + 3D surface ----------------
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
bb = np.linspace(10, 50, 200); ww = np.linspace(-0.22, 0.02, 200)
BB, WW = np.meshgrid(bb, ww)
J = np.mean((ys[None, None, :] - (BB[..., None] + WW[..., None] * xs[None, None, :])) ** 2, axis=-1)
LJ = np.log10(J); lev = np.linspace(LJ.min(), LJ.max(), 15)
fig = plt.figure(figsize=(11, 4.4))
ax = fig.add_subplot(1, 2, 1)
ax.contourf(BB, WW, LJ, levels=lev, cmap="Oranges", alpha=0.85, rasterized=True)
ax.contour(BB, WW, LJ, levels=lev, colors="white", linewidths=0.4, rasterized=True)
ax.scatter([b3], [w3], color=R, s=80, zorder=4, marker="*")
for lab, (bq, wq) in {"A": (40, -0.15), "B": (40, -0.16)}.items():
    ax.scatter([bq], [wq], color=G, s=45, zorder=4)
    ax.text(bq + 0.8, wq + 0.003, lab, color=G, weight="bold")
ax.set_xlabel("Intercept b"); ax.set_ylabel("Slope w"); ax.set_title("From above: contour lines")
ax.spines[["top", "right"]].set_visible(False)
a3 = fig.add_subplot(1, 2, 2, projection="3d")
band = np.floor((LJ - LJ.min()) / (LJ.max() - LJ.min()) * 14.999) / 14
a3.plot_surface(BB, WW, J, facecolors=plt.cm.Oranges(0.05 + 0.85 * band), rstride=2, cstride=2,
                linewidth=0, antialiased=False, shade=False, rasterized=True)
a3.scatter([b3], [w3], [mse3], color=R, s=90, marker="*", zorder=10)
a3.view_init(elev=32, azim=-58)
a3.set_xlabel("b"); a3.set_ylabel("w"); a3.set_zlabel("MSE", labelpad=12)
a3.set_title("From the side: the loss is a bowl")
fig.tight_layout(); save(fig, "F5")

# ---------------- bootstrap for F6 ----------------
rng = np.random.default_rng(0)
ranges = {}
for k, cols_ in M.items():
    j = cols_.index("horsepower"); co = []
    Xa, ya = Xtr[cols_].values, ytr.values
    for _ in range(500):
        idx = rng.integers(0, 313, 313)
        co.append(LinearRegression().fit(Xa[idx], ya[idx]).coef_[j])
    ranges[k] = (np.percentile(co, 2.5), np.percentile(co, 97.5))
    print("model", k, "range", np.round(ranges[k], 3))
labels = {1: "Model 1: hp", 2: "Model 2: hp, weight", 3: "Model 3: cylinders,\ndisplacement, hp, weight", 4: "Model 4: six features"}
fig, ax = plt.subplots(figsize=(7.2, 3.1))
for i, k in enumerate([1, 2, 3, 4]):
    lo, hi = ranges[k]; est = models[k][0].coef_[M[k].index("horsepower")]
    ax.plot([lo, hi], [-i, -i], color=G, lw=5, solid_capstyle="round", alpha=0.45)
    ax.scatter([est], [-i], color=G, s=45, zorder=3)
    ax.text(0.095, -i, f"{lo:.2f} to {hi:+.2f}", va="center", fontsize=10, color=GREY)
ax.axvline(0, color=GREY, lw=1, ls="--")
ax.set_yticks([0, -1, -2, -3]); ax.set_yticklabels([labels[k] for k in [1, 2, 3, 4]])
ax.set_xlim(-0.22, 0.20); ax.set_xlabel("Coefficient of hp (mpg per hp)\nDot: fit on all 313 training cars. Bar: 95% range.")
ax.text(0.095, 0.55, "95% range", fontsize=10, color=GREY)
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
fig.tight_layout(); save(fig, "F6")

# ---------------- F7: three-car line vs 79 test cars ----------------
w313 = models[1][0].coef_[0]; b313 = models[1][0].intercept_
fig, ax = plt.subplots(figsize=(7, 4))
xl = np.array([40, 230])
ax.scatter(Xte.horsepower, yte, facecolors="none", edgecolors=B, s=32, label="79 test cars")
ax.scatter(xs, ys, color=B, s=70, zorder=3, label="3 training cars")
ax.plot(xl, b3 + w3 * xl, color=G, lw=2.4, label=f"Line fitted to 3 cars (test RMSE {test_rmse3:.2f})")
ax.plot(xl, b313 + w313 * xl, color=G, lw=1.6, ls="--", label="Line fitted to 313 training cars (reference)")
ax.set_ylim(0, 48); ax.set_xlim(40, 235); ax.set_xlabel("Horsepower (hp)"); ax.set_ylabel("mpg")
ax.legend(frameon=False, fontsize=9.5, loc="upper right")
ax.set_title(f"RMSE is {np.sqrt(mse3):.2f} on the 3 cars but {test_rmse3:.2f} mpg on new cars")
fig.tight_layout(); save(fig, "F7")

# ---------------- F9a: hook car ----------------
m2 = models[2][0]
errs = np.abs(yte.values - m2.predict(Xte[M[2]])); target = models[2][2]
hi = int(np.argmin(np.abs(errs - target)))
car = Xte.iloc[hi]; pred = float(m2.predict(Xte[M[2]].iloc[[hi]])[0]); act = float(yte.iloc[hi])
print("hook car", Xte.index[hi], car.to_dict(), act, pred, abs(act - pred))
fig, ax = plt.subplots(figsize=(7.2, 2.5))
ax.axhspan(-0.5, 0.5, color="none")
ax.fill_betweenx([-0.18, 0.18], pred - target, pred + target, color=G, alpha=0.15, lw=0)
ax.plot([pred, act], [0, 0], color=R, lw=3, solid_capstyle="butt")
ax.scatter([pred], [0], color=G, s=110, zorder=3); ax.scatter([act], [0], color=B, s=110, zorder=3)
ax.text(pred + 0.3, 0.32, f"Model 2 prediction {pred:.1f}", ha="right", color=G, weight="bold")
ax.text(act - 0.3, 0.32, f"Actual {act:.1f}", ha="left", color=B, weight="bold")
ax.text((pred + act) / 2, -0.42, f"Error {abs(act - pred):.1f}", ha="center", color=R, weight="bold")
ax.text(pred - target, -0.22, f"Typical error ±{target:.2f}", ha="left", color=G, fontsize=9.5, va="top")
ax.set_xlim(2, 24); ax.set_ylim(-0.8, 0.7); ax.set_yticks([]); ax.set_xlabel("mpg")
ax.spines["left"].set_visible(False)
fig.tight_layout(); save(fig, "F9a")

# ---------------- F9b: residual of model 1 vs hp ----------------
m1 = models[1][0]
res = ytr.values - m1.predict(Xtr[["horsepower"]])
q = pd.qcut(Xtr.horsepower, 5, labels=False)
means = [res[q.values == i].mean() for i in range(5)]
centers = [Xtr.horsepower[q.values == i].mean() for i in range(5)]
print("quintile mean residuals", np.round(means, 2))
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.scatter(Xtr.horsepower, res, color=R, alpha=0.28, s=18, lw=0)
ax.axhline(0, color=GREY, lw=1)
ax.plot(centers, means, color=R, lw=2.4, marker="o")
for c, mu, off in zip(centers, means, [1.2, 1.4, -2.0, -2.0, 1.2]):
    ax.text(c, mu + off, f"{mu:+.1f}", ha="center", color=R, weight="bold", fontsize=10)
ax.set_xlabel("Horsepower (hp)"); ax.set_ylabel("Residual (mpg)")
ax.set_title("Model 1 residuals: mean per hp quintile goes positive, negative, positive")
fig.tight_layout(); save(fig, "F9b")

# ---------------- widget data ----------------
D = {}
# I3
D["I3"] = {"hp": xs.tolist(), "mpg": ys.tolist()}
# I6: all subsets
sets = {}
for mask in range(1, 64):
    cols_ = [c for i, c in enumerate(FEAT6) if mask >> i & 1]
    m = LinearRegression().fit(Xtr[cols_], ytr)
    hpc = float(m.coef_[cols_.index("horsepower")]) if "horsepower" in cols_ else None
    sets[mask] = {"hp": hpc, "rmse": rmse(ytr, m.predict(Xtr[cols_]))}
D["I6"] = {"features": FEAT6, "sets": sets}
# I7: fixed order of training cars, curves for n = 8..313
def curve(perm):
    Xp, yp = Xtr.values[perm], ytr.values[perm]
    tr, te = [], []
    for n in range(8, 314):
        m = LinearRegression().fit(Xp[:n], yp[:n])
        tr.append(rmse(yp[:n], m.predict(Xp[:n]))); te.append(rmse(yte, m.predict(Xte.values)))
    return np.array(tr), np.array(te)
curves = [curve(np.random.default_rng(s).permutation(313)) for s in range(40)]
med_tr = np.median([c[0] for c in curves], 0); med_te = np.median([c[1] for c in curves], 0)
dist = [np.mean(np.abs(np.minimum(c[1], 30) - np.minimum(med_te, 30))) + np.mean(np.abs(c[0] - med_tr)) for c in curves]
best = int(np.argmin(dist)); print("I7 chosen seed", best, "dist", dist[best])
tr, te = curves[best]
D["I7"] = {"n": list(range(8, 314)), "train": tr.tolist(), "test": [min(v, 30) for v in te.tolist()],
           "med_train": med_tr.tolist(), "med_test": [min(v, 30) for v in med_te.tolist()], "base": rmse(yte, base)}
# I5adv
Z = ((Xtr[["horsepower", "weight"]] - Xtr[["horsepower", "weight"]].mean()) / Xtr[["horsepower", "weight"]].std()).values
Z1 = np.c_[np.ones(313), Z]; yv = ytr.values
wstar = np.linalg.lstsq(Z1, yv, rcond=None)[0]
H = 2 / 313 * Z1.T @ Z1
Jstar = float(np.mean((yv - Z1 @ wstar) ** 2))
print("I5adv w*", wstar, "eig", np.linalg.eigvalsh(H), "Jstar", Jstar)
D["I5adv"] = {"wstar": wstar[1:].tolist(), "H": H[1:, 1:].tolist(), "Jstar": Jstar, "eig": np.linalg.eigvalsh(H[1:, 1:]).tolist()}
json.dump(D, open(f"{OUT}/widget_data.json", "w"))
print("done")
