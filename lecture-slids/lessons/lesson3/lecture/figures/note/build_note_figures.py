"""Lesson 3 Note: static figures. Reads numbers.json written by build_numbers.py.
Run (from this folder):  python build_note_figures.py            -> Chinese figures here
                         python build_note_figures.py --lang en  -> English figures in ../note_en/
                         python build_note_figures.py --slides   -> English, projector-size text in ../slides/
                                                                    (plus S2, S3 used only in class)
Colours (fixed across the course notes): data blue, model green, error red.
"""
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
SLIDES = "--slides" in os.sys.argv
EN = SLIDES or ("--lang" in os.sys.argv and os.sys.argv[os.sys.argv.index("--lang") + 1] == "en")
OUTDIR = os.path.join(HERE, "..", "slides" if SLIDES else "note_en") if EN else HERE
os.makedirs(OUTDIR, exist_ok=True)
if EN:  # every visible string goes through the translation table; leftover Chinese stops the build
    import matplotlib.text
    from i18n_en import tr, check
    _set_text = matplotlib.text.Text.set_text
    matplotlib.text.Text.set_text = lambda self, s: _set_text(self, check(tr(str(s)), "figure text") if s is not None else s)
N = json.load(open(os.path.join(HERE, "numbers.json")))
B, G, R = "#0072B2", "#00795A", "#C84B00"
GREY, LIGHT = "#6b6b6b", "#d9d9d9"
plt.rcParams.update({
    "font.family": ["DejaVu Sans", "Noto Sans CJK SC", "Hiragino Sans GB", "Arial Unicode MS"],
    "svg.fonttype": "path", "axes.unicode_minus": False, "mathtext.fontset": "dejavusans",
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11, "figure.dpi": 100,
})

if SLIDES:  # projector: text roughly 1.5 times larger at the same figure size
    plt.rcParams.update({"font.size": 16, "axes.titlesize": 17, "axes.labelsize": 16,
                         "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14})

df = pd.read_csv(os.path.join(HERE, "auto_mpg.csv"))
FEAT6 = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"]
Xtr, Xte, ytr, yte = train_test_split(df[FEAT6], df["mpg"].values, test_size=0.2, random_state=42)
hp, hpt = Xtr.horsepower.values, Xte.horsepower.values
car = N["car"]
grid = np.linspace(40, 235, 300)
lin = LinearRegression().fit(Xtr[["horsepower"]], ytr)
quad = make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), LinearRegression()).fit(Xtr[["horsepower"]], ytr)
gl = lin.predict(pd.DataFrame({"horsepower": grid}))
gq = quad.predict(pd.DataFrame({"horsepower": grid}))


def save(fig, name):
    fig.savefig(os.path.join(OUTDIR, f"{name}.svg"), bbox_inches="tight")
    plt.close(fig)


def mark_car(ax, pred, label_pred=True):
    ax.plot([car["hp"], car["hp"]], [car["mpg"], pred], color=R, lw=2.2, zorder=4)
    ax.scatter([car["hp"]], [car["mpg"]], s=90, facecolor="white", edgecolor=B, lw=2.2, zorder=5)
    ax.scatter([car["hp"]], [pred], s=50, color=G, zorder=5)


# ---------------- F1: the straight line and the opening car ----------------
fig, ax = plt.subplots(figsize=(6.6, 4.0))
ax.scatter(hp, ytr, s=12, color=B, alpha=0.45, lw=0, label="313 辆训练车")
ax.plot(grid, gl, color=G, lw=2.2, label="马力直线")
mark_car(ax, car["pred_line"])
ax.annotate(f"实测 {car['mpg']:.1f}", (car["hp"], car["mpg"]), xytext=(-78, 18), textcoords="offset points",
            color=B, weight="bold", arrowprops=dict(arrowstyle="-", color=B, lw=0.8))
ax.annotate(f"直线预测 {car['pred_line']:.1f}", (car["hp"], car["pred_line"]), xytext=(-110, -6), textcoords="offset points",
            color=G, weight="bold", arrowprops=dict(arrowstyle="-", color=G, lw=0.8))
ax.set(xlabel="马力 hp", ylabel="mpg", xlim=(40, 240), ylim=(0, 48))
ax.legend(frameon=False, loc="upper right")
save(fig, "F1")

# ---------------- F2: line vs quadratic, and grouped residuals ----------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 3.9), gridspec_kw=dict(width_ratios=[1.15, 1]))
a1.scatter(hp, ytr, s=12, color=B, alpha=0.45, lw=0)
a1.plot(grid, gl, color=G, lw=2, ls="--", label="直线")
a1.plot(grid, gq, color=G, lw=2.4, label="加入 hp² 后")
mark_car(a1, car["pred_quad"])
a1.set(xlabel="马力 hp", ylabel="mpg", xlim=(40, 240), ylim=(0, 48), title="左：两种模型")
a1.legend(frameon=False, loc="upper right")
g = N["resid_groups"]
a2.axhline(0, color=GREY, lw=0.8)
a2.plot(g["hp_mid"], g["line"], color=R, lw=2, ls="--", marker="o", mfc="white", label="直线")
a2.plot(g["hp_mid"], g["quad"], color=R, lw=2.4, marker="o", label="加入 hp² 后")
a2.set(xlabel="马力 hp（五组，各组中位数）", ylabel="平均残差（mpg）", title="右：各组的平均残差", ylim=(-3, 4))
a2.legend(frameon=False, loc="upper center")
fig.tight_layout()
save(fig, "F2")

# ---------------- F3: bike sharing, hour as raw number vs sin/cos ----------------
bk = N["bike"]
hrs = np.arange(24)
fig, ax = plt.subplots(figsize=(6.8, 3.9))
ax.scatter(hrs, bk["hourly_mean"], s=36, color=B, zorder=4, label="各小时的平均租车量（数据）")
ax.plot(hrs, bk["fits"]["raw"], color=G, lw=2, ls="--", label=(f"特征：小时本身（RMSE {bk['r2']['raw_rmse']:.0f}）" if SLIDES else f"特征：小时本身（R² {bk['r2']['raw']:.2f}）"))
ax.plot(hrs, bk["fits"]["sincos"], color=G, lw=1.8, ls=":", label=(f"特征：sin、cos（RMSE {bk['r2']['sincos_rmse']:.0f}）" if SLIDES else f"特征：sin、cos（R² {bk['r2']['sincos']:.2f}）"))
ax.plot(hrs, bk["fits"]["sincos2"], color=G, lw=2.4, label=(f"再加一组 sin、cos（RMSE {bk['r2']['sincos2_rmse']:.0f}）" if SLIDES else f"再加一组 sin、cos（R² {bk['r2']['sincos2']:.2f}）"))
ax.set(xlabel="一天中的小时", ylabel="每小时租车量", xticks=range(0, 24, 3), xlim=(-0.5, 23.5))
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=9.5)
save(fig, "F3")

# ---------------- F4a: five-fold cross-validation ----------------
fig, ax = plt.subplots(figsize=(9.4, 3.3)); ax.axis("off")
ax.set_xlim(0, 12.0); ax.set_ylim(-0.6, 6.0)
w = 1.3
for k in range(5):
    yy = 4.6 - k * 1.0
    ax.text(0.15, yy + 0.35, f"第 {k + 1} 轮", va="center", fontsize=10)
    for f in range(5):
        x0 = 1.2 + f * w
        val = f == k
        ax.add_patch(Rectangle((x0, yy), w - 0.08, 0.7, facecolor="white" if val else B, alpha=1 if val else 0.25,
                               edgecolor=B, lw=1.6 if val else 0.6, hatch="///" if val else None))
    ax.text(1.2 + 5 * w + 0.2, yy + 0.35, f"在第 {k + 1} 份上算误差", va="center", fontsize=10, color=GREY)
ax.add_patch(Rectangle((10.75, 0.6), 1.15, 4.7, facecolor=LIGHT, edgecolor=GREY, lw=0.8))
ax.text(11.32, 2.95, "测试集\n79 辆\n\n最后\n只用一次", ha="center", va="center", fontsize=10, color=GREY)
ax.text(1.2 + 2.5 * w, 5.65, "313 辆训练车分成 5 份", ha="center", weight="bold", color=B)
ax.add_patch(Rectangle((1.2, -0.45), 0.5, 0.35, facecolor=B, alpha=0.25, edgecolor=B, lw=0.6))
ax.text(1.8, -0.28, "用来拟合", va="center", fontsize=10)
ax.add_patch(Rectangle((3.4, -0.45), 0.5, 0.35, facecolor="white", edgecolor=B, lw=1.6, hatch="///"))
ax.text(4.0, -0.28, "用来验证（这一轮不参与拟合）", va="center", fontsize=10)
save(fig, "F4a")

# ---------------- F4b: cross-validation for the degree (left) and for lambda (right) ----------------
cv = N["cv_degree"]; p = N["p27"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 3.9))
d = [q["d"] for q in cv]
a1.errorbar(d, [q["cv"] for q in cv], yerr=[q["se"] for q in cv], color=R, lw=2.4, marker="o", capsize=3,
            label="交叉验证 RMSE（± 一个标准误）")
a1.plot(d, [q["train"] for q in cv], color=R, lw=2, ls="--", marker="o", mfc="white", label="训练 RMSE")
ch = N["cv_degree_choice"]["one_se_d"]
a1.annotate(f"选 {ch} 阶", (ch, cv[ch - 1]["cv"]), xytext=(25, 45), textcoords="offset points", weight="bold",
            arrowprops=dict(arrowstyle="->", color="#333"))
a1.set(xlabel="多项式阶数（只用马力）", ylabel="RMSE（mpg）", xticks=d, title="左：选阶数")
a1.legend(frameon=False, loc="upper right", fontsize=9.5)
lam = [q["lam"] for q in p["ridge_curve"]]
cvv = np.array([q["cv"] for q in p["ridge_curve"]]); se = np.array([q["se"] for q in p["ridge_curve"]])
a2.fill_between(lam, cvv - se, cvv + se, color=R, alpha=0.12, lw=0)
a2.plot(lam, cvv, color=R, lw=2.4, label="Ridge 的交叉验证 RMSE")
a2.axhline(p["ols"]["cv"], color=GREY, lw=1.2, ls="--")
a2.text(lam[0], p["ols"]["cv"] + 0.012, f"不加惩罚 {p['ols']['cv']:.2f}", color=GREY, fontsize=10)
a2.scatter([p["ridge"]["lam"]], [p["ridge"]["cv"]], color=R, s=50, zorder=5)
a2.set_xscale("log")
a2.set(xlabel="λ（对数刻度，27 个特征）", ylabel="RMSE（mpg）", title="右：选 λ")
a2.legend(frameon=False, loc="upper left", fontsize=9.5)
fig.tight_layout()
save(fig, "F4b")


# ---------------- F7a: Lasso path ----------------
lp = N["lasso_path"]
al = np.array(lp["alphas"]); C = np.array(lp["coefs"])
fig, ax = plt.subplots(figsize=(6.8, 4.0))
for row in C:
    ax.plot(al, row, color=G, lw=1.1, alpha=0.75)
ax.axvline(p["lasso"]["lam"], color="#333", lw=1, ls="--")
ax.text(p["lasso"]["lam"] * 1.15, ax.get_ylim()[1] * 0.85, f"交叉验证选的 λ\n保留 {p['lasso']['nz']} 项", fontsize=10)
for lv in ["0.001", "0.01", "0.1", "1"]:   # counts from the same Pipeline fits as the note's table
    ax.text(float(lv), ax.get_ylim()[0] * 0.92, f"{p['lasso_nz_at'][lv]} 项", ha="center", fontsize=9, color=GREY)
ax.set_xscale("log")
ax.set(xlabel="λ（对数刻度，向右惩罚更强）", ylabel="系数（标准化后）")
save(fig, "F7a")

# ---------------- F7b: geometry, circle vs diamond ----------------
fig, axs = plt.subplots(1, 2, figsize=(9.0, 4.2))
A = np.array([[1.0, 0.55], [0.55, 0.6]]); c = np.array([1.6, 1.0])
u = np.linspace(-0.6, 2.6, 300); U, V = np.meshgrid(u, u)
Dd = np.stack([U - c[0], V - c[1]])
Q = A[0, 0] * Dd[0] ** 2 + 2 * A[0, 1] * Dd[0] * Dd[1] + A[1, 1] * Dd[1] ** 2
t = 1.0
for ax, kind in zip(axs, ["L2", "L1"]):
    if kind == "L2":
        th = np.linspace(0, 2 * np.pi, 400); grid2 = np.c_[t * np.cos(th), t * np.sin(th)]
        ax.add_patch(Circle((0, 0), t, facecolor=G, alpha=0.15, edgecolor=G, lw=2))
        title = "Ridge:  $w_1^2+w_2^2\\leq t$"
    else:
        th = np.linspace(0, 1, 400)
        grid2 = np.concatenate([np.c_[t * (1 - s), t * s] for s in [th]] + [np.c_[-t * s, t * (1 - s)] for s in [th]] +
                               [np.c_[-t * (1 - s), -t * s] for s in [th]] + [np.c_[t * s, -t * (1 - s)] for s in [th]])
        ax.add_patch(Polygon([[t, 0], [0, t], [-t, 0], [0, -t]], facecolor=G, alpha=0.15, edgecolor=G, lw=2))
        title = "Lasso:  $|w_1|+|w_2|\\leq t$"
    dq = grid2 - c
    qv = A[0, 0] * dq[:, 0] ** 2 + 2 * A[0, 1] * dq[:, 0] * dq[:, 1] + A[1, 1] * dq[:, 1] ** 2
    best = grid2[np.argmin(qv)]
    ax.contour(U, V, Q, levels=sorted({0.1, 0.4, float(qv.min()), 1.6}), colors=R, linewidths=1.3)
    ax.scatter([c[0]], [c[1]], color=R, s=30, zorder=5)
    ax.text(c[0] + 0.06, c[1] + 0.06, "不加惩罚的解", fontsize=9, color=R)
    ax.scatter([best[0]], [best[1]], color=G, s=60, zorder=6, edgecolor="white")
    ax.annotate(f"加惩罚后的解\n({best[0]:.2f}, {best[1]:.2f})", best, xytext=(-95, 30 if kind == "L2" else -55),
                textcoords="offset points", fontsize=9, color=G, arrowprops=dict(arrowstyle="-", color=G))
    ax.axhline(0, color=GREY, lw=0.6); ax.axvline(0, color=GREY, lw=0.6)
    ax.set(xlim=(-1.3, 2.6), ylim=(-1.3, 2.3), aspect="equal", xlabel="$w_1$", ylabel="$w_2$", title=title)
fig.tight_layout()
save(fig, "F7b")
# ---------------- F10: the whole chain in one picture ----------------
steps = [("模型不够强", "残差有规律、误差大", B),
         ("特征工程", "多项式、sin/cos、比率、log、交互", G),
         ("标准化", "放到同一尺度，只用训练数据算", G),
         ("太灵活：过拟合", "训练误差低、新数据误差高，系数爆炸", R),
         ("正则化", "Ridge 缩小、Lasso 置零、Elastic Net 兼顾", G),
         ("交叉验证", "选阶数与 λ，不碰测试集", B),
         ("测试集读一次", "报告最终结果和局限", B)]
BW = 3.9 if EN else 2.6                       # English labels are longer
fig, ax = plt.subplots(figsize=(10.4 if EN else 7.2, 4.6)); ax.axis("off")
ax.set_xlim(0, 10 + (BW - 2.6) * 1.6); ax.set_ylim(0, len(steps))
for k, (h, sub, col) in enumerate(steps):
    yy = len(steps) - 1 - k
    ax.add_patch(Rectangle((0.2, yy + 0.12), BW, 0.76, facecolor=col, alpha=0.14, edgecolor=col, lw=1.4))
    ax.text(0.2 + BW / 2, yy + 0.5, h, ha="center", va="center", weight="bold", color=col, fontsize=13)
    ax.text(0.5 + BW, yy + 0.5, sub, va="center", fontsize=12.5, color="#333")
    if k < len(steps) - 1:
        ax.annotate("", xy=(0.2 + BW / 2, yy - 0.12), xytext=(0.2 + BW / 2, yy + 0.12), arrowprops=dict(arrowstyle="->", color="#666", lw=1.2))
save(fig, "F10")
# ---------------- slides only: S2 (guess first) and S3 (reveal and grouped residuals) ----------------
if SLIDES:
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.scatter(hp, ytr, s=18, color=B, alpha=0.45, lw=0, label="313 辆训练车")
    ax.plot(grid, gl, color=G, lw=2.6, label="马力直线")
    ax.axvline(car["hp"], color="#333", lw=1.4, ls="--")
    ax.text(car["hp"] - 3, 44, f"{car['hp']:.0f} hp", ha="right", fontsize=15, weight="bold")
    ax.set(xlabel="马力 hp", ylabel="mpg", xlim=(40, 240), ylim=(0, 48))
    ax.legend(frameon=False, loc="upper right")
    save(fig, "S2")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.0, 5.0), gridspec_kw=dict(width_ratios=[1.2, 1]))
    a1.scatter(hp, ytr, s=18, color=B, alpha=0.45, lw=0)
    a1.plot(grid, gl, color=G, lw=2.6)
    mark_car(a1, car["pred_line"])
    a1.annotate(f"实测 {car['mpg']:.1f}", (car["hp"], car["mpg"]), xytext=(-110, 30), textcoords="offset points",
                color=B, weight="bold", arrowprops=dict(arrowstyle="-", color=B, lw=0.8))
    a1.annotate(f"直线预测 {car['pred_line']:.1f}", (car["hp"], car["pred_line"]), xytext=(-190, -4), textcoords="offset points",
                color=G, weight="bold", arrowprops=dict(arrowstyle="-", color=G, lw=0.8))
    a1.set(xlabel="马力 hp", ylabel="mpg", xlim=(40, 240), ylim=(0, 48))
    g = N["resid_groups"]
    a2.axhline(0, color=GREY, lw=0.8)
    a2.bar(range(5), g["line"], color=[R if v > 0 else "white" for v in g["line"]], edgecolor=R, lw=2)
    for k, v in enumerate(g["line"]):
        a2.text(k, v + (0.25 if v > 0 else -0.45), f"{v:+.1f}", ha="center", fontsize=14, color=R)
    a2.set_xticks(range(5)); a2.set_xticklabels([f"{h:.0f}" for h in g["hp_mid"]])
    a2.set(xlabel="马力 hp（五组，各组中位数）", ylabel="平均残差（mpg）", ylim=(-3.2, 4.2))
    fig.tight_layout()
    save(fig, "S3")

    # S5: the straight line and the curve after adding hp^2 (one message: a new column bends the line)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.scatter(hp, ytr, s=18, color=B, alpha=0.45, lw=0)
    ax.plot(grid, gl, color=G, lw=2.2, ls="--", label=f"hp only: test RMSE {N['line']['test']:.2f}")
    ax.plot(grid, gq, color=G, lw=3, label=f"hp and hp²: test RMSE {N['quad']['test']:.2f}")
    ax.set(xlabel="Horsepower (hp)", ylabel="mpg", xlim=(40, 240), ylim=(0, 48))
    ax.legend(frameon=False, loc="upper right")
    save(fig, "S5")

    # S8: a ratio reveals a relation a weighted sum cannot form
    from sklearn.datasets import fetch_california_housing
    ca = fetch_california_housing(as_frame=True).frame
    ca = ca[(ca.AveRooms < 10) & (ca.AveBedrms < 2)]
    share = ca.AveBedrms / ca.AveRooms
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=True)
    for ax, x, lab, cr in [(a1, ca.AveBedrms, "Bedrooms per household", N["calif"]["corr_bedrooms"]),
                           (a2, share, "Bedrooms ÷ rooms", N["calif"]["corr_ratio"])]:
        q = pd.qcut(x, 10)
        m = ca.MedHouseVal.groupby(q, observed=True).mean()
        mid = x.groupby(q, observed=True).median()
        ax.plot(mid.values, m.values, color=B, lw=2.6, marker="o", ms=8)
        ax.set(xlabel=lab, title=f"correlation with price {cr:+.2f}".replace("-", "−"))
    a1.set_ylabel("Mean house value\n(100k dollars, ten groups)")
    fig.tight_layout()
    save(fig, "S8")

    # S9: a quantity spanning orders of magnitude spreads out after a log
    pop = fetch_california_housing(as_frame=True).frame.Population
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.0, 4.4))
    a1.hist(pop, bins=60, color=B, alpha=0.8)
    a1.set(xlabel="Population of a district", ylabel="Number of districts", title="Raw: squeezed to the left")
    a2.hist(np.log10(pop), bins=60, color=B, alpha=0.8)
    a2.set(xlabel="log10(population)", title="After log: spread out")
    fig.tight_layout()
    save(fig, "S9")

    # S10: polynomial features as the general recipe
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.scatter(hp, ytr, s=18, color=B, alpha=0.35, lw=0)
    for d, ls, lab in [(1, ":", "x"), (2, "--", "x, x²"), (5, "-", "x, x², x³, x⁴, x⁵")]:
        m = make_pipeline(PolynomialFeatures(d, include_bias=False), StandardScaler(), LinearRegression()).fit(Xtr[["horsepower"]], ytr)
        ax.plot(grid, m.predict(pd.DataFrame({"horsepower": grid})), color=G, lw=2.6, ls=ls, label=f"columns: {lab}")
    ax.set(xlabel="Horsepower (hp)", ylabel="mpg", xlim=(40, 240), ylim=(0, 48))
    ax.legend(frameon=False, loc="upper right")
    save(fig, "S10")

    # S11: three columns on wildly different scales, then standardised
    cols = {"hp": hp, "hp²": hp ** 2, "hp³": hp ** 3}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.0, 4.2))
    rng = np.random.default_rng(0)
    for k, (name, v) in enumerate(cols.items()):
        jit = k + rng.uniform(-0.18, 0.18, len(v))
        a1.scatter(v, jit, s=10, color=B, alpha=0.45, lw=0)
        a2.scatter((v - v.mean()) / v.std(), jit, s=10, color=B, alpha=0.45, lw=0)
    for ax in (a1, a2):
        ax.set_yticks(range(3)); ax.set_yticklabels(list(cols)); ax.set_ylim(-0.6, 2.6)
    a1.set(title="Raw values", xlabel="value")
    a2.set(title="Standardised: z = (x − mean) / SD", xlabel="z")
    fig.tight_layout()
    save(fig, "S11")

    # S13: underfitting and overfitting on the same ten cars
    W = N["widget"]; rows = {q["d"]: q for q in N["ten"]["degrees"]}
    fig, axs = plt.subplots(1, 3, figsize=(14.0, 4.6), sharey=True)
    for ax, d, lab in zip(axs, [1, 3, 7], ["Degree 1: underfitting", "Degree 3", "Degree 7: overfitting"]):
        ax.scatter(W["test_hp"], W["test_mpg"], s=16, facecolor="none", edgecolor="#999", lw=1)
        ax.plot(W["grid"], np.clip(W["deg_curves"][str(d)], -50, 100), color=G, lw=2.6)
        ax.scatter(N["ten"]["hp"], N["ten"]["mpg"], s=60, color=B, zorder=5)
        te = rows[d]["test"]
        ax.set(title=lab, xlabel="Horsepower (hp)", xlim=(40, 235), ylim=(0, 50))
        ax.text(0.03, 0.04, f"train {rows[d]['train']:.2f}\ntest {te:.2f}" if te < 100 else f"train {rows[d]['train']:.2f}\ntest {te:,.0f}",
                transform=ax.transAxes, fontsize=14, color=R)
    axs[0].set_ylabel("mpg")
    fig.tight_layout()
    save(fig, "S13")

    # S19: training error always prefers more; new cars unknown
    cvd = N["cv_degree"]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot([q["d"] for q in cvd], [q["train"] for q in cvd], color=R, lw=2.6, ls="--", marker="o", mfc="white", label="training RMSE")
    ax.text(6.0, 4.62, "new cars:  ?", fontsize=22, color=R, weight="bold")
    ax.set(xlabel="Polynomial degree (horsepower, 313 training cars)", ylabel="RMSE (mpg)", xticks=range(1, 11), ylim=(4.1, 5.05))
    ax.legend(frameon=False, loc="upper right")
    save(fig, "S19")

    # S6: the three discussion situations, raw data only (no answers)
    bike = pd.read_csv(os.path.join(HERE, "bike_hourly.csv"))
    hm = bike.groupby("hour")["count"].mean()
    ca6 = fetch_california_housing(as_frame=True).frame
    ca6 = ca6[(ca6.AveRooms < 10) & (ca6.AveBedrms < 2)]
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.4))
    axs[0].scatter(hm.index, hm.values, s=40, color=B)
    axs[0].set(xlabel="Hour of the day", ylabel="Rentals per hour", title="1  Bike rentals", xticks=range(0, 24, 6))
    q6 = pd.qcut(ca6.AveBedrms, 10)
    axs[1].plot(ca6.AveBedrms.groupby(q6, observed=True).median().values, ca6.MedHouseVal.groupby(q6, observed=True).mean().values,
                color=B, lw=2.6, marker="o", ms=8)
    axs[1].set(xlabel="Bedrooms per household", ylabel="Mean house value (100k $)", title="2  House prices", ylim=(0, 4))
    axs[2].hist(fetch_california_housing(as_frame=True).frame.Population, bins=60, color=B, alpha=0.8)
    axs[2].set(xlabel="Population of a district", ylabel="Number of districts", title="3  District population")
    fig.tight_layout()
    save(fig, "S6")


    # S4: the curved data and the straight line, for the "how can it bend?" question
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.scatter(hp, ytr, s=18, color=B, alpha=0.45, lw=0)
    ax.plot(grid, gl, color=G, lw=2.6)
    ax.set(xlabel="Horsepower (hp)", ylabel="mpg", xlim=(40, 240), ylim=(0, 48))
    save(fig, "S4")

    # S7: the three discussion answers on one page, one panel each
    bk = N["bike"]; hrs = np.arange(24)
    fig, axs = plt.subplots(1, 3, figsize=(15.0, 4.6))
    axs[0].scatter(hrs, bk["hourly_mean"], s=36, color=B)
    axs[0].plot(hrs, bk["fits"]["raw"], color=G, lw=2, ls="--", label=f"hour: RMSE {bk['r2']['raw_rmse']:.0f}")
    axs[0].plot(hrs, bk["fits"]["sincos2"], color=G, lw=2.6, label=f"sin, cos: RMSE {bk['r2']['sincos2_rmse']:.0f}")
    axs[0].set(xlabel="Hour of the day", ylabel="Rentals per hour", title="1  Hour → sin, cos", xticks=range(0, 24, 6))
    axs[0].legend(frameon=False, loc="upper left", fontsize=12)
    ca7 = fetch_california_housing(as_frame=True).frame
    ca7 = ca7[(ca7.AveRooms < 10) & (ca7.AveBedrms < 2)]
    sh = ca7.AveBedrms / ca7.AveRooms
    q7 = pd.qcut(sh, 10)
    axs[1].plot(sh.groupby(q7, observed=True).median().values, ca7.MedHouseVal.groupby(q7, observed=True).mean().values,
                color=B, lw=2.6, marker="o", ms=8)
    axs[1].set(xlabel="Bedrooms ÷ rooms", ylabel="Mean house value (100k $)", title="2  Bedrooms ÷ rooms", ylim=(0, 4))
    axs[2].hist(np.log10(fetch_california_housing(as_frame=True).frame.Population), bins=60, color=B, alpha=0.8)
    axs[2].set(xlabel="log10(population)", ylabel="Number of districts", title="3  log(population)")
    fig.tight_layout()
    save(fig, "S7")

    # S21: cross-validation chooses the degree
    cvd = N["cv_degree"]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.errorbar([q["d"] for q in cvd], [q["cv"] for q in cvd], yerr=[q["se"] for q in cvd], color=R, lw=2.6, marker="o", capsize=3,
                label="cross-validation RMSE (± one standard error)")
    ax.plot([q["d"] for q in cvd], [q["train"] for q in cvd], color=R, lw=2, ls="--", marker="o", mfc="white", label="training RMSE")
    ch = N["cv_degree_choice"]["one_se_d"]
    ax.annotate(f"choose degree {ch}", (ch, cvd[ch - 1]["cv"]), xytext=(30, 60), textcoords="offset points", weight="bold",
                arrowprops=dict(arrowstyle="->", color="#333"))
    ax.set(xlabel="Polynomial degree (horsepower, 313 training cars)", ylabel="RMSE (mpg)", xticks=range(1, 11))
    ax.legend(frameon=False, loc="upper right")
    save(fig, "S21")

print("figures ok")
