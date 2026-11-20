"""Lesson 7 Note V2: static figures (Chinese). Reads numbers_v2.json written by build_numbers_v2.py.
Run (from this folder):  python build_figures_v2.py      (PNGDIR=... also writes PNG copies for checking by eye)
NOTE_LANG=en writes the English version (F3-F14) to ../note_en_v2/; without it, the Chinese version here, unchanged.
Colours (fixed across the course notes): data blue, model green, error red.
F1 只报身高与沿长带量一个数   F2 勾股：散布 + 误差 = 固定   F3 投影方差随角度
F4 环                          F5 汽车：权重与油耗           F6 一张图变成一行数
F7 平均图与前 6 个主成分       F8 累计比例                   F9 两个主成分上的数字
F10 一页回顾
V7：F11 线上的哪一点   F12 三种落到线上的办法   F13 两个特征向量   F14 三维点挤在平面附近（F1、F2 不再用于正文）
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
N = json.load(open(os.path.join(HERE, "numbers_v2.json")))
EN = os.environ.get("NOTE_LANG") == "en"   # NOTE_LANG=en: English labels, written to ../note_en_v2/
OUT = os.path.join(os.path.dirname(HERE), "note_en_v2") if EN else HERE
os.makedirs(OUT, exist_ok=True)


def T(zh, en):
    return en if EN else zh


B, G, R, O = "#0072B2", "#00795A", "#C84B00", "#E69F00"
GREY, LIGHT = "#6b6b6b", "#d9d9d9"
plt.rcParams.update({
    "font.family": T(["Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"], ["Arial", "DejaVu Sans"]),
    "svg.fonttype": "path", "axes.unicode_minus": False, "mathtext.fontset": "dejavusans",
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11, "figure.dpi": 100,
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.svg"), bbox_inches="tight")
    if os.environ.get("PNGDIR"):
        fig.savefig(os.path.join(os.environ["PNGDIR"], f"{name}.png"), bbox_inches="tight", dpi=90)
    plt.close(fig)


def pct(v):
    return f"{100 * v:.0f}%"


A = N["arm"]
h, a = np.array(A["h"]), np.array(A["a"])
m = np.array([h.mean(), a.mean()])
u1 = np.array(A["u1"])


def cloud(ax, lim=24):
    ax.set_xlim(m[0] - lim, m[0] + lim)
    ax.set_ylim(m[1] - lim, m[1] + lim)
    ax.set_aspect("equal")
    ax.set_xlabel(T("身高（cm）", "Height (cm)"))
    ax.set_ylabel(T("臂展（cm）", "Arm span (cm)"))


if not EN:  # F1, F2 are not used in the Note body, so there is no English version
    # ---------------- F1: report height only, or one number along the band ----------------
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.9))
    ax = axs[0]
    cloud(ax)
    ax.axhline(m[1], color=G, lw=2.5)
    for x, y in zip(h, a):
        ax.plot([x, x], [y, m[1]], color=LIGHT, lw=0.7, zorder=1)
    ax.scatter(h, a, s=12, color=B, alpha=0.75, zorder=2)
    ax.set_title(f"只报身高：留住 {pct(A['share_height'])} 的散布")
    ax = axs[1]
    cloud(ax)
    L = 30
    ax.plot([m[0] - L * u1[0], m[0] + L * u1[0]], [m[1] - L * u1[1], m[1] + L * u1[1]], color=G, lw=2.5)
    for x, y in zip(h, a):
        z = (np.array([x, y]) - m) @ u1
        p = m + z * u1
        ax.plot([x, p[0]], [y, p[1]], color=LIGHT, lw=0.7, zorder=1)
    ax.scatter(h, a, s=12, color=B, alpha=0.75, zorder=2)
    ax.set_title(f"沿长带量一个数：留住 {pct(A['share'][0])}")
    ax.annotate("“个子多大”", xy=(m[0] + 17 * u1[0], m[1] + 17 * u1[1]), xytext=(m[0] + 3, m[1] + 19), color=G, fontsize=11,
                arrowprops=dict(arrowstyle="->", color=G))
    save(fig, "F1")

    # ---------------- F2: Pythagoras, one point ----------------
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ang = np.radians(28)
    u = np.array([np.cos(ang), np.sin(ang)])
    P = np.array([2.2, 3.1])
    z = P @ u
    F = z * u
    ax.plot([-1.5 * u[0], 5.2 * u[0]], [-1.5 * u[1], 5.2 * u[1]], color=G, lw=2.5)
    ax.plot([0, P[0]], [0, P[1]], color=GREY, lw=1.8)
    ax.plot([0, F[0]], [0, F[1]], color=B, lw=4, solid_capstyle="butt")
    ax.plot([P[0], F[0]], [P[1], F[1]], color=R, lw=3)
    v = (P - F) / np.linalg.norm(P - F)
    s = 0.22
    ax.plot([F[0] + s * v[0], F[0] + s * v[0] - s * u[0], F[0] - s * u[0]], [F[1] + s * v[1], F[1] + s * v[1] - s * u[1], F[1] - s * u[1]], color="k", lw=0.8)
    ax.scatter([0], [0], s=50, color="k", zorder=3)
    ax.scatter([P[0]], [P[1]], s=60, color=B, zorder=3)
    ax.text(-0.25, -0.45, "中心", fontsize=11)
    ax.text(P[0] + 0.12, P[1] + 0.1, "一个点", fontsize=11, color=B)
    ax.text(0.15 * P[0] - 0.9, 0.55 * P[1], "到中心的距离\n（固定）", fontsize=10.5, color=GREY)
    ax.text(F[0] * 0.55 + 0.1, F[1] * 0.55 - 0.55, "线上的一段：散布", fontsize=10.5, color=B)
    ax.text((P[0] + F[0]) / 2 + 0.15, (P[1] + F[1]) / 2, "到线的距离：\n还原误差", fontsize=10.5, color=R)
    ax.set_xlim(-1.6, 5.2)
    ax.set_ylim(-1.0, 3.8)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "F2")


# ---------------- F3: variance of the projection against the angle ----------------
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ang = np.array(A["angles"])
ax.plot(ang, A["spread"], color=B, lw=2.2, label=T("投影的方差 uᵀSu", "Variance of the projection uᵀSu"))
ax.plot(ang, A["error"], color=R, lw=1.6, ls="--", label=T("还原误差", "Reconstruction error"))
ax.axvline(A["theta"], color=G, lw=1.2)
ax.axvline(A["theta_min"], color=GREY, lw=1.0, ls=":")
ax.text(A["theta"] + 3, 52, T(f"最大：{A['theta']}°\nλ₁ = {A['lam'][0]}", f"Largest: {A['theta']}°\nλ₁ = {A['lam'][0]}"), color=G, fontsize=10.5, va="center")
ax.text(A["theta_min"] - 3, 52, T(f"最小：{A['theta_min']}°\nλ₂ = {A['lam'][1]}", f"Smallest: {A['theta_min']}°\nλ₂ = {A['lam'][1]}"), color=GREY, fontsize=10.5, va="center", ha="right")
ax.set_xlabel(T("方向 θ（度）", "Direction θ (degrees)"))
ax.set_ylabel("cm²")
ax.set_xlim(0, 179)
ax.set_ylim(0, 105)
ax.set_xticks([0, 30, 46.4, 90, 136.4, 179])
ax.set_xticklabels(["0", "30", "46.4", "90", "136.4", "179"])
ax.legend(loc="lower center", frameon=False, fontsize=10)
save(fig, "F3")

# ---------------- F4: a ring ----------------
Rg = N["ring"]
fig, ax = plt.subplots(figsize=(4.4, 4.4))
ax.scatter(Rg["x"], Rg["y"], s=12, color=B, alpha=0.75)
w = np.array(Rg["u1"])
w2 = np.array([-w[1], w[0]])
for vec, col, sh in [(w, G, Rg["share"][0]), (w2, O, Rg["share"][1])]:
    ax.plot([-7 * vec[0], 7 * vec[0]], [-7 * vec[1], 7 * vec[1]], color=col, lw=2.2)
ax.set_title(T(f"两个方向：{pct(Rg['share'][0])}、{pct(Rg['share'][1])}", f"Two directions: {pct(Rg['share'][0])}, {pct(Rg['share'][1])}"))
ax.set_aspect("equal")
ax.set_xlim(-7.5, 7.5)
ax.set_ylim(-7.5, 7.5)
ax.set_xticks([])
ax.set_yticks([])
save(fig, "F4")

# ---------------- F5: cars ----------------
Cr = N["cars"]
names = T(["缸数", "排量", "马力", "车重", "加速时间"], ["Cylinders", "Displacement", "Horsepower", "Car weight", "Acceleration time"])
fig, axs = plt.subplots(1, 3, figsize=(12.4, 3.8), gridspec_kw=dict(width_ratios=[1, 1, 1.15], wspace=T(0.42, 0.62)))
for ax, w, title, col in [(axs[0], Cr["raw_w1"], T(f"不标准化：第一主成分（{100 * Cr['raw_share'][0]:.1f}%）", f"Not standardised:\nfirst principal component ({100 * Cr['raw_share'][0]:.1f}%)"), R),
                          (axs[1], Cr["w1"], T(f"先标准化：第一主成分（{pct(Cr['share'][0])}）", f"Standardised first:\nfirst principal component ({pct(Cr['share'][0])})"), G)]:
    y = np.arange(5)[::-1]
    ax.barh(y, w, color=col, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_xlim(-0.75, 1.15)
    ax.set_title(title, fontsize=11.5)
    ax.set_xlabel(T("权重", "Weight"))
    for yy, ww in zip(y, w):
        ax.text(ww + (0.03 if ww >= 0 else -0.03), yy, f"{ww:.2f}" if abs(ww) >= 0.01 else f"{ww:.3f}", va="center",
                ha="left" if ww >= 0 else "right", fontsize=9.5)
ax = axs[2]
ax.scatter(Cr["pc1"], Cr["mpg"], s=10, color=B, alpha=0.6)
ax.set_xlabel(T("第一主成分的得分（“车有多大”）", "Score on the first principal component\n(“how big the car is”)"))
ax.set_ylabel(T("油耗（英里每加仑）", "Fuel economy (miles per gallon)"))
ax.set_title(T(f"PCA 没看油耗，相关 {Cr['corr_pc1_mpg']}", f"PCA never saw fuel economy; correlation {Cr['corr_pc1_mpg']}"), fontsize=11.5)
save(fig, "F5")

# ---------------- F6: one image becomes one row of 784 numbers ----------------
D = N["digits"]
img = np.array(D["original"]).reshape(28, 28)
fig = plt.figure(figsize=(11.0, 3.2))
ax = fig.add_axes([0.0, 0.08, 0.22, 0.84])
ax.imshow(img, cmap="Greys", vmin=0, vmax=1)
ROWS = [7, 13, 19]  # 0-based rows with ink
for rr, col in zip(ROWS, [B, G, R]):
    ax.add_patch(plt.Rectangle((-0.5, rr - 0.5), 28, 1, fill=False, ec=col, lw=1.6))
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(T("一张 28×28 的图", "One 28×28 image"))
for s_ in ax.spines.values():
    s_.set_visible(False)
ax2 = fig.add_axes([0.28, 0.42, 0.71, 0.18])
strip = np.r_[img[ROWS[0]], np.full(4, np.nan), img[ROWS[1]], np.full(4, np.nan), img[ROWS[2]]][None, :]
ax2.imshow(strip, cmap="Greys", vmin=0, vmax=1, aspect="auto")
for x0, col in [(0, B), (32, G), (64, R)]:
    ax2.add_patch(plt.Rectangle((x0 - 0.5, -0.5), 28, 1, fill=False, ec=col, lw=1.6))
ax2.set_xticks([13.5, 29.5, 45.5, 61.5, 77.5])
ax2.set_xticklabels([T(f"第 {q + 1} 行（第 {28 * q + 1}–{28 * q + 28} 个数）", f"Row {q + 1} (numbers {28 * q + 1}–{28 * q + 28})") if i % 2 == 0 else "…" for i, q in enumerate([ROWS[0], 0, ROWS[1], 0, ROWS[2]])], fontsize=9.5)
ax2.set_yticks([])
for s_ in ax2.spines.values():
    s_.set_visible(False)
fig.text(0.635, 0.82, T("一行一行接起来：一张图 = 784 个数 = 表里的一行", "Joined row after row: one image = 784 numbers = one row of the table"), ha="center", fontsize=12)
fig.text(0.635, 0.12, T("每张图一行、每个像素一列：和汽车那张表一样，只是 5 列变成 784 列", "One row per image, one column per pixel: like the car table, only with 784 columns instead of 5"), ha="center", fontsize=11, color=GREY)
save(fig, "F6")

# ---------------- F7: the mean image and the first six components ----------------
fig, axs = plt.subplots(1, 7, figsize=(12.4, 2.4))
axs[0].imshow(np.array(D["mean"]).reshape(28, 28), cmap="Greys", vmin=0, vmax=1)
axs[0].set_title(T("平均图 x̄", "Mean image x̄"), fontsize=11)
lim = max(np.abs(np.array(D["comps6"])).max(), 1e-6)
for j in range(6):
    axs[j + 1].imshow(np.array(D["comps6"][j]).reshape(28, 28), cmap="RdBu_r", vmin=-lim, vmax=lim)
    axs[j + 1].set_title(T(f"u{'₁₂₃₄₅₆'[j]}：{100 * D['evr'][j]:.1f}%", f"u{'₁₂₃₄₅₆'[j]}: {100 * D['evr'][j]:.1f}%"), fontsize=11)
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)
save(fig, "F7")

# ---------------- F8: cumulative share ----------------
cum = np.array(D["cum_all"])
fig, ax = plt.subplots(figsize=(6.8, 3.6))
k = np.arange(1, len(cum) + 1)
ax.plot(k, cum, color=B, lw=2.2)
for kk, txt in [(10, None), (50, None), (D["k90"], None)]:
    ax.scatter([kk], [cum[kk - 1]], color=G, s=28, zorder=3)
    ax.annotate(T(f"K = {kk}：{100 * cum[kk - 1]:.1f}%", f"K = {kk}: {100 * cum[kk - 1]:.1f}%"), xy=(kk, cum[kk - 1]), xytext=(kk + 14, cum[kk - 1] - 0.09), fontsize=10)
ax.axhline(0.9, color=GREY, lw=0.8, ls=":")
ax.set_xlabel(T("主成分个数 K", "Number of principal components K"))
ax.set_ylabel(T("累计比例", "Cumulative share"))
ax.set_xlim(0, 300)
ax.set_ylim(0, 1)
save(fig, "F8")

# ---------------- F9: the digits on two components ----------------
x, y, lab = np.array(D["pc2_x"]), np.array(D["pc2_y"]), np.array(D["pc2_label"])
cols = plt.cm.tab10(np.arange(10))
fig, ax = plt.subplots(figsize=(6.6, 5.4))
order = np.random.default_rng(0).permutation(len(x))
ax.scatter(x[order], y[order], s=4, c=cols[lab[order]], alpha=0.55, linewidths=0)
for d in range(10):
    ax.text(np.median(x[lab == d]), np.median(y[lab == d]), str(d), fontsize=15, fontweight="bold", ha="center", va="center",
            color=cols[d], bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=cols[d], alpha=0.9))
ax.set_xlabel(T("第一主成分的得分 z₁", "Score on the first principal component z₁"))
ax.set_ylabel(T("第二主成分的得分 z₂", "Score on the second principal component z₂"))
ax.set_title(T(f"{D['fit']} 张训练图，两个数留住 {pct(D['pc2_share'])}（颜色只为了看，PCA 没用标签）", f"{D['fit']} training images: two numbers keep {pct(D['pc2_share'])}\n(colour is only for viewing; PCA did not use the labels)"), fontsize=11)
save(fig, "F9")

# ---------------- F10: one-page recap (vertical flow chart, same style as the other lessons) ----------------
from matplotlib.patches import Rectangle
steps = T([("少存几个数", "车的前四个量相关 0.84–0.95；一张图 784 个数，很多在重复", B),
           ("画线就是投影", "转线：46° 时散得最开、丢得最少，两根柱子之和始终 97.68", G),
           ("投影的性质", "uᵀu = 1；z = uᵀx；丢掉的部分和线垂直；多个方向是叠加", G),
           ("还原误差", "Σ‖x − zu‖² = (n − 1)(tr S − uᵀSu)：误差最小 = 方差最大", G),
           ("特征向量", "Su = λu：46.4°，λ = 94.71、2.96；前 K 个，误差 = 丢掉的 λ 之和", G),
           ("什么时候不行", "环：55%、45%；单位不同先标准化（车重独占 99.8%）", R),
           ("两个应用", "车：“车有多大” 81%，和油耗相关 −0.82；数字：K = 50 留住 82.7%", B)],
          [("Store fewer numbers", "The first four car quantities correlate 0.84–0.95; an image is 784 numbers, many repeat", B),
           ("A line is a projection", "Turn the line: at 46° the spread is widest and the loss smallest; the two bars always sum to 97.68", G),
           ("Properties of the projection", "uᵀu = 1; z = uᵀx; the lost part is at right angles to the line; several directions add up", G),
           ("Reconstruction error", "Σ‖x − zu‖² = (n − 1)(tr S − uᵀSu): least error = largest variance", G),
           ("Eigenvectors", "Su = λu: 46.4°, λ = 94.71 and 2.96; take the top K, error = sum of the dropped λ", G),
           ("When it fails", "Ring: 55% and 45%; standardise when units differ (car weight alone takes 99.8%)", R),
           ("Two applications", "Cars: \"how big the car is\" 81%, correlation with fuel economy −0.82; digits: K = 50 keeps 82.7%", B)])
fig, ax = plt.subplots(figsize=(T(10.4, 13.4), 4.8)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, len(steps))
heads = [ax.text(0, 0, hd, ha="center", va="center", weight="bold", color=c, fontsize=13) for hd, _, c in steps]
fig.canvas.draw(); inv_t = ax.transData.inverted()
BW = max(np.ptp(inv_t.transform(t.get_window_extent())[:, 0]) for t in heads) + 0.5
for k, (hd, sub, col) in enumerate(steps):
    yy = len(steps) - 1 - k
    ax.add_patch(Rectangle((0.2, yy + 0.12), BW, 0.76, facecolor=col, alpha=0.14, edgecolor=col, lw=1.4))
    heads[k].set_position((0.2 + BW / 2, yy + 0.5))
    ax.text(0.5 + BW, yy + 0.5, sub, va="center", fontsize=12, color="#333")
    if k < len(steps) - 1:
        ax.annotate("", xy=(0.2 + BW / 2, yy - 0.12), xytext=(0.2 + BW / 2, yy + 0.12),
                    arrowprops=dict(arrowstyle="->", color="#666", lw=1.2))
save(fig, "F10")

# ---------------- F11: which point on the line? candidates and the foot ----------------
fig, ax = plt.subplots(figsize=(5.8, 4.4))
ang = np.radians(28)
u = np.array([np.cos(ang), np.sin(ang)])
P = np.array([2.2, 3.1])
zs = P @ u
ax.plot([-1.5 * u[0], 5.4 * u[0]], [-1.5 * u[1], 5.4 * u[1]], color=G, lw=2.5)
ax.annotate("", xy=u * 1.0, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="k", lw=1.6))
ax.text(u[0] * 0.55 + 0.05, u[1] * 0.55 - 0.42, T("u（长度 1）", "u (length 1)"), fontsize=13)
for zc in (1.6, 4.9):
    C = zc * u
    ax.plot([P[0], C[0]], [P[1], C[1]], color=GREY, lw=1.2, ls="--")
    ax.scatter([C[0]], [C[1]], s=40, color=GREY, zorder=3)
    ax.text(C[0] + 0.05, C[1] - 0.42, f"z = {zc}", fontsize=12.5, color=GREY)
F = zs * u
ax.plot([P[0], F[0]], [P[1], F[1]], color=R, lw=2.6)
v = (P - F) / np.linalg.norm(P - F)
q = 0.2
ax.plot([F[0] + q * v[0], F[0] + q * v[0] + q * u[0], F[0] + q * u[0]], [F[1] + q * v[1], F[1] + q * v[1] + q * u[1], F[1] + q * u[1]], color="k", lw=0.8)
ax.scatter([F[0]], [F[1]], s=60, color=G, zorder=4)
ax.text(F[0] + 0.3, F[1] - 0.62, T(f"最好的一点\nz = uᵀx = {zs:.1f}", f"The best point\nz = uᵀx = {zs:.1f}"), fontsize=13, color=G)
ax.scatter([0], [0], s=40, color="k", zorder=3)
ax.text(-0.35, -0.45, T("原点", "Origin"), fontsize=13)
ax.scatter([P[0]], [P[1]], s=80, color=B, zorder=4)
ax.text(P[0] + 0.12, P[1] + 0.08, T("一个点 x", "A point x"), fontsize=13.5, color=B)
ax.set_xlim(-1.6, 5.4)
ax.set_ylim(-1.0, 3.8)
ax.set_aspect("equal")
ax.axis("off")
save(fig, "F11")

# ---------------- F12: three ways to drop a point onto a line ----------------
idx = np.random.default_rng(3).choice(len(h), 45, replace=False)
fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.4))
lines = [(T("竖直落下：用身高拟合臂展", "Drop vertically: fit arm span from height"), A["reg_h2a"], "v"), (T("水平移过去：用臂展拟合身高", "Move across: fit height from arm span"), A["reg_a2h"], "h"), (T("垂直落下：离点最近", "Drop at a right angle: closest to the point"), A["theta"], "p")]
for ax, (t, deg, kind) in zip(axs, lines):
    cloud(ax, lim=20)
    d = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
    ax.plot([m[0] - 30 * d[0], m[0] + 30 * d[0]], [m[1] - 30 * d[1], m[1] + 30 * d[1]], color=G, lw=2.3)
    for i in idx:
        p0 = np.array([h[i], a[i]]) - m
        if kind == "v":
            f = np.array([p0[0], p0[0] * d[1] / d[0]])
        elif kind == "h":
            f = np.array([p0[1] * d[0] / d[1], p0[1]])
        else:
            f = (p0 @ d) * d
        ax.plot([p0[0] + m[0], f[0] + m[0]], [p0[1] + m[1], f[1] + m[1]], color=R, lw=0.9)
    ax.scatter(h[idx], a[idx], s=14, color=B, zorder=3)
    ax.set_title(T(f"{t}\n线的角度 {deg:.1f}°", f"{t}\nangle of the line {deg:.1f}°"), fontsize=11)
save(fig, "F12")

# ---------------- F13: the two eigenvectors on the cloud ----------------
fig, ax = plt.subplots(figsize=(5.4, 5.0))
cloud(ax)
ax.scatter(h, a, s=12, color=B, alpha=0.6)
u2 = np.array(A["u2"])
for vec, lam, sh, col, name in ((u1, A["lam"][0], A["share"][0], G, T("第一个方向", "First direction")), (u2, A["lam"][1], A["share"][1], O, T("第二个方向", "Second direction"))):
    L = 2.2 * np.sqrt(lam) + 3
    ax.annotate("", xy=m + L * vec, xytext=m, arrowprops=dict(arrowstyle="-|>", color=col, lw=2.4))
    off = np.array(T([0.6, -1.6], [-13.5, 1.0])) if col == G else np.array([-10.5, 1.5])
    ax.text(*(m + L * vec + off), T(f"{name}\nλ = {lam:.2f}，{pct(sh)}", f"{name}\nλ = {lam:.2f}, {pct(sh)}"), fontsize=10.5, color=col, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
save(fig, "F13")

# ---------------- F14: a 3-D cloud near a plane ----------------
P3 = np.array(N["plane3d"]["x"])
V3 = np.array(N["plane3d"]["v"])
sh3 = N["plane3d"]["share"]
# show the cloud in a frame where the plane is tilted 25 degrees from the floor, so the plane and the red segments are visible
tq = np.radians(25)
Rt = np.array([[1, 0, 0], [0, np.cos(tq), -np.sin(tq)], [0, np.sin(tq), np.cos(tq)]])
Y = (P3 @ V3.T) @ Rt.T
E = np.eye(3) @ Rt.T
Fy = ((P3 @ V3[:2].T) @ np.eye(3)[:2]) @ Rt.T
fig = plt.figure(figsize=(6.6, 5.4))
ax = fig.add_subplot(projection="3d")
gx, gy = np.meshgrid(np.linspace(-11, 11, 2), np.linspace(-6, 6, 2))
pl = gx[..., None] * E[0] + gy[..., None] * E[1]
ax.plot_surface(pl[..., 0], pl[..., 1], pl[..., 2], color=G, alpha=0.18, linewidth=0)
for p0, f in zip(Y, Fy):
    ax.plot([p0[0], f[0]], [p0[1], f[1]], [p0[2], f[2]], color=R, lw=0.8)
ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], s=9, color=B, depthshade=False)
ax.quiver(0, 0, 0, *(E[0] * 9), color=G, lw=2.6, arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, *(E[1] * 5.5), color=O, lw=2.6, arrow_length_ratio=0.15)
ax.set_title(T(f"三个量，点挤在一个平面附近\n第一个方向 {pct(sh3[0])}（绿），第二个 {pct(sh3[1])}（橙），垂直于平面的只占 {pct(sh3[2])}（红线）", f"Three quantities; the points crowd near a plane\nfirst direction {pct(sh3[0])} (green), second {pct(sh3[1])} (orange),\nperpendicular to the plane only {pct(sh3[2])} (red lines)"), fontsize=10.5)
ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
ax.set_xlim(-11, 11); ax.set_ylim(-8, 8); ax.set_zlim(-6, 6)
ax.view_init(elev=22, azim=-58)
save(fig, "F14")
print("figures ok")
