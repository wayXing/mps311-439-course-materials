"""Lesson 7 Note: static figures. Reads numbers.json written by build_numbers.py.
Run (from this folder):  python build_note_figures.py            -> Chinese figures here
                         python build_note_figures.py --lang en  -> English figures in ../note_en/
                         python build_note_figures.py --slides   -> English, projector-size text in ../slides/
Colours (fixed across the course notes): data blue, model green, error red.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
SLIDES = "--slides" in os.sys.argv
EN = SLIDES or "en" in " ".join(os.sys.argv[1:]).split()  # also accepts "--lang en" passed as one word
OUTDIR = os.path.join(HERE, "..", "slides" if SLIDES else "note_en") if EN else HERE
os.makedirs(OUTDIR, exist_ok=True)
if EN:  # every visible string goes through the translation table; leftover Chinese stops the build
    from i18n_en import tr, check
    _set_text = matplotlib.text.Text.set_text
    matplotlib.text.Text.set_text = lambda self, s: _set_text(self, check(tr(str(s)), "figure text") if s is not None else s)
N = json.load(open(os.path.join(HERE, "numbers.json")))
B, G, R = "#0072B2", "#00795A", "#C84B00"
GREY, LIGHT = "#6b6b6b", "#d9d9d9"
YACC = "Validation accuracy" if EN else "验证准确率"
plt.rcParams.update({
    "font.family": ["DejaVu Sans", "Hiragino Sans GB", "Arial Unicode MS"],
    "svg.fonttype": "path", "axes.unicode_minus": False, "mathtext.fontset": "dejavusans",
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11, "figure.dpi": 100,
})
if SLIDES:  # projector: text roughly 1.5 times larger at the same figure size
    plt.rcParams.update({"font.size": 16, "axes.titlesize": 17, "axes.labelsize": 16,
                         "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14})


def save(fig, name):
    if SLIDES:  # projector: no label smaller than 14 pt
        for t in fig.findobj(matplotlib.text.Text):
            if t.get_text() and t.get_fontsize() < 14:
                t.set_fontsize(14)
    fig.savefig(os.path.join(OUTDIR, f"{name}.svg"), bbox_inches="tight")
    if os.environ.get("PNGDIR"):  # optional PNG copies for checking the figures by eye
        fig.savefig(os.path.join(os.environ["PNGDIR"], f"{name}{'_en' if EN and not SLIDES else ''}{'_s' if SLIDES else ''}.png"),
                    bbox_inches="tight", dpi=90)
    plt.close(fig)


def digit(ax, v, title=None, cmap="Greys", lim=None):
    a = np.array(v).reshape(28, 28)
    if lim is None:
        ax.imshow(a, cmap=cmap, vmin=0, vmax=1)
    else:
        ax.imshow(a, cmap=cmap, vmin=-lim, vmax=lim)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=13)


D = N["digits"]
T = D["table"]

# ---------------- F1: keep 10 pixels or 10 combinations, with the error images ----------------
fig = plt.figure(figsize=(11.6, 5.4))
orig = np.array(D["original"])
ax0 = fig.add_axes([0.0, 0.52, 0.15, 0.38])
digit(ax0, orig, f"原图（数字 {D['img_label']}）")
for col, (rec, name, err) in enumerate([(D["pix_recs"]["10"], "只留 10 个像素", D["pix_rec_err_img"]["10"]),
                                         (D["recs"]["10"], "10 个主成分", D["rec_err_img"]["10"])]):
    a1 = fig.add_axes([0.18 + col * 0.17, 0.52, 0.15, 0.38])
    digit(a1, rec, name)
    a2 = fig.add_axes([0.18 + col * 0.17, 0.04, 0.15, 0.38])
    digit(a2, orig - np.array(rec), f"原图 − 还原\n重建误差 {err:.1f}", cmap="RdBu_r", lim=1)
fig.text(0.075, 0.23, "下排：误差图\n红 = 还原得太白\n蓝 = 还原得太黑", ha="center", va="center", fontsize=13, color="#333")
ax = fig.add_axes([0.62, 0.3, 0.36, 0.4])
names = ["只留 10 个像素", "10 个主成分"]
vals = [D["pixels"]["10"]["share"], T["10"]["cum"]]
ax.barh(names, vals, color=[LIGHT, G], height=0.5)
for k, v in enumerate(vals):
    ax.text(v + 0.01, k, f"{v:.1%}", va="center", fontsize=11, weight="bold")
ax.set(xlim=(0, 1), xlabel="保住的总方差（所有验证图平均）")
ax.invert_yaxis()
save(fig, "F1")

# ---------------- F2: cumulative explained variance ----------------
cum = D["cum"]
fig, ax = plt.subplots(figsize=(7.2, 3.8))
ks = np.arange(1, 785)
ax.plot(ks, cum, color=G, lw=2.4)
for k, off in [(10, (8, -22)), (50, (8, -22)), (D["k90"], (8, -26))]:
    ax.scatter([k], [cum[k - 1]], color=G, s=40, zorder=5)
    ax.annotate(f"k = {k}：{cum[k - 1]:.1%}", (k, cum[k - 1]), xytext=off, textcoords="offset points", fontsize=10, color=G)
ax.axhline(0.9, color="#999", ls=":", lw=1)
ax.text(198, 0.905, "90%", ha="right", va="bottom", fontsize=10, color="#555")
ax.set(xlabel="主成分个数 k（只画到 200；全部是 784）", ylabel="累计比例", xlim=(0, 200), ylim=(0, 1.02))
save(fig, "F2")

# ---------------- F4: the mean image and the first six components ----------------
fig, axes = plt.subplots(1, 7, figsize=(12.0, 2.4))
digit(axes[0], D["mean"], "平均图")
lim = max(np.abs(np.array(D["comps"])).max(), 1e-9)
for k in range(6):
    digit(axes[k + 1], D["comps"][k], f"PC{k + 1}\n{D['evr'][k]:.1%}", cmap="RdBu_r", lim=lim)
fig.suptitle("每个主成分是 784 个像素的一组权重（红为正，蓝为负）", y=1.08, fontsize=12)
save(fig, "F4")

# ---------------- F5: validation accuracy against k for four methods ----------------
kk = [2, 5, 10, 20, 30, 40, 50, 100]
fig, ax = plt.subplots(figsize=(7.6, 4.0))
for key, name, col, mk in [("qda", "QDA", G, "o"), ("lr", "logistic regression", B, "s"), ("lda", "LDA", "#7a8ba0", "^"), ("tree", "决策树（深度 15）", "#8a6d3b", "D")]:
    ax.plot(kk, [T[str(k)][key] for k in kk], marker=mk, color=col, lw=2, ms=5, label=name)
ax.scatter([10, 20], [D["pixels"]["10"]["qda"], D["pixels"]["20"]["qda"]], marker="x", color=R, s=60, zorder=6, label="QDA，只留像素")
ax.text(100, 0.24, f"全部 784 个像素、不收缩的 QDA：{T['784']['qda']:.3f}", ha="right", fontsize=10, color=G)
ax.set(xlabel="主成分个数 k", ylabel=YACC, xticks=kk, ylim=(0.2, 1.0))
ax.legend(frameon=False, loc="lower right", fontsize=9.5, bbox_to_anchor=(1.0, 0.1))
save(fig, "F5")

# ---------------- F6: digits on the first two components ----------------
px, py, pl = np.array(D["pc2_x"]), np.array(D["pc2_y"]), np.array(D["pc2_label"])
cmap = plt.get_cmap("tab10")
fig, ax = plt.subplots(figsize=(6.6, 5.4))
for k in range(10):
    ax.scatter(px[pl == k], py[pl == k], s=10, color=cmap(k), alpha=0.45)
lab = np.array([[px[pl == k].mean(), py[pl == k].mean()] for k in range(10)])
pos = lab.copy()
for _ in range(300):                                   # push labels apart so none hides another
    for i in range(10):
        for j in range(10):
            d = pos[i] - pos[j]
            if i != j and np.hypot(*d) < 0.9:
                pos[i] += 0.02 * (d / (np.hypot(*d) + 1e-9))
for k in range(10):
    if np.hypot(*(pos[k] - lab[k])) > 0.15:
        ax.plot(*zip(lab[k], pos[k]), color=cmap(k), lw=1.2)
    ax.text(pos[k][0], pos[k][1], str(k), fontsize=16, weight="bold", ha="center", va="center", color=cmap(k),
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=cmap(k), alpha=0.95))
ax.set(xlabel=f"第 1 个主成分（{D['pc2_evr'][0]:.0%}）", ylabel=f"第 2 个主成分（{D['pc2_evr'][1]:.0%}）")
save(fig, "F6")

# ---------------- F7: lumps, 30 features on two components ----------------
Lm = N["lumps"]
lx, ly, ll = np.array(Lm["x"]), np.array(Lm["y"]), np.array(Lm["label"])
fig, ax = plt.subplots(figsize=(6.4, 4.8))
ax.scatter(lx[ll == 0], ly[ll == 0], s=14, facecolor="none", edgecolor=GREY, lw=0.8, label="良性")
ax.scatter(lx[ll == 1], ly[ll == 1], s=14, color=B, alpha=0.75, lw=0, label="恶性")
ax.set(xlabel=f"第 1 个主成分（{Lm['evr'][0]:.0%}）", ylabel=f"第 2 个主成分（{Lm['evr'][1]:.0%}）")
ax.legend(frameon=False)
save(fig, "F7")

# ---------------- F8: the whole chain in one picture ----------------
steps = [("维数太多", "784 个数画不出来，QDA 也估不出散布", B),
         ("候选：挑像素", "10 个像素只保住 3.7% 的方差", R),
         ("候选：组合", "投影到一个方向；散得最开 = 还原误差最小", G),
         ("主成分", "依次找垂直的方向；解释方差比例说明留住多少", G),
         ("选 k", "累计比例与验证准确率一起看", B),
         ("规矩", "只在拟合部分上 fit；验证集、测试集只 transform", B)]
fig, ax = plt.subplots(figsize=(11.6 if EN else 8.4, 4.0))
ax.axis("off")
ax.set_xlim(0, 10)
ax.set_ylim(0, len(steps))
heads = [ax.text(0, len(steps) - 0.5 - k, s[0], ha="center", va="center", weight="bold", color=s[2], fontsize=13)
         for k, s in enumerate(steps)]
fig.canvas.draw()
to_data = ax.transData.inverted()
BW = max(np.ptp(to_data.transform(t.get_window_extent())[:, 0]) for t in heads) + 0.5
for k, (h, sub, col) in enumerate(steps):
    yy = len(steps) - 1 - k
    ax.add_patch(Rectangle((0.2, yy + 0.12), BW, 0.76, facecolor=col, alpha=0.14, edgecolor=col, lw=1.4))
    heads[k].set_x(0.2 + BW / 2)
    ax.text(0.5 + BW, yy + 0.5, sub, va="center", fontsize=12.5, color="#333")
    if k < len(steps) - 1:
        ax.annotate("", xy=(0.2 + BW / 2, yy - 0.12), xytext=(0.2 + BW / 2, yy + 0.12),
                    arrowprops=dict(arrowstyle="->", color="#666", lw=1.2))
save(fig, "F8")

# ---------------- slides only ----------------
if SLIDES:
    # S1: one image is 784 numbers
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), gridspec_kw=dict(width_ratios=[1, 1.2]))
    digit(axes[0], D["original"], f"数字 {D['img_label']}")
    a = np.array(D["original"]).reshape(28, 28)
    r0, c0 = 10, 9  # a 7 x 7 block from the middle of the image
    axes[0].add_patch(Rectangle((c0 - 0.5, r0 - 0.5), 7, 7, fill=False, edgecolor=G, lw=2))
    axes[1].axis("off")
    for i in range(7):
        for j in range(7):
            v = a[r0 + i, c0 + j]
            axes[1].text(j, 6 - i, f"{v:.2f}", ha="center", va="center", fontsize=12, color="#333" if v < 0.5 else B)
    axes[1].set(xlim=(-0.6, 6.6), ylim=(-0.6, 6.6))
    axes[1].set_title("绿框里的 7 × 7 = 49 个数；整张图是 784 个数", fontsize=15)
    save(fig, "S1")
    # S2: candidate 1 only: keep the 10 most spread pixels
    fig = plt.figure(figsize=(10.0, 3.6))
    a1 = fig.add_axes([0.0, 0.1, 0.22, 0.72])
    a2 = fig.add_axes([0.25, 0.1, 0.22, 0.72])
    digit(a1, D["original"], f"原图（数字 {D['img_label']}）")
    digit(a2, D["pix_recs"]["10"], "只留 10 个像素")
    ax = fig.add_axes([0.6, 0.3, 0.38, 0.4])
    ax.barh(["只留 10 个像素"], [D["pixels"]["10"]["share"]], color=LIGHT, height=0.5)
    ax.text(D["pixels"]["10"]["share"] + 0.02, 0, f"{D['pixels']['10']['share']:.1%}", va="center", weight="bold")
    ax.set(xlim=(0, 1), xlabel="保住的总方差")
    save(fig, "S2")
    # S3: a combination of two features is a projection onto a line (here: the x1 axis)
    Cl = N["cloud"]
    cx, cy = np.array(Cl["x"]), np.array(Cl["y"])
    cx, cy = cx - cx.mean(), cy - cy.mean()
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    for xx, yy in zip(cx, cy):
        ax.plot([xx, xx], [yy, 0], color="#ccc", lw=0.7)
    ax.scatter(cx, cy, s=14, color=B, alpha=0.75)
    ax.axhline(0, color=G, lw=3)
    ax.set(aspect="equal", xlim=(-5.5, 5.5), ylim=(-5.5, 5.5), xlabel="x₁", ylabel="x₂", title="只留 x₁：投影到横轴，保住 68%")
    save(fig, "S3")
    # S4: one point, its projection and the right angle: spread + error = fixed
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    u = np.array([np.cos(np.radians(31.2)), np.sin(np.radians(31.2))])
    pnt = np.array([1.2, 2.6])
    proj = (pnt @ u) * u
    ax.plot([-3 * u[0], 4.2 * u[0]], [-3 * u[1], 4.2 * u[1]], color=G, lw=3)
    ax.plot([0, pnt[0]], [0, pnt[1]], color="#333", lw=2)
    ax.plot([0, proj[0]], [0, proj[1]], color=B, lw=5)
    ax.plot([pnt[0], proj[0]], [pnt[1], proj[1]], color=R, lw=3)
    ax.scatter([0], [0], color="#333", s=40, zorder=5)
    ax.scatter([pnt[0]], [pnt[1]], color=B, s=80, zorder=5)
    ax.text(0.15, 1.6, "到中心：固定", fontsize=13, color="#333", rotation=63)
    ax.text(proj[0] * 0.45, proj[1] * 0.45 - 0.45, "在线上：散布", fontsize=13, color=B, rotation=31)
    ax.text((pnt[0] + proj[0]) / 2 + 0.25, (pnt[1] + proj[1]) / 2 + 0.35, "到线：误差", fontsize=13, color=R)
    ax.text(-0.2, -0.45, "中心", fontsize=12, color="#555")
    ax.set(aspect="equal", xlim=(-1.5, 4), ylim=(-1.2, 3.2), xticks=[], yticks=[])
    for sp in ax.spines.values():
        sp.set_visible(False)
    save(fig, "S4")
print("figures ok")
