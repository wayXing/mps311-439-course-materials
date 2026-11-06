"""Lesson 6 Note V3: new static figure F6 (feature importance before and after removing mean concave points).
Reads numbers_v3.json. Style as ../note/build_note_figures.py (model green, light grey for the rest)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
N = json.load(open(os.path.join(HERE, "numbers_v3.json")))
G, LIGHT = "#00795A", "#d9d9d9"
plt.rcParams.update({"font.family": ["DejaVu Sans", "Hiragino Sans GB", "Arial Unicode MS"], "svg.fonttype": "path",
                     "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11})
I = N["imp30"]
fig, axes = plt.subplots(1, 2, figsize=(12.0, 3.4), gridspec_kw=dict(wspace=0.95))
for ax, top, title in [(axes[0], I["top"], f"全部 30 个特征（交叉验证 {I['cv_all']:.3f}）"),
                       (axes[1], I["top_drop"], f"拿掉 mean concave points（交叉验证 {I['cv_drop']:.3f}）")]:
    cols = [G if k == 0 else LIGHT for k in range(len(top))]
    ax.barh([t[0] for t in top], [t[1] for t in top], color=cols, height=0.55)
    for k, t in enumerate(top):
        ax.text(t[1] + 0.01, k, f"{t[1]:.2f}", va="center", fontsize=10)
    ax.set(xlim=(0, 0.85), xlabel="特征重要性（深度 5 的树）", title=title)
    ax.invert_yaxis()
fig.savefig(os.path.join(HERE, "F6.svg"), bbox_inches="tight")
if os.environ.get("PNGDIR"):
    fig.savefig(os.path.join(os.environ["PNGDIR"], "F6.png"), bbox_inches="tight", dpi=90)
