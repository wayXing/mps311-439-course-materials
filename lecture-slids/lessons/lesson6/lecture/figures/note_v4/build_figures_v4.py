"""Lesson 6 Note: static figures F1 (Titanic depth-2 tree), F2 (training, cross-validation and five validation splits
against depth), F6 (the small investigation: importance after removing one feature). Reads numbers_v4.json.
Style as ../note/build_note_figures.py: survived = blue, died = grey, model green."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text
from matplotlib.patches import FancyBboxPatch
HERE = os.path.dirname(os.path.abspath(__file__))
N = json.load(open(os.path.join(HERE, "numbers_v4.json")))
EN = "en" in " ".join(os.sys.argv[1:]).split()          # python build_figures_v4.py --lang en -> ../note_v4_en/
OUTDIR = os.path.join(HERE, "..", "note_v4_en") if EN else HERE
os.makedirs(OUTDIR, exist_ok=True)
TR = {"方框里的数：生还 : 遇难（535 名训练乘客）": "numbers: survived : died (535 training passengers)",
      "5 种验证集分法（各 134 人）": "5 validation splits (134 each)", "训练（535 人）": "training (535)",
      "5 折交叉验证": "5-fold cross-validation", "树的最大深度": "max depth of the tree", "准确率": "Accuracy",
      "全部 6 个特征": "all 6 features", "拿掉舱位": "without class", "拿掉性别": "without sex", "特征重要性": "feature importance",
      "头等或二等舱？": "1st or 2nd class?", "年龄 ≤ 13？": "Age ≤ 13?", "男性？": "Male?", "判为生还": "survived", "判为遇难": "died",
      "最下面一行是叶": "bottom row: the leaves", "根 →": "root →", "兄弟姐妹/配偶": "siblings/spouse", "父母/子女": "parents/children",
      "性别": "sex", "舱位": "class", "年龄": "age", "票价": "fare", "（": " (", "）": ")", "是": "yes", "否": "no"}
if EN:
    import re as _re
    _set = matplotlib.text.Text.set_text
    def _tr(s):
        for k in sorted(TR, key=len, reverse=True):
            s = s.replace(k, TR[k])
        if _re.search(r"[\u4e00-\u9fff]", s):
            raise SystemExit(f"untranslated figure text: {s}")
        return s
    matplotlib.text.Text.set_text = lambda self, s: _set(self, _tr(str(s)) if s is not None else s)
B, G, GREY, LIGHT = "#0072B2", "#00795A", "#6b6b6b", "#d9d9d9"
plt.rcParams.update({"font.family": ["DejaVu Sans", "Hiragino Sans GB", "Arial Unicode MS"], "svg.fonttype": "path",
                     "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11})
def save(fig, name):
    fig.savefig(os.path.join(OUTDIR, f"{name}.svg"), bbox_inches="tight")
    if os.environ.get("PNGDIR"):
        fig.savefig(os.path.join(os.environ["PNGDIR"], f"{name}_v4.png"), bbox_inches="tight", dpi=90)
    plt.close(fig)

# F1: the depth-2 tree, drawn as questions
nodes = N["t2"]["nodes"]
q = {"sex": "男性？", "age": "年龄 ≤ 13？", "pclass": "头等或二等舱？"}
pos = {0: (5, 3), 1: (2.6, 1.7), 4: (7.4, 1.7), 2: (1.15, 0.45), 3: (3.75, 0.45), 5: (6.25, 0.45), 6: (8.85, 0.45)}
fig, ax = plt.subplots(figsize=(10.4, 4.6)); ax.axis("off"); ax.set_xlim(-0.3, 10.3); ax.set_ylim(-0.25, 3.7)
# sklearn: left = feature <= thr. sex<=0.5 is male (yes); age<=13 yes; pclass<=2.5 is first or second class (yes)
for nd in nodes:
    x, yy = pos[nd["i"]]; s, d = nd["surv"], nd["n"] - nd["surv"]
    if nd["left"] >= 0:
        ax.add_patch(FancyBboxPatch((x - 1.35, yy - 0.28), 2.7, 0.56, boxstyle="round,pad=0.04", fc="#eef3f8", ec=B, lw=1.4))
        ax.text(x, yy + 0.07, q[nd["feat"]], ha="center", va="center", fontsize=12, weight="bold")
        ax.text(x, yy - 0.15, f"{s} : {d}", ha="center", va="center", fontsize=10.5, color="#444")
        for child, lab in [(nd["left"], "是"), (nd["right"], "否")]:
            cx, cy = pos[child]
            ax.annotate("", xy=(cx, cy + 0.32), xytext=(x, yy - 0.3), arrowprops=dict(arrowstyle="->", color="#666", lw=1.2))
            ax.text((x + cx) / 2 + (-0.15 if lab == "是" else 0.15), (yy + cy) / 2 + 0.05, lab, ha="center", fontsize=11, color="#333")
    else:
        col = B if s > d else GREY
        ax.add_patch(FancyBboxPatch((x - 1.15, yy - 0.3), 2.3, 0.6, boxstyle="round,pad=0.04", fc="white", ec=col, lw=1.8))
        ax.text(x, yy + 0.1, "判为生还" if s > d else "判为遇难", ha="center", va="center", fontsize=11.5, color=col, weight="bold")
        ax.text(x, yy - 0.14, f"{s} : {d}", ha="center", va="center", fontsize=10.5, color="#444")
ax.text(3.7, 3.0, "根 →", ha="right", va="center", fontsize=10.5, color="#666")
ax.text(0, 3.45, "方框里的数：生还 : 遇难（535 名训练乘客）", fontsize=10.5, color="#444")
ax.text(5, -0.2, "最下面一行是叶", ha="center", fontsize=10.5, color="#666")
save(fig, "F1")

# F2: training accuracy, cross-validation, and five validation splits
rows = N["sweep"]["rows"]; dep = [r["d"] for r in rows]
fig, ax = plt.subplots(figsize=(7.6, 4.0))
for k, sp in enumerate(N["valsplits"]["splits"]):
    ax.plot(dep, sp["acc"], "-", color=G, lw=1.1, alpha=0.5, label="5 种验证集分法（各 134 人）" if k == 0 else None)
    ax.scatter([sp["best"]], [sp["acc"][sp["best"] - 1]], color=G, s=28, zorder=5)
ax.plot(dep, [r["train"] for r in rows], "o-", color=B, lw=2.2, label="训练（535 人）")
ax.plot(dep, [r["cv"] for r in rows], "s-", color="#333", lw=2.4, label="5 折交叉验证")
ax.set(xlabel="树的最大深度", ylabel="准确率", xticks=dep, ylim=(0.68, 1.0))
ax.legend(frameon=False, loc="upper left")
save(fig, "F2")

# F6: importance with all features, without pclass, without sex
inv = N["inv"]["drops"]; full = N["best"]["imp"]
names = {"sex": "性别", "pclass": "舱位", "age": "年龄", "fare": "票价", "sibsp": "兄弟姐妹/配偶", "parch": "父母/子女"}
panels = [(full, f"全部 6 个特征（{N['best']['cv']:.3f}）"), (inv["pclass"]["imp"], f"拿掉舱位（{inv['pclass']['cv']:.3f}）"),
          (inv["sex"]["imp"], f"拿掉性别（{inv['sex']['cv']:.3f}）")]
fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.2), sharex=True, gridspec_kw=dict(wspace=0.55))
order = ["sex", "pclass", "age", "fare", "sibsp", "parch"]
for ax, (imp, title) in zip(axes, panels):
    keys = [k for k in order if k in imp]
    ax.barh([names[k] for k in keys], [imp[k] for k in keys], color=[G if k == "fare" else LIGHT for k in keys], height=0.55)
    for j, k in enumerate(keys):
        ax.text(imp[k] + 0.01, j, f"{imp[k]:.3f}", va="center", fontsize=10)
    ax.set(xlim=(0, 0.75), title=title, xlabel="特征重要性")
    ax.invert_yaxis()
save(fig, "F6")
