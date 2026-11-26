"""Lesson 10 Note v2 and Slides v2: the figures that are new or changed in v2.
Run (from this folder):  python build_figures_v2.py            -> Chinese figures here (note_v2/)
                         python build_figures_v2.py --lang en  -> English figures in ../note_en_v2/
                         python build_figures_v2.py --slides   -> English, projector-size text in ../slides_v2/
  F4  the CNN layer by layer (as ../note/F4, without the lesson number in its last line)
  F8  the one-page recap, nine steps
  F9  the loss: the 10 probabilities of two validation images (numbers_v2.json)
Reads ../note/numbers.json and numbers_v2.json. Colours as in ../note/build_note_figures.py: data blue, model green, error red.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text
from matplotlib.patches import Rectangle, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
SLIDES = "--slides" in sys.argv
ENG = SLIDES or "en" in sys.argv[1:]                  # English text: slides, or the English Note
OUTDIR = os.path.join(HERE, "..", "slides_v2") if SLIDES else os.path.join(HERE, "..", "note_en_v2") if ENG else HERE
os.makedirs(OUTDIR, exist_ok=True)
N = json.load(open(os.path.join(HERE, "..", "note", "numbers.json")))
N2 = json.load(open(os.path.join(HERE, "numbers_v2.json")))
B, G, R = "#0072B2", "#00795A", "#C84B00"
GREY = "#6b6b6b"
plt.rcParams.update({
    "font.family": ["DejaVu Sans", "Hiragino Sans GB", "Arial Unicode MS"],
    "svg.fonttype": "path", "axes.unicode_minus": False, "mathtext.fontset": "dejavusans",
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11, "figure.dpi": 100,
})
if SLIDES:
    plt.rcParams.update({"font.size": 16, "axes.titlesize": 17, "axes.labelsize": 16,
                         "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14})
EN = {  # every visible string of the slides version; a missing one stops the build
    "输入": "input", "卷积\n16 个 3×3": "convolution\n16 of 3×3", "最大池化\n2×2": "max pooling\n2×2",
    "卷积\n32 个 3×3": "convolution\n32 of 3×3", "展平": "flatten", "10 个 softmax": "10 softmax", "权重": "weights",
    "一共 {a} 个权重（全连接网络：{b}）": "{a} weights in all (the fully connected network: {b})",
    "真实数字 {t}，判成 {p}": "true {t}, predicted {p}", "概率": "probability", "数字": "digit",
    "损失 −log p = {v}": "loss −log p = {v}",
    "问题": "Problem", "启发": "Idea", "模型": "Model", "损失": "Loss", "求解": "Solution", "检验": "Test", "应用": "Application",
    "扩展": "Extension", "局限": "Limitations",
    "全连接网络打乱不怕，右移 2 像素却掉到 0.870": "shuffling does not hurt the fully connected network; a 2-pixel move drops it to 0.870",
    "只看 3×3，一个滤波器到处用：108,160 → 160 个权重": "look at 3×3, one filter everywhere: 108,160 → 160 weights",
    "卷积 + 最大池化 + softmax，12,810 个权重": "convolution + max pooling + softmax, 12,810 weights",
    "交叉熵：真实数字的概率取 −log": "cross-entropy: −log of the true digit's probability",
    "梯度下降；共享滤波器的梯度是各位置之和": "gradient descent; a shared filter's gradient adds up all positions",
    "平移 2 像素仍有 0.962；打乱后掉到 0.945": "still 0.962 after a 2-pixel move; 0.945 after shuffling",
    "验证 0.987 对 0.970；错图 130 对 300 张": "validation 0.987 against 0.970; 130 against 300 wrong images",
    "补边、步长、dropout、训练时随机平移（4 像素 {v}）": "padding, stride, dropout, random moves in training (4 pixels: {v})",
    "只吸收小的平移；更慢；结构选择多": "small moves only; slower; many structural choices",
}


def t(s, **kw):
    s = EN[s] if ENG else s
    return s.format(**kw) if kw else s


def save(fig, name):
    if SLIDES:
        for x in fig.findobj(matplotlib.text.Text):
            if x.get_text() and x.get_fontsize() < 14:
                x.set_fontsize(14)
    fig.savefig(os.path.join(OUTDIR, f"{name}.svg"), bbox_inches="tight")
    if os.environ.get("PNGDIR"):
        fig.savefig(os.path.join(os.environ["PNGDIR"], f"{name}{'_s' if SLIDES else ''}.png"), bbox_inches="tight", dpi=90)
    plt.close(fig)


D, C = N["dense"], N["cnn"]

# ---------------- F4: the CNN, layer by layer ----------------
steps = [("输入", "28×28×1", "0"), ("卷积\n16 个 3×3", "26×26×16", f"{C['layer_params'][0]:,}"), ("最大池化\n2×2", "13×13×16", "0"),
         ("卷积\n32 个 3×3", "11×11×32", f"{C['layer_params'][2]:,}"), ("最大池化\n2×2", "5×5×32", "0"),
         ("展平", "800", "0"), ("10 个 softmax", "10", f"{C['layer_params'][5]:,}")]
fig, ax = plt.subplots(figsize=(17, 3.4) if SLIDES else (11, 3.0))
for k, (name, shape, params) in enumerate(steps):
    x = k * 1.6
    col = G if params != "0" else B
    ax.add_patch(FancyBboxPatch((x, 0), 1.4, 1.3, boxstyle="round,pad=0.03", facecolor="white", edgecolor=col, lw=2))
    ax.text(x + 0.7, 1.02, t(name), ha="center", va="center", fontsize=11.5, color=col, weight="bold", linespacing=1.0)
    ax.text(x + 0.7, 0.62, shape, ha="center", va="center", fontsize=11.5)
    ax.text(x + 0.7, 0.25, f"{t('权重')} {params}", ha="center", va="center", fontsize=11, color=GREY)
    if k < len(steps) - 1:
        ax.annotate("", (x + 1.58, 0.65), (x + 1.43, 0.65), arrowprops=dict(arrowstyle="->", color=GREY))
ax.text(5.5, -0.35, t("一共 {a} 个权重（全连接网络：{b}）", a=f"{C['params']:,}", b=f"{D['params']:,}"), ha="center", fontsize=11)
ax.set(xlim=(-0.1, 11.1), ylim=(-0.6, 1.4))
ax.axis("off")
save(fig, "F4")

# ---------------- F9: the loss of two validation images ----------------
ims = N2["loss"]["images"]
fig, axes = plt.subplots(1, 4, figsize=(17, 4.2) if SLIDES else (11, 2.8),
                         gridspec_kw=dict(width_ratios=[1, 2.2, 1, 2.2]))
for k, im in enumerate(ims):
    a_img, a_bar = axes[2 * k], axes[2 * k + 1]
    a_img.imshow(np.array(im["img"]), cmap="Greys")
    a_img.set(xticks=[], yticks=[])
    for sp in a_img.spines.values():
        sp.set_visible(False)
    a_img.set_title(t("真实数字 {t}，判成 {p}", t=im["true"], p=im["pred"]), fontsize=10)
    colours = [GREY] * 10
    colours[im["pred"]] = R if im["pred"] != im["true"] else G
    colours[im["true"]] = G
    a_bar.bar(range(10), im["p"], color=colours)
    a_bar.set(xticks=range(10), ylim=(0, 1.05), xlabel=t("数字"), ylabel=t("概率"))
    a_bar.set_title(t("损失 −log p = {v}", v=f"{abs(im['minus_log']):.3f}" if abs(im["minus_log"]) < 0.1 else f"{im['minus_log']:.2f}"),
                    color=G if im["pred"] == im["true"] else R)
fig.tight_layout()
save(fig, "F9")

# ---------------- F8: the whole lesson in nine steps (Note only) ----------------
if not SLIDES:
    E = N2["extension"]
    steps = [("问题", "全连接网络打乱不怕，右移 2 像素却掉到 0.870", R),
             ("启发", "只看 3×3，一个滤波器到处用：108,160 → 160 个权重", G),
             ("模型", "卷积 + 最大池化 + softmax，12,810 个权重", G),
             ("损失", "交叉熵：真实数字的概率取 −log", G),
             ("求解", "梯度下降；共享滤波器的梯度是各位置之和", G),
             ("检验", "平移 2 像素仍有 0.962；打乱后掉到 0.945", B),
             ("应用", "验证 0.987 对 0.970；错图 130 对 300 张", B),
             ("扩展", t("补边、步长、dropout、训练时随机平移（4 像素 {v}）", v=f"{E['random_translation']['moved4']:.3f}"), B),
             ("局限", "只吸收小的平移；更慢；结构选择多", R)]
    fig, ax = plt.subplots(figsize=(10.4 if ENG else 8.6, 5.6))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(steps))
    BW = 1.9 if ENG else 1.3
    for k, (h, sub, col) in enumerate(steps):
        yy = len(steps) - 1 - k
        ax.add_patch(Rectangle((0.2, yy + 0.12), BW, 0.76, facecolor=col, alpha=0.14, edgecolor=col, lw=1.4))
        ax.text(0.2 + BW / 2, yy + 0.5, t(h), ha="center", va="center", weight="bold", color=col, fontsize=13)
        ax.text(0.5 + BW, yy + 0.5, t(sub) if sub in EN else sub, va="center", fontsize=12, color="#333")
        if k < len(steps) - 1:
            ax.annotate("", xy=(0.2 + BW / 2, yy - 0.12), xytext=(0.2 + BW / 2, yy + 0.12),
                        arrowprops=dict(arrowstyle="->", color="#666", lw=1.2))
    save(fig, "F8")
print("figures ok", OUTDIR)
