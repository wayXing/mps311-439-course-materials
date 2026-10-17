"""English text for the Lesson 9 figures and widgets.
build_note_figures.py and build_widgets.py run once per language; with --lang en (or --slides) every
visible string passes through tr(), and any Chinese left over stops the build.
tr() replaces the longest Chinese strings first, and punctuation last.
"""
import re

PAIRS = [
    ("（同一验证集）", " (same validation set)"),
    ("蓝：这里有墨水，\n单元的加权和变小", "blue: ink here\nlowers the weighted sum"),
    ("红：这里有墨水，\n单元的加权和变大", "red: ink here\nraises the weighted sum"),
    ("每条线是一个 logistic 单元：它把平面分成两边。蓝色阴影是“在每条线的内侧”的地方，机器把那里的点判为内圈（蓝点）。拖动滑块转动、平移三条线，让阴影正好盖住蓝点。",
     "Each line is one logistic unit: it splits the plane into two sides. The blue shading is where a point is on the inner side of every line; the machine calls those points inner ring (blue dots). Drag the sliders to turn and shift the three lines until the shading covers exactly the blue dots."),
    ("这里的线不是手摆的：每个网络都用交叉熵和梯度下降训练了 300 步，机器自己摆。绿线是每个隐藏单元的直线（它的输入加权和等于 0 的地方），蓝色阴影是网络判为内圈的地方。",
     "Nobody placed these lines by hand: each network was trained for 300 steps with cross-entropy and gradient descent, and the machine placed them. Each green line belongs to one hidden unit (where its weighted sum of the inputs is 0); the blue shading is where the network says inner ring."),
    ("去掉中间那一步：不管几个单元，阴影的边界都只是一条直线。", "Without the middle step: however many units, the edge of the shading is a single straight line."),
    ("另一种弯法", "Another bend"),
    ("轮：验证损失最低", ": lowest validation loss"),
    ("差距越拉越大", "the gap keeps growing"),
    ("每个：一条直线 + sigmoid\n给出一个新特征", "Each: a straight line + sigmoid\ngives one new feature"),
    ("交叉熵 + 梯度下降；误差往回传，一小批一步", "cross-entropy + gradient descent; error passed back, a batch per step"),
    ("环：一条直线 50%；先想到“距离”才 99%", "Ring: one line 50%; 99% only after we think of distance"),
    ("每个一条直线；给一个新特征，再组合", "one line each; one new feature each, then combine"),
    ("去掉激活；仍是一条直线（50.3%）", "no activation:; still one straight line (50.3%)"),
    ("92.8% → 97.8%；验证损失第 4 轮最低", "92.8% → 97.8%; validation loss lowest at epoch 4"),
    ("用 3 个新特征\n给出内圈的概率", "uses the 3 new features\nto give P(inner ring)"),
    ("网络，去掉中间的激活（5 轮）", "Network, no activation in the middle (5 epochs)"),
    ("人想出的特征：到中心的距离", "Distance to the centre (our feature)"),
    ("验证集准确率（10,000 张）", "Validation accuracy (10,000 images)"),
    ("先想到这个特征：准确率 ", "With that feature first: accuracy "),
    ("sigmoid：压到 0 和 1 之间", "sigmoid: into 0 to 1"),
    ("验证损失最低：第 ", "Lowest validation loss: epoch "),
    ("一条直线：准确率 ", "One straight line: accuracy "),
    ("ReLU：负的变成 0", "ReLU: negatives become 0"),
    ("几个小 logistic", "A few small logistics"),
    ("中间：\n3 个小 logistic", "Middle: 3 small\nlogistic regressions"),
    ("输出：\n再一个 logistic", "Output:\none more"),
    ("输入：\n两个坐标", "Input:\ntwo coordinates"),
    ("隐藏单元个数：", "Hidden units: "),
    ("中间那一步：", "Middle step: "),
    ("训练后的准确率", "Accuracy after training"),
    ("　　　　　平移 ", "shift "),
    (" 条线　转动 ", ": turn "),
    ("只用第 1 条线", "Use line 1 only"),
    ("网络（20 轮）", "Network (20 epochs)"),
    ("网络（5 轮）", "Network (5 epochs)"),
    ("中间要弯一下", "Bend in the middle"),
    ("内圈（类 1）", "inner ring (class 1)"),
    ("外圈（类 0）", "outer ring (class 0)"),
    ("交叉熵损失", "Cross-entropy loss"),
    ("看一种摆法", "Show one placement"),
    ("特征要人想", "Features need a person"),
    ("轮（epoch）", "Epoch"),
    (" 轮", ""),
    ("单元 ", "Unit "),
    ("准确率", "Accuracy"),
    ("去掉", "None"),
    ("损失", "Loss"),
    ("点数", "Points"),
    ("训练", "Training"),
    ("验证", "Validation"),
    ("第 ", "Line "),
    ("（", " ("), ("）", ")"), ("：", ": "), ("，", ", "), ("、", ", "), ("；", "; "),
]
_PUNCT = {"（", "）", "：", "，", "、", "；"}
_ORDER = sorted([p for p in PAIRS if p[0] not in _PUNCT], key=lambda p: -len(p[0])) + [p for p in PAIRS if p[0] in _PUNCT]
_CJK = re.compile(r"[　-〿一-鿿＀-￯]")


def tr(s):
    for zh, en in _ORDER:  # longest Chinese first, punctuation last
        s = s.replace(zh, en)
    return s


def check(s, where):
    left = _CJK.findall(s)
    if left:
        raise SystemExit(f"untranslated text in {where}: {''.join(left)[:40]}")
    return s
