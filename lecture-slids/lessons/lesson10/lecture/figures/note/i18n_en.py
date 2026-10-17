"""English text for the Lesson 10 figures and widgets.
build_note_figures.py and build_widgets.py run once per language; with --lang en (or --slides) every
visible string passes through tr(), and any Chinese left over stops the build.
tr() replaces the longest Chinese strings first, and punctuation last.
"""
import re

PAIRS = [
    ("滤波器的 9 个权重乘上图里 3×3 一小块的 9 个像素，再加起来，得到特征图上的一个数。同一个滤波器从左上角滑到右下角，就得到整张特征图：哪里黑，就是这个滤波器在那里“亮”。",
     "The 9 weights of the filter multiply the 9 pixels of a 3×3 patch of the image, and the products are added up: that gives one number of the feature map. Sliding the same filter from the top left to the bottom right gives the whole feature map: where it is dark, the filter \"lights up\"."),
    ("验证集的 10,000 张图全部右移，左边空出的列补成白色。网络没有重新训练，只是重新测一次准确率。",
     "All 10,000 validation images are moved to the right, and the empty columns on the left are filled with white. The networks are not trained again; only their accuracy is measured again."),
    ("上排：第一层的 3×3 滤波器（红正蓝负）；下排：它们在这张 7 上的特征图（越黑越“亮”）",
     "Top: first-layer 3×3 filters (red positive, blue negative). Bottom: their feature maps on this 7 (darker = lights up more)"),
    ("12,810 个权重；验证 0.987；平移 2 像素仍有 0.962", "12,810 weights; validation 0.987; still 0.962 after a 2-pixel move"),
    ("全连接网络不看邻居；平移 2 像素却掉到 0.870", "the fully connected network ignores neighbours; but falls to 0.870 after a 2-pixel move"),
    ("一个 3×3 滤波器到处用；108,160 → 160 个权重", "one 3×3 filter used everywhere; 108,160 → 160 weights"),
    ("混淆矩阵与常见错误；改对 214 张，新错 44 张", "confusion matrix, common mistakes; 214 fixed, 44 new"),
    ("每 2×2 留最大；小的平移影响变小", "keep the largest of each 2×2; small moves matter less"),
    (" 个权重（Lesson 9 的全连接网络：", " weights in all (the Lesson 9 network: "),
    ("3×3 滤波器（点格子改：1 → 0 → −1）", "3×3 filter (click a cell: 1 → 0 → −1)"),
    ("特征图（26×26，负数记 0）", "Feature map (26×26, negatives set to 0)"),
    ("全连接网络的验证准确率（5 轮）", "Validation accuracy, fully connected network (5 epochs)"),
    ("所有位置共用同一套 3×3 权重", "All positions share one set of 3×3 weights"),
    ("每个位置单独一套 3×3 权重", "Own 3×3 weights at each position"),
    ("CNN 在验证集上：错 ", "CNN on the validation set: "),
    ("特征图的一角（4×4）", "A corner of a feature map (4×4)"),
    ("每 2×2 留最大（2×2）", "Largest of each 2×2 (2×2)"),
    ("图（点一下选位置）", "Image (click to choose a position)"),
    (" 张（全连接 ", " wrong (fully connected: "),
    ("这个位置的值：", "Value at this position: "),
    ("最大池化\n2×2", "Max pooling\n2×2"),
    ("最大池化 2×2", "Max pooling 2×2"),
    ("卷积\n16 个 3×3", "Conv\n16 of 3×3"),
    ("卷积\n32 个 3×3", "Conv\n32 of 3×3"),
    ("整张图右移：", "Move every image right by "),
    ("16 个检测器：", "16 detectors: "),
    ("10 个 softmax", "10 softmax"),
    ("像素打乱后", "Pixels shuffled"),
    ("全连接网络", "Fully connected"),
    ("真实的数字", "True digit"),
    ("预测的数字", "Predicted digit"),
    ("局部 + 共享", "Local + shared"),
    ("打乱也不怕", "Shuffling does not hurt"),
    ("最大池化", "Max pooling"),
    ("错误分析", "Error analysis"),
    ("原来的图", "Original image"),
    (" → 判 ", " → predicted "),
    (" 个像素", " pixels"),
    ("个像素", "pixels"),
    (" 个权重", " weights"),
    ("全是 1", "All ones"),
    ("滤波器：", "Filter: "),
    ("滤波器 ", "filter "),
    ("滑过去", "Slide over"),
    ("权重 ", "weights "),
    ("一共 ", ""),
    ("数字：", "Digit: "),
    ("横边", "Horizontal edge"),
    ("竖边", "Vertical edge"),
    ("斜边", "Diagonal edge"),
    ("展平", "Flatten"),
    ("输入", "Input"),
    (" 张）", ")"),
    ("真 ", "true "),
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
