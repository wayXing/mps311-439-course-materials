"""English text for the Lesson 7 figures and widgets.
build_note_figures.py and build_widgets.py run once per language; with --lang en (or --slides) every
visible string passes through tr(), and any Chinese left over stops the build.
tr() replaces the longest Chinese strings first, and punctuation last.
"""
import re

PAIRS = [
    ('一张没参与拟合的验证图。还原 = 平均图 + 每个主成分乘上这张图在它上面的得分，只加前 k 个。重建误差是 784 个像素上“原图 − 还原”的平方和。拖动 k，看从几个开始认得出是几，误差怎样变小。', "A validation image not used for fitting. Rebuilt = mean image + each component times this image's score on it, adding only the first k. The reconstruction error is the sum of the squares of (original − rebuilt) over the 784 pixels. Drag k: from how many can you read the digit, and how does the error shrink?"),
    ('误差图：红 = 还原得太白，蓝 = 还原得太黑', 'Error image: red = rebuilt too light, blue = rebuilt too dark'),
    ('下排：误差图\n红 = 还原得太白\n蓝 = 还原得太黑', 'Bottom row: error images\nred = rebuilt too light\nblue = rebuilt too dark'),
    ('个像素（绿框），其余填平均值', ' pixels (green boxes), the rest filled with the average'),
    ('主成分个数 k（只画到 200；全部是 784）', 'Number of components k (drawn up to 200; all is 784)'),
    ('保住的总方差（所有验证图平均）', 'Total variance kept (average over validation images)'),
    ('原图 − 还原（误差）', 'Original − rebuilt (error)'),
    ('这张图的重建误差：', 'Reconstruction error of this image: '),
    ('10 个像素只保住 3.7% 的方差', '10 pixels keep only 3.7% of the variance'),
    ('原图 − 还原', 'Original − rebuilt'),
    ('重建误差 ', 'reconstruction error '),
    ('原来', 'before'),
    ("200 个点，两个特征。绿线是一条过中心的线，每个点沿灰色短线落到绿线上（投影）。右边两根柱子：投影后的点沿绿线散得多开（蓝），以及把点换成它的投影后丢掉的距离平方的平均（红，还原误差）。转动绿线，看两根柱子怎样变。",
     "200 points, two features. The green line passes through the centre, and each point drops onto it along a short grey line (its projection). The two bars on the right show how spread out the projections are along the green line (blue), and the average squared distance lost when each point is replaced by its projection (red, the reconstruction error). Turn the green line and watch the two bars."),
    ("绿线固定在散得最开的方向（第一个数）。橙线是第二个方向，拖动它。右边是两个数（点在绿线上的位置、在橙线上的位置）的相关系数：越接近 ±1，第二个数越是在重复第一个。",
     "The green line is fixed at the widest-spread direction (the first number). The orange line is a second direction: drag it. On the right is the correlation between the two numbers (where a point falls on the green line, and on the orange line). The closer it is to +1 or -1, the more the second number repeats the first."),
    ("按散得开的程度，从大到小留下前几个像素，其余像素用它们在拟合图像里的平均值填上。拖动个数，看保住的方差和图像怎样变。",
     "Pixels are kept in order of how spread out they are, largest first; every other pixel is filled with its average over the fitting images. Drag the number and watch the variance kept and the image."),
    ("与第一个数的相关系数", "Correlation with the first number"),
    ("沿第二条线的散布", "Spread along the second line"),
    ("第二条线的方向", "Direction of the second line"),
    ("留下的像素个数", "Number of pixels kept"),
    (" 个像素</div>", " pixels</div>"),
    ("只留 ", "Keep "),
    ("只在拟合部分上 fit；验证集、测试集只 transform", "Fit on the fitting part only; only transform the validation and test sets"),
    ("每个主成分是 784 个像素的一组权重（红为正，蓝为负）", "Each component is a set of 784 pixel weights (red positive, blue negative)"),
    ("依次找垂直的方向；解释方差比例说明留住多少", "Find perpendicular directions in turn; the explained variance ratio says how much is kept"),
    ("投影到一个方向；散得最开 = 还原误差最小", "Project onto a direction; widest spread = smallest reconstruction error"),
    ("784 个数画不出来，QDA 也估不出散布", "784 numbers cannot be drawn, and QDA cannot estimate the spread"),
    ("累计比例与验证准确率一起看", "Look at the cumulative share and the validation accuracy together"),
    ("只留 x₁：投影到横轴，保住 68%", "Keep x₁ only: project onto the horizontal axis, 68% kept"),
    ("到中心：固定", "to the centre: fixed"),
    ("在线上：散布", "along the line: spread"),
    ("到线：误差", "to the line: error"),
    ("中心", "centre"),
    ("绿框里的 7 × 7 = 49 个数；整张图是 784 个数", "The 7 × 7 = 49 numbers in the green box; the whole image is 784 numbers"),
    ("决策树（深度 15）", "Decision tree (depth 15)"),
    ("QDA，只留像素", "QDA, pixels only"),
    ("累计解释方差比例", "Cumulative explained variance ratio"),
    ("累计比例", "Cumulative share"),
    ("主成分个数 k", "Number of components k"),
    ("只留 10 个像素", "10 pixels only"),
    ("10 个主成分", "10 components"),
    ("候选：挑像素", "Candidate: pick pixels"),
    ("候选：组合", "Candidate: combine"),
    ("保住的总方差：", "Total variance kept: "),
    ("保住的总方差", "Total variance kept"),
    ("每张图要存的数：", "Numbers stored per image: "),
    ("（原来 784）", " (784 before)"),
    ("散得最开的方向", "Widest-spread direction"),
    ("个主成分还原", " components"),
    ("两者之和 =", "Sum of the two ="),
    ("个主成分（", " ("),
    ("原图（数字 ", "Original (digit "),
    ("全部 784 个像素、不收缩的 QDA：", "All 784 pixels, QDA without shrinking: "),
    ("验证准确率", "Validation accuracy"),
    ("维数太多", "Too many dimensions"),
    ("线的方向", "Direction of the line"),
    ("换一张：", "Another image: "),
    ("还原误差", "Reconstruction error"),
    ("平均图", "Mean image"),
    ("主成分", "Principal components"),
    ("规矩", "The rule"),
    ("选几个", "Choose how many"),
    (" 个\n", "\n"),
    ("第 ", "Component "),
    ("数字 ", "Digit "),
    ("保住 ", "Kept: "),
    ("图 1", "Image 1"), ("图 2", "Image 2"), ("图 3", "Image 3"), ("图 4", "Image 4"),
    ("原图", "Original"),
    ("散布", "Spread"),
    ("恶性", "Malignant"),
    ("良性", "Benign"),
    ("用 ", "From "),
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
