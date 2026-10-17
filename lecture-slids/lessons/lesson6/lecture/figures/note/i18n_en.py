"""English text for the Lesson 6 figures and widgets.
build_note_figures.py and build_widgets.py run once per language; with --lang en (or --slides) every
visible string passes through tr(), and any Chinese left over stops the build.
Order matters: whole captions and longer phrases first, so a short phrase never breaks a longer one.
"""
import re

PAIRS = [
    ("类 1（里圈）", "class 1 (inner ring)"),
    ("类 0（外圈）", "class 0 (outer ring)"),
    ("判错", "wrong"),
    # whole widget captions first
    ("一个节点里有 100 个点：20 个蓝点挤在中间一段，80 个灰点散在各处。上面是这些点（上下错开以免重叠），竖线是切的位置。下面两条曲线：每个切点的错分率（两边各按多数类判，错的比例）和 Gini（两边各自的 Gini，按点数加权平均）。拖动切线，看哪条曲线会动。",
     "One node holds 100 points: 20 blue points crowd into a band in the middle, and 80 grey points are spread everywhere. The points are at the top (spread up and down so they do not overlap), and the vertical line is the cut. The two curves below show, for every cut, the error rate (each side predicts its majority class; the share that is wrong) and the Gini (each side's Gini, averaged with the number of points as weights). Drag the cut and see which curve moves."),
    ("两个特征：细胞核平均半径（横轴）和平均纹理（纵轴）。蓝色区域判为恶性。点是 319 个用来拟合的肿块：蓝点恶性，灰圈良性；勾选后换成验证集的 107 个肿块：它们是从训练集里分出的，没有参与拟合。拖动深度，看矩形怎样越切越碎，两个准确率怎样变。",
     "Two features: mean nucleus radius (across) and mean texture (up). The blue region is called malignant. The points are the 319 lumps used for fitting: blue dots are malignant, grey circles benign. Tick the box to show the 107 validation lumps instead: they were split off the training set and took no part in fitting. Drag the depth and watch the rectangles get smaller and the two accuracies change."),
    ("Lesson 4 的环形数据：蓝点是类 1（里圈），灰色空心圆是类 0（外圈）。拖四个滑块移动四条切线，方框里的判为类 1，判错的点画橙圈。你能写出一条准确率超过 0.95 的规则吗？",
     "The ring data of Lesson 4: blue dots are class 1 (inner ring), grey circles are class 0 (outer ring). The four sliders move four cut lines; points inside the box are called class 1, and wrongly classified points get an orange ring. Can you write a rule with an accuracy above 0.95?"),
    # figure text
    ("交叉验证：每个深度拟合 5 次", "Cross-validation: 5 fits per depth"),
    ("验证集：每个深度拟合 1 次", "Validation set: 1 fit per depth"),
    ("最后只读一次", "read once, at the end"),
    ("最下面一行是叶", "The bottom row holds the leaves"),
    ("训练集", "training set"),
    ("拟合 ", "fit "),
    ("拟合", "fit"),
    ("验证", "validate"),
    ("测试", "test"),
    ("方框里的数：恶性 : 良性（训练肿块）", "Numbers in a box: malignant : benign (training lumps)"),
    ("，判为类 1；否则判为类 0", ", call it class 1; otherwise class 0"),
    ("直线（logistic regression）：准确率", "A straight line (logistic regression): accuracy"),
    ("按 Gini 选：两边越纯越好；错分率常常看不出差别", "Choose by Gini: the purer both sides, the better; the error rate often cannot tell"),
    ("数据少用交叉验证，数据多用验证集；选出的深度不能在搜索范围的两端", "Little data: cross-validation; much data: a validation set; the chosen depth must not be at either end of the range"),
    ("测试集只读一次；与别的方法同一交叉验证比较", "Read the test set once; compare methods with the same cross-validation"),
    ("环形数据：人能写出“如果……就……”的规则", "Ring data: a person can write \"if ... then\" rules"),
    ("准确率（训练集上的 5 折交叉验证）", "Accuracy (5-fold cross-validation, training set)"),
    ("树的最大深度（搜索 1 到 20）", "Maximum depth of the tree (searched from 1 to 20)"),
    ("同样的办法用在每一边，长成一棵树", "The same method on each side grows a tree"),
    ("验证集（107 个，没参与拟合）", "Validation set (107, not used for fitting)"),
    ("训练集满分，没见过的数据变差", "Perfect on training data, worse on unseen data"),
    ("5 折交叉验证（426 个）", "5-fold cross-validation (426)"),
    ("训练（319 个，拟合用）", "Training (319, used for fitting)"),
    ("每种分法选出的最好深度：", "Best depth chosen by each split: "),
    ("手写的方框：准确率", "Hand-written box: accuracy"),
    ("特征重要性（决策树）", "Feature importance (decision tree)"),
    (" 以后\n树不再长", " and deeper:\nno more growth"),
    ("片叶）：准确率", " leaves): accuracy"),
    ("深度 4 的树（", "Depth-4 tree ("),
    ("5 种验证集分法", "5 validation splits"),
    ("决策树（深度", "Decision tree (depth"),
    ("细胞核平均纹理", "Mean nucleus texture"),
    ("细胞核平均半径", "Mean nucleus radius"),
    ("最好：深度", "Best: depth"),
    ("树的最大深度", "Maximum depth of the tree"),
    ("一次切一刀", "One cut at a time"),
    ("两边再切", "Cut each side again"),
    ("直线不行", "A line fails"),
    ("切到底", "Cut to the end"),
    ("选深度", "Choose the depth"),
    ("判为良性", "Benign"),
    ("判为恶性", "Malignant"),
    ("恶性 ", "malignant "),
    ("，良性 ", ", benign "),
    ("验证（", "Validation ("),
    ("训练（", "Training ("),
    (" 张）", ")"),
    ("片叶", "leaves"),
    ("深度 ", "depth "),
    ("半径 ≤", "radius ≤"),
    ("纹理 ≤", "texture ≤"),
    ("检验", "Test"),
    ("准确率", "accuracy"),
    ("类 0", "Class 0"),
    ("类 1", "Class 1"),
    ("恶性", "Malignant"),
    ("良性", "Benign"),
    # widget controls and live text
    ("直线（logistic regression）：", "Straight line (logistic regression): "),
    ("显示验证集的点", "Show the validation lumps"),
    ("验证集上的准确率", "Accuracy on the validation lumps"),
    ("训练准确率", "Training accuracy"),
    ("验证准确率", "Validation accuracy"),
    ("切点的位置", "Position of the cut"),
    ("最大深度", "Maximum depth"),
    ("叶子数", "Leaves"),
    ("错分率", "Error rate"),
    ("左边：", "Left: "),
    ("右边：", "Right: "),
    (" 蓝，", " blue, "),
    (" 灰", " grey"),
    ("切在", "Cut at"),
    ("如果 ", "If "),
    (" 且 ", " and "),
    ("<label>左 ", "<label>Left "),
    ("<label>右 ", "<label>Right "),
    ("<label>下 ", "<label>Bottom "),
    ("<label>上 ", "<label>Top "),
    # tree diagram words, after everything longer
    (">根<", ">root<"),
    ("根 →", "root →"),
    ("叶", "leaf"),
    ("是", "yes"),
    ("否", "no"),
    # punctuation left between translated pieces
    ("？", "?"), ("（", " ("), ("）", ")"), ("：", ": "), ("，", ", "), ("、", ", "), ("；", "; "),
]
_CJK = re.compile(r"[　-〿一-鿿＀-￯]")


_PUNCT = {"？", "（", "）", "：", "，", "、", "；"}
_ORDER = sorted([p for p in PAIRS if p[0] not in _PUNCT], key=lambda p: -len(p[0])) + [p for p in PAIRS if p[0] in _PUNCT]


def tr(s):
    for zh, en in _ORDER:  # longest Chinese first, punctuation last
        s = s.replace(zh, en)
    return s


def check(s, where):
    left = _CJK.findall(s)
    if left:
        raise SystemExit(f"untranslated text in {where}: {''.join(left)[:40]}")
    return s
