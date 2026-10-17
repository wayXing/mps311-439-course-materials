"""English text for the Lesson 5 figures and widgets.
build_note_figures.py and build_widgets.py run once per language; with --lang en (or --slides) every
visible string passes through tr(), and any Chinese left over stops the build.
Order matters: whole captions and longer phrases first, so a short phrase never breaks a longer one.
"""
import re

PAIRS = [
    ('<b>+</b> 是每一类的中心；每一类两层椭圆：实线内圈约含 40% 的点，虚线外圈约含 86%', '<b>+</b> is the centre of each class; two ellipses per class: the solid inner one holds about 40% of its points, the dashed outer one about 86%'),
    ('虚线椭圆：每一类的描述（约含 86% 的点）；绿色区域：判为类 1；', 'Dashed ellipse: the description of each class (about 86% of its points); green region: called class 1; '),
    ('虚线椭圆：收缩后每一类的散布；绿色区域：判为类 1', 'Dashed ellipse: the spread of each class after shrinking; green region: called class 1'),
    ('判错', 'wrong'),
    # whole widget captions first
    ("每一类画成一个椭圆：十字是中心（平均值），内圈和外圈分别约含这一类 40% 和 86% 的点。点一下或拖动黑点，右边显示：黑点在每一类里有多常见（这一类的钟形“小山”在那里有多高），乘上这一类占的比例，两者谁大就判谁。",
     "Each class is drawn as an ellipse: the cross is its centre (the mean), and the inner and outer rings hold about 40% and 86% of its points. Click or drag the black dot. On the right you see how common the dot is in each class (how high that class's bell-shaped hill is there), times the share of that class. The larger one wins."),
    ("虚线椭圆是每一类的描述（约含 86% 的点）。“共用一个散布”时，两个椭圆形状、大小、方向都一样，只是中心不同；“各有散布”时，每类用自己的。绿色区域判为类 1，判错的点画橙圈。两个按钮来回切换，看边界的形状。",
     "The dashed ellipses describe each class (about 86% of its points). With one shared spread, the two ellipses have the same shape, size and direction, and only their centres differ. With one spread per class, each class uses its own. The green region is called class 1, and wrongly classified points get an orange ring. Switch between the two buttons and watch the shape of the boundary."),
    ("蓝点是类 1（120 个），灰色空心圆是类 0（240 个）。绿色一侧判为类 1，判错的点画橙圈。拖两个滑块，看能不能找到一条直线把两类分开；再点按钮，看 logistic regression 找到的直线。",
     "Blue dots are class 1 (120 points), grey circles are class 0 (240 points). The green side is called class 1, and wrongly classified points get an orange ring. Drag the two sliders: can you find a straight line that separates the two classes? Then press the button to see the line that logistic regression finds."),
    ("每类自己的散布（r = 0，就是 QDA）和一个半径为 1 的圆按比例混合：散布 = (1 − r) × 自己的散布 + r × 圆。拖动 r，看两个虚线椭圆怎样变圆，绿色边界怎样变。",
     "Each class's own spread (r = 0, which is QDA) is mixed with a circle of radius 1: spread = (1 − r) × own spread + r × circle. Drag r and watch the two dashed ellipses become round, and the green boundary change."),
    # figure text, longer phrases first
    ("往“圆”收一点（reg_param），用交叉验证选，不落在两端", "Pull towards a circle (reg_param); choose by CV, not at an end"),
    ("灰色带：只搜 0.01、0.1、0.3、0.5，最好的 0.01 落在左端", "Grey band: search only 0.01, 0.1, 0.3, 0.5; its best, 0.01, is at the left end"),
    ("\n（每类同一个圆，\n只看离哪类平均图近）", "\n(every class the same circle:\nonly the nearest class average counts)"),
    ("全收成圆：", "All circle: "),
    ("共用散布：直线（LDA）；各有散布：弯线（QDA）", "Shared spread: straight (LDA); own spread: curved (QDA)"),
    ("reg_param（往“圆”收多少；对数刻度）", "reg_param (how far towards a circle; log scale)"),
    ("一紧一散的两类，最好的直线也只比猜多数类好一点", "One tight class, one spread out: the best line barely beats the majority guess"),
    ("在这一类里有多常见 × 这一类的比例，谁大判谁", "How common in this class × share of the class; the larger wins"),
    ("一维：钟形曲线；二维：椭圆（中心与散布）", "1-D: a bell curve; 2-D: an ellipse (centre and spread)"),
    ("同一交叉验证下比较；更灵活不一定更好", "Compare with the same cross-validation; more flexible is not always better"),
    ("准确率（训练集上的 5 折交叉验证）", "Accuracy (5-fold cross-validation, training set)"),
    ("数字 0 每个像素散多开（蓝越深越开）\n浅红：", "How spread each pixel of digit 0 is (darker blue: more)\nlight red: "),
    (" 个像素永远是 0", " pixels are always 0"),
    ("每一类的中心：训练图像的平均", "The centre of each class: the average training image"),
    ("类 1（120 个）", "Class 1 (120)"),
    ("类 0（240 个）", "Class 0 (240)"),
    ("训练肿块 426 个", "426 training lumps"),
    ("良性（267 个）", "Benign (267)"),
    ("恶性（159 个）", "Malignant (159)"),
    ("LDA（共用散布）", "LDA (shared spread)"),
    ("QDA（各有散布）", "QDA (own spread)"),
    ("LDA：共用散布", "LDA: shared spread"),
    ("QDA：各有散布", "QDA: own spread"),
    ("要从数据里估的数", "Numbers to estimate from the data"),
    ("细胞核平均半径", "Mean nucleus radius"),
    ("交叉验证准确率", "Cross-validated accuracy"),
    ("良性\n平均 ", "Benign\nmean "),
    ("恶性\n平均 ", "Malignant\nmean "),
    ("\n标准差 ", "\nSD "),
    ("估不出来时", "Cannot estimate"),
    ("描述每一类", "Describe each class"),
    ("永远猜良性", "Always guess benign"),
    ("不收缩：", "No shrinking: "),
    ("两个候选", "Two candidates"),
    ("直线不够", "A line is not enough"),
    ("，误报 ", ", false alarms "),
    ("漏诊 ", "missed "),
    ("最好：", "Best: "),
    ("\n准确率 ", "\naccuracy "),
    ("高 ", "height "),
    ("检验", "Test"),
    ("比较", "Compare"),
    ("密度", "Density"),
    ("共用的散布：仍然是平的\n（", "Shared spread: still flat\n("),
    (" 个像素在所有图像里都是 0）", " pixels are 0 in every image)"),
    ("只留对角线：仍然是平的\n（这一类里恒为 0 的像素）", "Diagonal only: still flat\n(a pixel always 0 in this class)"),
    ("一个圆：每个方向都有厚度", "A circle: thick in every direction"),
    ("蓝色虚线：一类自己的散布，在“像素 b”方向上没有厚度；绿色：往它收的形状", "Blue dashed: one class's own spread, with no thickness along pixel b. Green: the shape we pull it towards"),
    ("两类共用一个 Σ", "One shared Σ"),
    ("候选 1：往共用的散布收", "Candidate 1:\ntowards the shared spread"),
    ("候选 2：只留对角线", "Candidate 2:\nkeep only the diagonal"),
    ("候选 3：往一个圆收", "Candidate 3:\ntowards a circle"),
    ("蓝色虚线：一类自己的散布，在“像素 b”方向上没有厚度", "Blue dashed: one class's own spread, with no thickness along pixel b"),
    ("每类有自己的 Σ", "Each class its own Σ"),
    ("像素 a", "pixel a"),
    ("像素 b", "pixel b"),
    # widget controls
    ("logistic regression 的直线", "Logistic regression's line"),
    ("往“圆”收多少 r", "Pull towards a circle, r"),
    ("在类 1 里有多常见", "How common in class 1"),
    ("在类 0 里有多常见", "How common in class 0"),
    ("× 比例 1/3", "× share 1/3"),
    ("× 比例 2/3", "× share 2/3"),
    ("每类有自己的散布", "One spread per class"),
    ("两类共用一个散布", "One shared spread"),
    ("永远猜类 0：", "Always guess class 0: "),
    ("类 1 的概率", "Probability of class 1"),
    ("判为类 0", "called class 0"),
    ("判为类 1", "called class 1"),
    ("准确率", "Accuracy"),
    ("方向", "Direction"),
    ("位置", "Position"),
    ("类 0", "Class 0"),
    ("类 1", "Class 1"),
    # punctuation left between translated pieces
    ("（", " ("), ("）", ")"), ("：", ": "), ("，", ", "), ("、", ", "), ("；", "; "),
]
_CJK = re.compile(r"[　-〿一-鿿＀-￯]")


def tr(s):
    for zh, en in sorted(PAIRS, key=lambda p: -len(p[0])):
        s = s.replace(zh, en)
    return s


def check(s, where):
    left = _CJK.findall(s)
    if left:
        raise SystemExit(f"untranslated text in {where}: {''.join(left)[:40]}")
    return s
