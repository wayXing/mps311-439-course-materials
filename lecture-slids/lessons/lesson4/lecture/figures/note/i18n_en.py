"""English text for the Lesson 4 figures and widgets.
build_note_figures.py and build_widgets.py run once per language; with --lang en (or --slides) every
visible string passes through tr(), and any Chinese left over stops the build.
Order matters: whole captions and longer phrases first, so a short phrase never breaks a longer one.
"""
import re

PAIRS = [
    ("数据长这样：每个数字一张 28 × 28 的图", "What the data look like: one 28 × 28 image per digit"),
    ("模型用全部 30 个测量值。", "The model uses all 30 measurements. "),
    # whole widget captions first
    ("蓝点是恶性，灰色空心圆是良性，两个特征都已标准化。绿线是 p = 0.5 的位置，绿色越深，模型给“恶性”的概率越高。先只动 w₁ 和 w₂，再只动 b，看线怎样变。",
     "Blue dots are malignant, grey circles are benign; both features are standardised. The green line is where p = 0.5; the darker the green, the higher the probability of malignant. Move only w₁ and w₂ first, then only b, and watch the line."),
    ("426 个训练样本，每个的概率来自 5 折交叉验证（它所在的那一折没有参加训练）。上图：蓝色是恶性的概率分布，灰色是良性的；概率不小于 t 的判为恶性；纵轴为开方刻度，矮柱也看得见。红色数字是两种错误：漏诊和误报。右边的曲线是 ROC 曲线，绿点是当前阈值的位置。",
     "426 training tumours; each probability comes from 5-fold cross-validation (its own fold was not used for training). Top: blue is the distribution for malignant, grey for benign; a probability of at least t is called malignant; the vertical scale is square-root, so short bars stay visible. The red numbers are the two kinds of error: missed cancers and false alarms. The curve on the right is the ROC curve; the green dot is the current threshold."),
    ("交叉验证给出的“恶性”概率", "Cross-validated probability of malignant"),
    ("ROC 曲线（AUC ", "ROC curve (AUC "),
    ("每一类的 784 个权重排成 28 × 28（红为正，蓝为负）", "The 784 weights of each class as a 28 × 28 image (red positive, blue negative)"),
    ("模型给“恶性”的概率 p（实际是恶性）", "p given to malignant (lump is malignant)"),
    ("直线的输出超出 0 到 1，还被远处的点拉动", "A line's output leaves 0 to 1, and far points pull it"),
    ("按错误代价在训练集内选，测试集只读一次", "Chosen by error costs inside training; test set read once"),
    ("准确率之外，看漏诊和误报各有多少", "Beyond accuracy: count misses and false alarms"),
    ("输出恶性的概率，边界是一条直线", "Outputs a probability; the boundary is straight"),
    ("没有闭式解，沿下坡一步步走", "No closed form: walk downhill step by step"),
    (r"新特征：到中心的距离 √\(x₁² \+ x₂²\)", "Distance from the centre √(x₁² + x₂²)"),
    ("每个点是一个训练肿块：上排蓝点是恶性（1），下排灰圈是良性（0），上下稍微错开以免重叠。先自己拖两个滑块，看能不能让直线既贴近两排点、又不跑出 0 到 1（红色区域）；再点“最小二乘直线”；最后勾选假想肿块，看最小二乘直线和切开的位置怎样变。",
     "Each point is a training lump: blue dots on top are malignant (1), grey circles below are benign (0), spread a little so they do not overlap. First drag the two sliders yourself: can a straight line stay close to both rows without leaving 0 to 1 (the red bands)? Then press \"Least-squares line\". Finally tick the imaginary lumps and watch how the least-squares line and its cut move."),
    ("左：压缩函数本身，横轴是直线的输出 z（可以是任何数），纵轴是压缩后的输出。右：用它压缩后，模型在训练肿块上画出的曲线。截断用的是最小二乘直线；阶跃和 S 形用的是同一条直线 z = wx + b。三个都试一遍：哪一个既在 0 到 1 之间，又说得出“有多确定”？",
     "Left: the squashing function itself; the horizontal axis is the line's output z (any number), the vertical axis is the squashed output. Right: the curve the model then draws over the training lumps. Clipping uses the least-squares line; the step and the S shape use the same line z = wx + b. Try all three: which one stays between 0 and 1 and can still say how sure it is?"),
    ("426 个训练肿块，每个的概率来自 5 折交叉验证（它所在的那一折没有参加训练）。横轴是模型算出的 z，绿色 S 形曲线把它压成概率 p。上排蓝点是恶性，下排灰圈是良性。阈值 t 是曲线上的一个高度，对应横轴上的一条竖线：竖线右边判为恶性。拖动阈值，竖线左右移动；被判错的点画上橙圈。z 小于 −10 或大于 10 的点画在两端，所以两端各堆成一条竖线。",
     "426 training lumps; each probability comes from 5-fold cross-validation (its own fold was not used for training). The horizontal axis is the model's z; the green S curve squashes it into a probability p. Blue dots on top are malignant, grey circles below are benign. The threshold t is a height on the curve, which matches one vertical line on the horizontal axis: everything to its right is called malignant. Drag the threshold and the line slides left or right; misclassified lumps get an orange ring. Points with z below −10 or above 10 are drawn at the ends, so each end shows a column of points."),
    ("输出终于在 0 到 1 之间，但有硬折角，不平滑；中间那条直线仍靠最小二乘去拟合，远处的点照样拉动它。", "The output is finally between 0 and 1, but it has sharp corners, so it is not smooth; and the line in the middle is still fitted by least squares, so far points still pull it."),
    ("只剩 0 和 1：说不出“有多确定”；参数稍微改一点，输出要么不变、要么突然跳，没法一步步调。", "Only 0 and 1 are left: it cannot say how sure it is. Change a parameter a little and the output either stays the same or jumps, so it cannot be tuned step by step."),
    ("处处在 0 到 1 之间；分界处是 0.5（拿不准），离得越远越确定；处处平滑，参数可以一点点调。", "Always between 0 and 1; 0.5 (not sure) at the boundary, and surer the further away; smooth everywhere, so the parameters can be tuned little by little."),
    ("加入 20 个半径 40 的假想肿块（都是恶性）", "Add 20 imaginary lumps of radius 40 (all malignant)"),
    ("模型算出的 z（右边越像恶性）", "z from the model (further right looks more malignant)"),
    ("预测落在 0 到 1 之外：", "Predictions outside 0 to 1: "),
    ("按 0.5 切开的位置：半径 ", "Cut at 0.5: radius "),
    ("阶跃（大于 0 就是 1）", "Step (1 if z > 0)"),
    (r"S 形：1 / \(1 \+ e⁻ᶻ\)", "S shape: 1 / (1 + e⁻ᶻ)"),
    ("截断到 0 到 1", "Clip to 0 to 1"),
    ("漏诊：", "Missed cancers: "),
    ("误报：", "False alarms: "),
    ("直线的输出 z", "Output of the line, z"),
    ("最小二乘直线", "Least-squares line"),
    ("判为恶性 →", "called malignant →"),
    ("← 判为良性", "← called benign"),
    ("压缩方式：", "Squash with: "),
    ("斜率 w", "Slope w"),
    ("截距 b", "Intercept b"),
    # regular expressions (numbers inside)
    (r"从 w = 0 出发", r"Start at w = 0"),
    (r"阈值 ([\d.]+)", r"threshold \1"),
    (r"ROC 曲线：AUC = ([\d.]+)（放大的左上角）", r"ROC curve: AUC = \1 (top-left corner, enlarged)"),
    ("误报率（误报数 / 良性总数）", "False alarm rate (false alarms / benign lumps)"),
    (r"开场的肿块：半径 ([\d.]+)", r"Opening lump: radius \1"),
    ("p ≥ 0.5：判为恶性", "p ≥ 0.5: called malignant"),
    (r"再加 20 个半径 (\d+) 的假想肿块", r"Plus 20 imaginary lumps of radius \1"),
    (r"两个原始特征：直线的准确率只有 ([\d.]+)", r"Straight line: accuracy \1"),
    (r"一个新特征：准确率 ([\d.]+)", r"Distance feature: accuracy \1"),
    (r"测试集 (\d+) 张：准确率 ([\d.]+)", r"\1 test images: accuracy \2"),
    (r"只用半径：准确率 ([\d.]+)", r"Radius only: accuracy \1"),
    (r"30 个特征：准确率 ([\d.]+)", r"30 features: accuracy \1"),
    (r"按 0\.5 切：([\d.]+)", r"Cut at 0.5: \1"),
    (r"p = ([\d.]+)：([\d.]+)", r"p = \1: \2"),
    (r"最低点 w = ([\d.]+)", r"Minimum at w = \1"),
    (r"良性（(\d+) 例）", r"Benign (\1)"),
    (r"恶性（(\d+) 例）", r"Malignant (\1)"),
    (r"第 (\d+) 步", r"Step \1"),
    (r"数字 (\d)", r"Digit \1"),
    # figure and widget phrases
    ("细胞核平均半径（标准化）", "Mean nucleus radius (standardised)"),
    ("细胞核纹理（标准化）", "Nucleus texture (standardised)"),
    ("细胞核平均半径", "Mean nucleus radius"),
    ("半径（标准化后）的权重 w", "Weight w of the radius (standardised)"),
    ("训练集上的平均交叉熵", "Mean training cross-entropy"),
    ("训练集平均交叉熵", "Mean training cross-entropy"),
    ("按 p = 0.5 判断的准确率", "Accuracy at p = 0.5"),
    ("标签 / 直线的预测", "Label / the line's prediction"),
    (r"平方误差 \(1 − p\)²", "Squared error (1 − p)²"),
    ("交叉熵 −log p", "Cross-entropy −log p"),
    ("抓到\n（真阳性）", "Caught\n(true positive)"),
    ("漏诊\n（假阴性）", "Missed\n(false negative)"),
    ("误报\n（假阳性）", "False alarm\n(false positive)"),
    ("排除\n（真阴性）", "Cleared\n(true negative)"),
    ("答案是类别", "The answer is a class"),
    ("模型：sigmoid", "Model: sigmoid"),
    ("损失：交叉熵", "Loss: cross-entropy"),
    ("求解：梯度下降", "Solve: gradient descent"),
    ("检验：基线与混淆矩阵", "Test: baseline and confusion matrix"),
    ("决策：阈值", "Decide: threshold"),
    ("自信地错了，罚得很重", "Confident mistakes cost a lot"),
    ("恶性的概率 p", "Probability of malignant, p"),
    ("判错", "wrong"),
    ("完整曲线，绿框是左边放大的部分", "Whole curve; the green box is the part enlarged"),
    ("恶性", "malignant"),
    ("良性", "benign"),
    ("这一例的损失", "Loss for this lump"),
    ("超出 0 到 1", "Outside 0 to 1"),
    ("预测的数字", "Predicted digit"),
    ("实际的数字", "Actual digit"),
    ("预测恶性", "Predicted\nmalignant"),
    ("预测良性", "Predicted\nbenign"),
    ("实际恶性", "Actually malignant"),
    ("实际良性", "Actually benign"),
    ("良性（0）", "Benign (0)"),
    ("恶性（1）", "Malignant (1)"),
    ("w₁（半径）", "w₁ (radius)"),
    ("w₂（纹理）", "w₂ (texture)"),
    ("训练数据", "Training data"),
    ("永远猜良性", "Always guess benign"),
    ("只用半径", "Radius only"),
    ("最多罚 1", "never more than 1"),
    ("准确率（训练集上的交叉验证）", "Accuracy (cross-validated, training set)"),
    ("肿块数", "Number of lumps"),
    ("阈值 t", "Threshold t"),
    ("召回率", "Recall"),
    ("精确率", "Precision"),
    ("准确率", "Accuracy"),
    ("误报率", "False alarm rate"),
    ("类别", "Class"),
    # punctuation left between translated pieces
    ("（", " ("), ("）", ")"), ("：", ": "), ("，", ", "), ("、", ", "),
]
_CJK = re.compile(r"[　-〿一-鿿＀-￯]")


def tr(s):
    for zh, en in sorted(PAIRS, key=lambda p: -len(p[0])):
        s = re.sub(zh, en, s) if "\\" in en or "(" in zh else s.replace(zh, en)
    return s


def check(s, where):
    left = _CJK.findall(s)
    if left:
        raise SystemExit(f"untranslated text in {where}: {''.join(left)[:40]}")
    return s
