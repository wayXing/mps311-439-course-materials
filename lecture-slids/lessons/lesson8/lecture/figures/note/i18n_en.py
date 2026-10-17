"""English text for the Lesson 8 figures and widgets.
build_note_figures.py and build_widgets.py run once per language; with --lang en (or --slides) every
visible string passes through tr(), and any Chinese left over stops the build.
tr() replaces the longest Chinese strings first, and punctuation last.
"""
import re

PAIRS = [
    ("群内散布\n（距离平方之和）", "Within-cluster\nsum of squares"),
    ("（Ward 距离）", " (Ward distance)"),
    ("（Ward）", " (Ward)"),
    ("（空格为 0）", " (blank = 0)"),
    ("+ 蓝、红：两群各自的中心；+ 绿：合并后的中心", "+ blue, red: the centre of each group; + green: the centre after merging"),
    ("十字是三个中心。先点 ①：每个点涂成离它最近的中心的颜色。再点 ②：每个中心挪到自己那群点的平均位置。交替点，看群内散布（每个点到自己中心的距离平方之和）怎样变。“换个起点”在两个起点之间切换。",
     "The crosses are the three centres. Press ① first: each point takes the colour of its nearest centre. Then press ②: each centre moves to the average position of its group's points. Press them in turn and watch the within-cluster sum of squares (the squared distance of every point to its own centre, added up). \"Try another start\" switches between two starting points."),
    ("每种方法都分成 2 群，颜色是它分出的群。右边是找回的比例：每一群按它里面最多的那一组算对。“最近的两点”：两群里最近的两个点有多近；Ward：合并后群内散布增加多少。",
     "Every method splits the points into 2 groups; the colours are its groups. On the right is the share recovered: each group counts as right for its most common true group. \"Nearest points\": how close the two nearest points of the two groups are. Ward: how much the within-cluster sum of squares grows when they merge."),
    ("K-means（k = 10）：多数投票找回", "K-means (k = 10): majority vote recovers"),
    ("逐步合并最近的两群；“多近”怎么量会改变结果", "Merge the two nearest groups step by step; how \"near\" is measured changes the result"),
    ("分给最近的中心 ↔ 中心挪到平均，交替到停", "Assign to the nearest centre ↔ move centres to the mean, until nothing changes"),
    ("计数表、多数投票；数据太多先子采样", "Count table, majority vote; subsample when there is too much data"),
    ("离同一个中心近；群内散布越小越好", "Close to one centre; the smaller the within-cluster sum of squares, the better"),
    ("环、长带：按中心切，切线是直的", "Rings, strips: cutting by centres gives straight cuts"),
    ("树状图：每次合并最近的两群", "Dendrogram: each step merges the two nearest groups"),
    ("只有点：有没有自然的几群？", "Only points: are there natural groups?"),
    ("群内散布（距离平方之和）", "Within-cluster sum of squares"),
    ("合并时两群的距离", "Distance between the two groups at the merge"),
    ("k = 3 之后下降变慢", "After k = 3 the drop slows down"),
    ("真实的数字（聚类时没用）", "True digit (not used for clustering)"),
    ("180 个点，没有标签", "180 points, no labels"),
    ("从三团里各取几个点（", "A few points from each of the three blobs ("),
    ("在这里横切：3 群", "Cut here: 3 groups"),
    ("层次聚类：最近的两点", "Hierarchical: nearest points"),
    ("层次聚类：complete", "Hierarchical: complete"),
    ("层次聚类：average", "Hierarchical: average"),
    ("层次聚类：最近两点（single）", "Hierarchical: nearest points (single)"),
    ("层次聚类：single", "Hierarchical: single"),
    ("层次聚类：Ward", "Hierarchical: Ward"),
    ("：停在 ", ": stops at "),
    ("第几轮", "Round"),
    ("一种分法：每个点连到自己的中心", "One grouping: each point joined to its centre"),
    ("最近的两点：这条线", "Nearest points: this line"),
    ("Ward：合并后散布增加多少", "Ward: growth in within-cluster sum of squares"),
    ("K-means（k = 10）的计数表", "Count table of K-means (k = 10)"),
    ("24 个点", "24 points"),
    ("合并高度", "Merge height"),
    ("树状图", "Dendrogram"),
    ("中心不再移动：停下了。", "The centres no longer move: it has stopped."),
    ("① 分给最近的中心", "① Assign to the nearest centre"),
    ("② 中心挪到平均", "② Move centres to the mean"),
    ("多数投票准确率", "Majority-vote accuracy"),
    ("找回原来的两组", "Share of the two true groups recovered"),
    ("已做的步数：", "Steps done: "),
    ("一个群吞掉", "one group swallows"),
    ("群的个数 k", "Number of groups k"),
    ("什么叫一群", "What is a group"),
    ("失败的形状", "Shapes that fail"),
    ("\n多数：", "\nmost: "),
    ("检查与规模", "Check and scale"),
    ("换个起点", "Try another start"),
    ("群内散布（对数刻度）", "Sum of squares (log scale)"),
    ("群内散布", "Within-cluster sum of squares"),
    ("层次聚类", "Hierarchical clustering"),
    ("没有标签", "No labels"),
    ("两条带", "Two strips"),
    ("两个环", "Two rings"),
    ("形状：", "Shape: "),
    ("方法：", "Method: "),
    (" 张图", " images"),
    ("起点 ", "Start "),
    (" 个）", ")"),
    ("群 ", "Group "),
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
