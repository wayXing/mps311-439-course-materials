"""English text for the Lesson 3 figures and widgets.
build_note_figures.py and build_widgets.py run once per language; with --lang en every
visible string passes through tr(), and any Chinese left over stops the build.
Order matters: longer phrases first, so a short phrase never breaks a longer one.
"""
import re

PAIRS = [
    # whole widget captions first, before any phrase inside them is replaced
    ("蓝点是用来拟合的 10 辆车，灰色空心圆是 79 辆测试车，绿色曲线是拟合结果。下方是各个系数（特征已标准化）的大小，纵轴是对数刻度，每格相差 10 倍；实心为正，空心为负。先预测，再把阶数一格一格往上拖。",
     "Blue dots are the 10 cars used for fitting, grey circles are the 79 test cars, and the green curve is the fit. Below are the sizes of the coefficients (features standardised) on a log scale, where each step is 10 times larger; filled bars are positive, hollow bars negative. Predict first, then move the degree up one step at a time."),
    ("同样的 10 辆车，阶数固定为 7，下方的系数图与上一个组件相同。滑块最左端是 λ = 0，即上一个组件里 7 阶的那条曲线；向右每 8 格，λ 增大 10 倍。",
     "The same 10 cars, with the degree fixed at 7; the coefficient chart below is the same as in the previous widget. The far left of the slider is λ = 0, the degree-7 curve from the previous widget. Every 8 steps to the right, λ grows 10 times."),
    # regular expressions (numbers inside)
    (r"交叉验证选的 λ\n保留 (\d+) 项", r"λ chosen by CV\nkeeps \1 terms"),
    (r"在第 (\d+) 份上算误差", r"error on fold \1"),
    (r"第 (\d+) 轮", r"Round \1"),
    (r"选 (\d+) 阶", r"choose degree \1"),
    (r"(\d+) 项", r"\1 terms"),
    # figures
    ("313 辆训练车分成 5 份", "313 training cars split into 5 folds"),
    ("313 辆训练车", "313 training cars"),
    ("马力 hp（五组，各组中位数）", "Horsepower (five groups, group medians)"),
    ("马力直线", "Horsepower line"),
    ("直线预测 ", "Line predicts "),
    ("实测 ", "Actual "),
    ("马力 hp", "Horsepower (hp)"),
    ("加入 hp² 后", "With hp² added"),
    ("左：两种模型", "Left: two models"),
    ("平均残差（mpg）", "Mean residual (mpg)"),
    ("右：各组的平均残差", "Right: mean residual by group"),
    ("各小时的平均租车量（数据）", "Mean rentals in each hour (data)"),
    ("特征：小时本身", "Feature: hour as a number"),
    ("特征：sin、cos", "Features: sin, cos"),
    ("再加一组 sin、cos", "Plus a second sin, cos pair"),
    ("一天中的小时", "Hour of the day"),
    ("每小时租车量", "Rentals per hour"),
    ("测试集\n79 辆\n\n最后\n只用一次", "Test set\n79 cars\n\nused once,\nat the end"),
    ("用来验证（这一轮不参与拟合）", "used to validate (not used to fit in this round)"),
    ("用来拟合", "used to fit"),
    ("交叉验证 RMSE（± 一个标准误）", "Cross-validation RMSE (± one standard error)"),
    ("多项式阶数（只用马力）", "Polynomial degree (horsepower only)"),
    ("左：选阶数", "Left: choosing the degree"),
    ("Ridge 的交叉验证 RMSE", "Ridge cross-validation RMSE"),
    ("λ（对数刻度，27 个特征）", "λ (log scale, 27 features)"),
    ("λ（对数刻度，向右惩罚更强）", "λ (log scale, stronger penalty to the right)"),
    ("右：选 λ", "Right: choosing λ"),
    ("系数（标准化后）", "Coefficient (standardised)"),
    ("不加惩罚的解", "Unpenalised solution"),
    ("加惩罚后的解", "Penalised solution"),
    ("不加惩罚 ", "No penalty "),
    # flow diagram
    ("模型不够强", "Model not strong enough"),
    ("残差有规律、误差大", "Residuals show a pattern, errors are large"),
    ("多项式、sin/cos、比率、log、交互", "Polynomials, sin/cos, ratios, log, interactions"),
    ("特征工程", "Feature engineering"),
    ("放到同一尺度，只用训练数据算", "Same scale, computed on training data only"),
    ("标准化", "Standardisation"),
    ("太灵活：过拟合", "Too flexible: overfitting"),
    ("训练误差低、新数据误差高，系数爆炸", "Low training error, high error on new data, huge coefficients"),
    ("正则化", "Regularisation"),
    ("Ridge 缩小、Lasso 置零、Elastic Net 兼顾", "Ridge shrinks, Lasso sets to zero, Elastic Net does both"),
    ("选阶数与 λ，不碰测试集", "Choose degree and λ without the test set"),
    ("测试集读一次", "Read the test set once"),
    ("报告最终结果和局限", "Report the final result and its limits"),
    ("交叉验证", "Cross-validation"),
    # widgets
    ("训练 RMSE（10 辆车）", "Training RMSE (10 cars)"),
    ("测试 RMSE（79 辆新车）", "Test RMSE (79 new cars)"),
    ("系数的大小", "Size of the coefficients"),
    ("训练 RMSE", "Training RMSE"),
    ("测试 RMSE", "Test RMSE"),
    ("大于 100", "over 100"),
    ("|系数|", "|coefficient|"),
    ("阶数 ", "Degree "),
    ("直线", "Straight line"),
    # punctuation left between translated pieces
    ("（", " ("), ("）", ")"), ("：", ": "), ("、", ", "),
]
_CJK = re.compile(r"[　-〿一-鿿＀-￯]")


def tr(s):
    for zh, en in PAIRS:
        s = re.sub(zh, en, s) if "\\" in en or "(" in zh else s.replace(zh, en)
    return s


def check(s, where):
    left = _CJK.findall(s)
    if left:
        raise SystemExit(f"untranslated text in {where}: {''.join(left)[:40]}")
    return s
