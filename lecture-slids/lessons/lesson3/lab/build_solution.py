"""从当前 Worksheet 生成带题号、文字答案和执行输出的 Solution，并做三项检查：
1. 每个填空块的空数与 FILLS 一致；2. 每个 Answer 里的代码行都出现在填好的代码里；
3. 用真实 Jupyter 内核从第一格顺序执行，输出写回 lab_solution.ipynb，并断言关键核对值。
运行：uv run --no-project --python 3.12 --with scikit-learn,matplotlib,pandas,nbclient,nbformat,ipykernel python build_solution.py
（用 uv；~/.ipython 坏掉时先设 IPYTHONDIR 到一个可写的临时目录，见 COURSE_MATERIAL_WORKFLOW.md 第 5 节）
"""
from pathlib import Path
import re
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parent
# 每个含 ____ 的代码块按出现顺序给出填写内容
FILLS = [
    ["horsepower"],                   # 0.2
    ["2", "2"],                       # 1.1
    ["fit_transform", "transform"],   # 2.2
    ["0.01"],                         # 3.1
    ["Lasso"],                        # 3.2
    ["kf"],                           # 4.2
    ["Ridge"],                        # 4.3
    ["best_alpha_ridge"],             # 4.4
]
# 执行后必须出现在输出里的核对值（与 worksheet 的 Success check 一致）
EXPECT = [
    "(313, 6) (79, 6)", "40.606", "test RMSE: 4.2914", "'horsepower weight'", "7 1.89 7512.49",
    "test RMSE: 4.3610", "coefficients exactly zero: 0", "coefficients exactly zero: 4",
    "[3.43 4.31 5.24 4.59 4.28]", "(313, 27)", "position: 8", "test RMSE: 2.63",
    "True 11 11", "Lasso best alpha: 0.0025", "prediction for 225 hp: [14.76", "largest Ridge coefficient: 15.4",
]

qmd = (ROOT / "lab_worksheet.qmd").read_text()
core = qmd.split("# How today's lab helps with Assignment 1")[0]
fence = re.compile(r"```python\n(.*?)```", re.S)
answer_re = re.compile(r'::: \{[^\n]*title="Answer"[^\n]*\}\n(.*?)\n:::', re.S)
callout_re = re.compile(r"::: \{[^\n]*\}\n.*?\n:::", re.S)

nb = nbformat.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.cells.append(nbformat.v4.new_markdown_cell(
    "# Lab 3 Solution: Making the Model Stronger\n\n"
    "This solution follows the current worksheet task by task, including written answers. "
    "Run the Core tasks from top to bottom; the Advanced Study answers follow. "
    "The data URL is the same as in the worksheet and works in a blank Colab notebook. "
    "Try the worksheet before opening these answers."
))
fi = 0
for match in re.finditer(r"^### (\d\.\d [^\n]+)\n(.*?)(?=^### |^## |^# |\Z)", core, re.M | re.S):
    title, body = match.groups()
    answers = answer_re.findall(body)
    task_body = callout_re.sub("", body)              # drop Answer, Ask AI and Hint callouts
    nb.cells.append(nbformat.v4.new_markdown_cell("## " + title))
    filled_blocks = []
    for code in fence.findall(task_body):
        if "____" in code:
            assert code.count("____") == len(FILLS[fi]), f"Blank count changed in {title}"
            for value in FILLS[fi]:
                code = code.replace("____", value, 1)
            fi += 1
        filled_blocks.append(code)
        nb.cells.append(nbformat.v4.new_code_cell(code))
    for ans in answers:
        for code in fence.findall(ans):
            for line in code.strip().splitlines():
                assert not line.strip() or any(line.strip() in block for block in filled_blocks), (
                    f"Worksheet Answer differs from filled solution in {title}: {line}")
        prose = fence.sub("", ans).strip()
        if prose:
            nb.cells.append(nbformat.v4.new_markdown_cell(prose))
assert fi == len(FILLS), (fi, len(FILLS))

nb.cells.append(nbformat.v4.new_markdown_cell(
    "# Try with AI: one possible answer for each box\n\nYour AI may write it differently; what matters is that the Check in the worksheet passes."))
for section in (ROOT / "ai_examples.py").read_text().split("#%%"):
    section = section.strip()
    if section:
        nb.cells.append(nbformat.v4.new_code_cell(section + "\n"))

nb.cells.append(nbformat.v4.new_markdown_cell(
    "# Advanced Study\n\nRun the Core tasks first. A, B and C below are worked investigations, "
    "including the worksheet self-checks."))
titles = {"A": "Condition number and standardisation", "B": "Write Lasso yourself",
          "C": "Coefficient paths, and one more idea"}
for section in (ROOT / "as_solutions.py").read_text().split("#%%"):
    section = section.strip()
    if not section:
        continue
    for key, name in titles.items():
        if f"# ---- Advanced Study {key} ----" in section:
            nb.cells.append(nbformat.v4.new_markdown_cell(f"## {key}. {name}"))
    nb.cells.append(nbformat.v4.new_code_cell(section + "\n"))

# 强制使用执行本脚本的 Python 环境，避免命中其他环境的内核。
km = KernelManager(kernel_name="python3")
km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
NotebookClient(nb, timeout=600, kernel_manager=km,
               resources={"metadata": {"path": str(ROOT)}}).execute()

text = ""
images = 0
for cell in nb.cells:
    for out in cell.get("outputs", []):
        text += out.get("text", "") + str(out.get("data", {}).get("text/plain", ""))
        images += "image/png" in out.get("data", {})
        assert out.get("output_type") != "error", f"Error in cell: {cell.source[:80]}"
norm = lambda s: " ".join(s.split())                    # ignore spacing differences
missing = [e for e in EXPECT if norm(e) not in norm(text)]
assert not missing, f"Expected values not found in outputs: {missing}"
# worksheet 里直接印出的三张表（2.1 范围、4.1 两种分法、4.4 汇总）由这里重算并断言
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
# 4.3 的说法“按折标准化选出的 α 相同”由这里保证：用作业的 pipeline 写法再选一次
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
cars = pd.read_csv(ROOT.parent.parent / "lesson2" / "lab" / "auto_mpg.csv")
Xa, ya = cars.drop(columns="mpg"), cars["mpg"]
Xtr, _, ytr, _ = train_test_split(Xa, ya, test_size=0.2, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=0)
grid = np.logspace(-3, 2, 26)
per_fold = [-cross_val_score(make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), Ridge(alpha=a)),
                             Xtr, ytr, cv=kf, scoring="neg_root_mean_squared_error").mean() for a in grid]
assert abs(grid[int(np.argmin(per_fold))] - 0.0398) < 1e-3, grid[int(np.argmin(per_fold))]
hp = Xtr["horsepower"]
assert (hp.min(), hp.max(), (hp ** 3).min(), (hp ** 3).max()) == (46, 230, 97336, 12167000)
rows = []
for seed in (0, 1):
    a, b, ya_, yb_ = train_test_split(Xtr[["horsepower"]], ytr, test_size=0.2, random_state=seed)
    row = []
    for d in (1, 2, 6):
        p_, s_ = PolynomialFeatures(d, include_bias=False), StandardScaler()
        m_ = LinearRegression().fit(s_.fit_transform(p_.fit_transform(a)), ya_)
        row.append(round(mean_squared_error(yb_, m_.predict(s_.transform(p_.transform(b)))) ** 0.5, 2))
    rows.append(row)
assert rows == [[4.5, 3.43, 3.31], [4.72, 3.71, 4.05]], rows
_, Xte, _, yte = train_test_split(Xa, ya, test_size=0.2, random_state=42)
def _cvsd(model, X):
    s = -cross_val_score(model, X, ytr, cv=kf, scoring="neg_root_mean_squared_error")
    return round(s.mean(), 2), round(s.std(), 2)
def _test(model, Xf, Xt):
    return round(mean_squared_error(yte, model.fit(Xf, ytr).predict(Xt)) ** 0.5, 2)
P27, S27 = PolynomialFeatures(2, include_bias=False), StandardScaler()
Z27f = S27.fit_transform(P27.fit_transform(Xtr)); Z27t = S27.transform(P27.transform(Xte))
Q = lambda X: np.c_[X["horsepower"], X["horsepower"] ** 2]
table = [
    (_cvsd(LinearRegression(), Xtr[["horsepower"]]), _test(LinearRegression(), Xtr[["horsepower"]], Xte[["horsepower"]])),
    (_cvsd(LinearRegression(), Q(Xtr)), _test(LinearRegression(), Q(Xtr), Q(Xte))),
    (_cvsd(LinearRegression(), Z27f), _test(LinearRegression(), Z27f, Z27t)),
    (_cvsd(Ridge(alpha=0.0398107), Z27f), _test(Ridge(alpha=0.0398107), Z27f, Z27t)),
    (_cvsd(Lasso(alpha=0.00251189, max_iter=100000), Z27f), _test(Lasso(alpha=0.00251189, max_iter=100000), Z27f, Z27t)),
]
assert table == [((4.98, 0.42), 4.71), ((4.37, 0.58), 4.29), ((2.91, 0.23), 2.72),
                 ((2.88, 0.18), 2.64), ((2.9, 0.18), 2.61)], table
# 警告信息里带有本机的安装路径；公开的 Solution 不留本机路径
for cell in nb.cells:
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream":
            out["text"] = re.sub(r"/\S*/site-packages/", "", out["text"])
nbformat.write(nb, ROOT / "lab_solution.ipynb")
print(f"Validated {fi} blank blocks, all Core answers, {len(EXPECT)} check values and {images} figures; "
      f"wrote {len(nb.cells)} cells.")
print(text[text.find("1 4.61"):text.find("1 4.61") + 200])
print(text[text.find("standardised:"):text.find("standardised:") + 600])
