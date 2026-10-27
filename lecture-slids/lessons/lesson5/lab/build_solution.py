"""从当前 Worksheet 生成带题号、文字答案和执行输出的 Solution，并做三项检查：
1. 每个填空块的空数与 FILLS 一致；2. 每个 Answer 里的代码行都出现在填好的代码里；
3. 用真实 Jupyter 内核从第一格顺序执行，输出写回 lab_solution.ipynb，并断言关键核对值。
运行：uv run --no-project --python 3.12 --with "scikit-learn==1.6.1",matplotlib,pandas,scipy,nbclient,nbformat,ipykernel python build_solution.py
（scikit-learn 固定 1.6.1：1.8 起 QDA 遇到不满秩的协方差直接报 LinAlgError，2.2、3.1、3.2 会停下）
（用 uv；~/.ipython 坏掉时先设 IPYTHONDIR 到一个可写的临时目录，见 COURSE_MATERIAL_WORKFLOW.md 第 5 节）
"""
from pathlib import Path
import re
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parent
# 课程的 MNIST 子集在公共镜像推送之前，公开网址上还没有：执行时改读本地同一个文件，写出时换回公开网址
DATA_URL = "https://raw.githubusercontent.com/wayXing/mps311-439-course-materials/main/lecture-slids/lessons/data/mnist10k.npz"
LOCAL_DATA = (ROOT / "../../data/mnist10k.npz").resolve().as_uri()
# 每个含 ____ 的代码块按出现顺序给出填写内容
FILLS = [
    ["1", "1", "0", "0"],                                             # 1.1
    ["1", "0"],                                                       # 1.2
    ["LinearDiscriminantAnalysis"],                                   # 2.1
    ["0.001"],                                                        # 3.3
]
# 执行后必须出现在输出里的核对值（与 worksheet 的 Success check 一致）
EXPECT = [
    "(426, 30) 159", "17.35", "12.17", "0.1139", "0.9788", "0.96", "0.9576",
    "0 0.712", "0.0001 0.843", "0.001 0.863", "0.01 0.784", "0.1 0.557", "0.5 0.505", "1 0.804", "0.8608",
    "heights at 12.8:", "0.7194444444444444", "0.8666666666666667",
    "test accuracy, mine: 0.958  sklearn: 0.958", "rank of the 784 x 784 matrix: 471", "largest weight: 19.2  intercept: -2.45 -2.45",
]
qmd = (ROOT / "lab_worksheet.qmd").read_text()
core = qmd.split("# After class: how today's lab helps with Assignment 2")[0]
fence = re.compile(r"```python\n(.*?)```", re.S)
answer_re = re.compile(r'::: \{[^\n]*title="Answer"[^\n]*\}\n(.*?)\n:::', re.S)
callout_re = re.compile(r"::: \{[^\n]*\}\n.*?\n:::", re.S)

nb = nbformat.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.cells.append(nbformat.v4.new_markdown_cell(
    "# Lab 5 Solution: Describe Each Class, Then Decide\n\n"
    "This solution follows the current worksheet task by task, including written answers. "
    "Run the Core tasks from top to bottom; the Advanced Study answers follow. "
    "The data come with scikit-learn, so the notebook runs in a blank Colab notebook. "
    "Try the worksheet before opening these answers."
))
fi = 0
for match in re.finditer(r"^### (\d\.\d [^\n]+)\n(.*?)(?=^### \d|^## Part |\Z)", core, re.M | re.S):  # 不在 "# " 处断开：代码里的注释行也以 "# " 开头
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
titles = {"A": "A Gaussian classifier in numpy", "B": "Why one shared spread gives a straight line", "C": "Two more ideas"}
for section in (ROOT / "as_solutions.py").read_text().split("#%%"):
    section = section.strip()
    if not section:
        continue
    for key, name in titles.items():
        if f"# ---- Advanced Study {key} ----" in section:
            nb.cells.append(nbformat.v4.new_markdown_cell(f"## {key}. {name}"))
    nb.cells.append(nbformat.v4.new_code_cell(section + "\n"))

# 强制使用执行本脚本的 Python 环境，避免命中其他环境的内核；~/.ipython 在本机是坏链接，改用临时目录。
import os, tempfile
_tmp = tempfile.mkdtemp()
os.environ.setdefault("IPYTHONDIR", _tmp)
os.environ.setdefault("JUPYTER_RUNTIME_DIR", _tmp)
km = KernelManager(kernel_name="python3")
km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
for cell in nb.cells:
    cell.source = cell.source.replace(DATA_URL, LOCAL_DATA)
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
assert not missing, f"Expected values not found in outputs: {missing}\n{text[:3000]}"
# 3.1 里给出的 logistic regression 对照值、Advanced Study A 的“差 0 个”由这里重算
import warnings
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")
import numpy as np
_d = np.load(ROOT / "../../data/mnist10k.npz")
_a, _b, _c, _e = train_test_split(_d["X"] / 255, _d["y"], test_size=0.25, random_state=0, stratify=_d["y"])
assert round(cross_val_score(LogisticRegression(max_iter=1000, C=0.1), _a, _c, cv=KFold(5, shuffle=True, random_state=0)).mean(), 3) == 0.901
assert "\n0\n" in "\n" + text.replace("\r", "") + "\n" or "0\ntest accuracy" in text
# 警告信息里带有本机的安装路径；公开的 Solution 不留本机路径
for cell in nb.cells:
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream":
            out["text"] = re.sub(r"/\S*/site-packages/", "", out["text"])
for cell in nb.cells:
    cell.source = cell.source.replace(LOCAL_DATA, DATA_URL)
nbformat.write(nb, ROOT / "lab_solution.ipynb")
print(f"Validated {fi} blank blocks, all Core answers, {len(EXPECT)} check values and {images} figures; "
      f"wrote {len(nb.cells)} cells.")
