"""从当前 Worksheet 生成带题号、文字答案和执行输出的 Solution，并做三项检查：
1. 每个填空块的空数与 FILLS 一致；2. 每个 Answer 里的代码行都出现在填好的代码里；
3. 用真实 Jupyter 内核从第一格顺序执行，输出写回 lab_solution.ipynb，并断言关键核对值。
运行：uv run --no-project --python 3.12 --with scikit-learn==1.6.1,matplotlib,pandas,scipy,nbclient,nbformat,ipykernel python build_solution.py
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
    ["0.25"],          # 0.1
    ["1"],             # 1.1
    ["35", "84"],      # 1.2
    ["X_digits_val", "y_digits_val"],   # 2.2
    ["0"],             # 3.1 (the call read off the depth-2 tree)
    ["15"],            # 3.2
]
# 执行后必须出现在输出里的核对值（与 worksheet 的 Success check 一致）
EXPECT = [
    "(319, 30) (107, 30) (143, 30)", "0\n15.045", "228 35", "91 84", "0.26 0.142 0.226", "5625 1875 2500",
    "1 0.195 0.195", "15 0.996 0.781", "20 1.0 0.771",
    "mean concave points 0.717", "worst concavity 0.069", "worst radius 0.061", "mean texture 0.034", "worst area 0.025",
    "mean radius 0.017 mean perimeter 0.0", "correlation of radius and perimeter: 0.998", "your call: 0 the tree's call: 0", "0.776",
    "highest: Gini 0.5 entropy 1.0", "best validation: 0.781 at depth 15",
    "left node, radius: best cut 13.405 weighted Gini 0.218", "left node, texture: best cut 18.635", "tree, left node: 0 13.405", "fewest errors of any cut: 35 of 228 ; without a cut: 35",
    "mean concave points", "worst perimeter", "worst concave points", "random forest, validation accuracy:",
]
qmd = (ROOT / "lab_worksheet.qmd").read_text()
core = qmd.split("# After class: how today's lab helps with Assignment 2")[0]
fence = re.compile(r"```python\n(.*?)```", re.S)
answer_re = re.compile(r'::: \{[^\n]*title="Answer"[^\n]*\}\n(.*?)\n:::', re.S)
callout_re = re.compile(r"::: \{[^\n]*\}\n.*?\n:::", re.S)

nb = nbformat.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.cells.append(nbformat.v4.new_markdown_cell(
    "# Lab 6 Solution: A Chain of Yes-or-No Questions\n\n"
    "This solution follows the current worksheet task by task, including written answers. "
    "Run the Core tasks from top to bottom; the Advanced Study answers follow. "
    "The data come with scikit-learn, so the notebook runs in a blank Colab notebook. "
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
titles = {"A": "Find the best cut yourself", "B": "How stable is a tree?", "C": "Many trees"}
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
