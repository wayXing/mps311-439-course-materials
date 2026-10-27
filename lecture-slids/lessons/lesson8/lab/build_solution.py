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
# 课程的 MNIST 子集在公共镜像推送之前，公开网址上还没有：执行时改读本地同一个文件，写出时换回公开网址
DATA_URL = "https://raw.githubusercontent.com/wayXing/mps311-439-course-materials/main/lecture-slids/lessons/data/mnist10k.npz"
LOCAL_DATA = (ROOT / "../../data/mnist10k.npz").resolve().as_uri()
# 每个含 ____ 的代码块按出现顺序给出填写内容
FILLS = [
    ["10"],               # 1.1
    ["y_digits_fit"],     # 2.1
    ['"ward"'],           # 4.1 (the worksheet also asks for "single"; the solution shows ward and adds single below)
]
# 执行后必须出现在输出里的核对值（与 worksheet 的 Success check 一致）
EXPECT = [
    "(5625, 50)", "169025.6", "5625", "[0 0 1 1 2 3 4 6 7 8]", "0.576", "0 169033.3 0.586", "1 169714.5 0.599", "2 169349.4 0.594",
    "3 169025.8 0.576", "4 169026.9 0.581", "0.597", "single: 0.115", "k = 10: 169026", "500 images, majority vote: 0.592",
    "best 162656.2 worst 164594.3", "scikit-learn, k = 12: 162836.0", "eps 0.5 groups",
]
qmd = (ROOT / "lab_worksheet.qmd").read_text()
core = qmd.split("# After class: how today's lab helps with Assignment 2")[0]
fence = re.compile(r"```python\n(.*?)```", re.S)
answer_re = re.compile(r'::: \{[^\n]*title="Answer"[^\n]*\}\n(.*?)\n:::', re.S)
callout_re = re.compile(r"::: \{[^\n]*\}\n.*?\n:::", re.S)

nb = nbformat.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.cells.append(nbformat.v4.new_markdown_cell(
    "# Lab 8 Solution: Finding Groups Without Labels\n\n"
    "This solution follows the worksheet task by task, including written answers. "
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

nb.cells.append(nbformat.v4.new_code_cell(
    "# 4.1 again, with linkage=\"single\"\n"
    "hc = AgglomerativeClustering(n_clusters=10, linkage=\"single\")\n"
    "groups_hc = hc.fit_predict(Z_fit)\n"
    "t = np.zeros((10, 10), dtype=int)\n"
    "for i in range(len(Z_fit)):\n"
    "    t[groups_hc[i], y_digits_fit[i]] += 1\n"
    "print(\"single:\", round(t.max(axis=1).sum() / len(Z_fit), 3))\n"
    "print(np.bincount(groups_hc))\n"))
nb.cells.append(nbformat.v4.new_markdown_cell(
    "# Try with AI: one possible answer for each box\n\nYour AI may write it differently; what matters is that the Check in the worksheet passes."))
for section in (ROOT / "ai_examples.py").read_text().split("#%%"):
    section = section.strip()
    if section:
        nb.cells.append(nbformat.v4.new_code_cell(section + "\n"))

nb.cells.append(nbformat.v4.new_markdown_cell(
    "# Advanced Study\n\nRun the Core tasks first. A, B and C below are worked investigations, "
    "including the worksheet self-checks."))
titles = {"A": "K-means in numpy, from ten starts", "B": "How many groups?", "C": "Groups by density"}
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
