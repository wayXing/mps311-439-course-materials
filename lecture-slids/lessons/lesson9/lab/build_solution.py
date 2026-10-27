"""从当前 Worksheet 生成带题号、文字答案和执行输出的 Solution，并做三项检查：
1. 每个填空块的空数与 FILLS 一致；2. 每个 Answer 里的代码行都出现在填好的代码里；
3. 用真实 Jupyter 内核从第一格顺序执行，输出写回 lab_solution.ipynb，并断言关键核对值。
运行：python build_solution.py   （用装了 TensorFlow 的 Python；本机见生产流程第 5 节的 arm64 虚拟环境）
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
    ['"sigmoid"'],                    # 1.1
    ["128", '"relu"', '"softmax"'],   # 2.1
    ['"linear"', '"linear"'],         # 3.1
]
# 执行后必须出现在输出里的核对值（与 worksheet 的 Success check 一致）
EXPECT = [
    "hidden units 1 accuracy 0.69", "hidden units 2 accuracy 0.82", "hidden units 3 accuracy 0.97", "hidden units 4 accuracy 0.997", "hidden units 1 accuracy 0.503", "hidden units 4 accuracy 0.503",
    "(50000, 784) (10000, 784)", "[5 0 4 1 9 2 1 3 1 4]", "118282", "100480 16512 1290 118282", "7850",
    "0.9645 0.9716 0.9741 0.9743 0.97", "0.9159 0.9176 0.919 0.9205 0.9203", "4 0.093", "lowest validation loss at epoch 7 0.078", "final gap (val loss - train loss): 0.016 without dropout: 0.091", 
    "early stop after 7 epochs", "diagonal 9743", "loss 0.651 -> 0.019 accuracy 0.997", "agree: True",
]
qmd = (ROOT / "lab_worksheet.qmd").read_text()
core = qmd.split("# After class: how today's lab helps with Assignment 2")[0]
fence = re.compile(r"```python\n(.*?)```", re.S)
answer_re = re.compile(r'::: \{[^\n]*title="Answer"[^\n]*\}\n(.*?)\n:::', re.S)
callout_re = re.compile(r"::: \{[^\n]*\}\n.*?\n:::", re.S)

nb = nbformat.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.cells.append(nbformat.v4.new_markdown_cell(
    "# Lab 9 Solution: Letting the Model Make Its Own Features\n\n"
    "This solution follows the worksheet task by task, including written answers. "
    "Run the Core tasks from top to bottom; the Advanced Study answers follow. "
    "MNIST comes with Keras (about 11 MB), so the notebook runs in a blank Colab notebook. "
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
        raw = code
        if "____" in code:
            assert code.count("____") == len(FILLS[fi]), f"Blank count changed in {title}"
            for value in FILLS[fi]:
                code = code.replace("____", value, 1)
            fi += 1
        filled_blocks.append(code)
        nb.cells.append(nbformat.v4.new_code_cell(code))
        if title.startswith("1.1"):
            ring_linear = raw.replace("____", '"linear"', 1)  # 1.2 Question 2: the same cell without the squeeze
        if title.startswith("1.2"):
            ring_plot = code
    if title.startswith("1.2"):                        # run Question 2, then put "sigmoid" back
        nb.cells.append(nbformat.v4.new_markdown_cell('Question 2: 1.1 and 1.2 again with `"linear"` in the blank.'))
        nb.cells.append(nbformat.v4.new_code_cell(ring_linear))
        nb.cells.append(nbformat.v4.new_code_cell(ring_plot))
    for ans in answers:
        for code in fence.findall(ans):
            for line in code.strip().splitlines():
                assert not line.strip() or any(" ".join(line.split()) in " ".join(block.split()) for block in filled_blocks), (
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
titles = {"A": "A one-hidden-layer network in numpy", "B": "Check one gradient by hand", "C": "Dropout"}
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
NotebookClient(nb, timeout=1200, kernel_manager=km,
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
nbformat.write(nb, ROOT / "lab_solution.ipynb")
print(f"Validated {fi} blank blocks, all Core answers, {len(EXPECT)} check values and {images} figures; "
      f"wrote {len(nb.cells)} cells.")
