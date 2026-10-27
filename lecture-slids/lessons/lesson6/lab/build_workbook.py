"""从当前 Worksheet 生成学生用的 Colab workbook（lab_workbook.ipynb），并检查它能跑通：
1. 代码全部放进 notebook，学生不用复制；只留 ML 判断的空（本讲没有纯 API 的空，PREFILL 为空）；
2. 每个 Question 后面加一个“Your answer”文字格，Question first 的格子在代码之前，先写判断再运行；
3. 不放 Answer（答案只在网页 worksheet 里，折叠）；
4. 检查：把空按 build_solution.py 的 FILLS 填好，用真实 Jupyter 内核从头执行，断言 Core 的核对值。
   MNIST 在执行检查时改读本地 ../../data/mnist10k.npz，写出的 workbook 保留公开网址（同 build_solution.py）。
写出的 workbook 不含输出。
运行：uv run --no-project --python 3.12 --with scikit-learn==1.6.1,matplotlib,pandas,scipy,nbclient,nbformat,ipykernel python build_workbook.py
（用 uv；~/.ipython 坏掉时先设 IPYTHONDIR 到一个可写的临时目录，见 COURSE_MATERIAL_WORKFLOW.md 第 5 节）
"""
from pathlib import Path
import os
import re
import sys
import tempfile
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parent
# 不需要 ML 判断的空直接给出。本讲的六个空都是 ML 判断（验证集比例、深度、读计数、验证集、读树、选定深度），不预填
PREFILL = {}
# Question first 的任务：代码下面的 Success check 会泄露答案，在 workbook 里折叠起来
# 有空的任务全部折叠，全篇一致：先自己填、运行，再打开核对（2026-10-27 评审）。
# 2.2 的 Success check 会泄露 Question first 的答案；3.1 的会泄露读图的那个空。
FOLD_CHECK = {"0.1", "1.1", "1.2", "2.2", "3.1", "3.2"}
# 各空按出现顺序的填写内容，只用于检查（与 build_solution.py 的 FILLS 相同）
FILLS = [["0.25"], ["1"], ["35", "84"], ["X_digits_val", "y_digits_val"], ["0"], ["15"]]
EXPECT = [
    "(319, 30) (107, 30) (143, 30)", "0\n15.045", "228 35", "91 84", "0.26 0.142 0.226", "5625 1875 2500",
    "1 0.195 0.195", "15 0.996 0.781", "20 1.0 0.771",
    "mean concave points 0.717", "worst concavity 0.069", "worst radius 0.061", "mean texture 0.034", "worst area 0.025",
    "mean radius 0.017 mean perimeter 0.0", "correlation of radius and perimeter: 0.998",
    "your call: 0 the tree's call: 0", "0.776",
]
WORKSHEET_URL = "https://ml.wxing.me/materials/lesson6/lab.html"
# 课程的 MNIST 子集：检查时读本地同一个文件，写出的 workbook 保留公开网址
DATA_URL = "https://raw.githubusercontent.com/wayXing/mps311-439-course-materials/main/lecture-slids/lessons/data/mnist10k.npz"
LOCAL_DATA = (ROOT / "../../data/mnist10k.npz").resolve().as_uri()

qmd = (ROOT / "lab_worksheet.qmd").read_text()
core = qmd.split("## Part 0:")[1].split("# After class: how today's lab helps with Assignment 2")[0]
core = "## Part 0:" + core
# The worksheet's opening, word for word: what the lab is for, and the four steps of Using AI
overview = qmd.split("## What this lab is for\n")[1].split("\n## How to work")[0].strip()
using_ai = qmd.split("### Using AI\n")[1].split("\n## Part 0:")[0].strip()
fence = re.compile(r"```python\n(.*?)```", re.S)
callout = re.compile(r'::: \{[^\n]*title="([^"]+)"[^\n]*\}\n(.*?)\n:::', re.S)

md = nbformat.v4.new_markdown_cell
code_cell = nbformat.v4.new_code_cell
nb = nbformat.v4.new_notebook()
nb.metadata.update({
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
    "colab": {"provenance": []},
})
nb.cells += [
    md("# Lab 6 Workbook: A Chain of Yes-or-No Questions\n## MPS311/439 Machine Learning\n\n"
       "The same tasks as the [Lab 6 worksheet](" + WORKSHEET_URL + "), with the code already in place: **nothing to copy**.\n\n"
       "**What you do today.** The Python is given. What is left to you are the machine-learning decisions: "
       "the blanks `____` and the questions. That is the lab. A cell with a blank stops with an error until you fill it.\n\n"
       "After the four Parts come the after-class part and the **Advanced Study** (MPS439: investigation A is required). "
       "Only the folded Answers are not here: they stay on the [worksheet](" + WORKSHEET_URL + ")."),
    md("## What this lab is for\n\n" + overview),
    md("## How to work\n\n"
       "1. **File → Save a copy in Drive**, and work in the copy so your answers are saved.\n"
       "2. Run the cells from top to bottom (**▶** or **Shift + Enter**); each cell needs the cells above it.\n"
       "3. Where you see **Your answer**, double-click the cell and write your answer **before** you run the next code cell. "
       "A guess written down first is how you find out what you really expected.\n"
       "4. Fill each `____`, run the cell, and compare with the **Success check** below it.\n"
       "5. Stuck on a blank? Open the folded Hint. The folded Answers are on the [worksheet](" + WORKSHEET_URL + "); "
       "if you fall behind, copy one and carry on.\n"
       "6. The folded **Try with AI** boxes are for when you finish early, or after class."),
]
ai_boxes = 0                                                 # the first box also explains the four steps


def your_answer():
    return md("**Your answer:** *(double-click to write)*")


def add_text(text):
    """Markdown 段：非空就成一格；含 Question 的段后面跟一个“Your answer”格。"""
    text = text.strip()
    if not text:
        return
    start = text.find("**Question")
    if start > 0:                                            # 问题单独成格，不和前面的 Hint、Success check 挤在一格
        nb.cells.append(md(text[:start].strip()))
        text = text[start:]
    nb.cells.append(md(text))
    if "**Question" in text:
        nb.cells.append(your_answer())


def add_body(body, task):
    """按顺序把一段正文拆成文字格和代码格。"""
    pos = 0
    for block in fence.finditer(body):
        add_text(body[pos:block.start()])
        source = block.group(1).rstrip("\n")
        for old, new in PREFILL.get(task, []):
            source = source.replace(old, new)
        nb.cells.append(code_cell(source))
        pos = block.end()
    add_text(body[pos:])


def add_ai_box(name, inner):
    """一个折叠的文字格；框里若有代码（数字图那一框），代码放在折叠格下面的代码格里。"""
    global ai_boxes
    ai_boxes += 1
    text = ("**Using AI: the four steps**\n\n" + using_ai + "\n\n---\n\n") if ai_boxes == 1 else ""
    block = fence.search(inner)
    if block:
        text += inner[:block.start()].replace("Run this cell first, without changing it.",
                                             "Run the code cell just below this box first, without changing it.")
        text += inner[block.end():]
    else:
        text += inner
    nb.cells.append(md(f"<details><summary><b>{name}</b></summary>\n\n" + text.strip() + "\n\n</details>"))
    if block:
        nb.cells.append(code_cell("# Optional: the code for the Try with AI box above\n" + block.group(1).rstrip("\n")))


for part in re.split(r"^(?=## Part )", core, flags=re.M):
    if not part.strip():
        continue
    head, _, rest = part.partition("\n")
    intro, *tasks = re.split(r"^(?=### )", rest, flags=re.M)
    nb.cells.append(md(head.replace("## ", "# ", 1) + "\n\n" + intro.strip()))
    for task_text in tasks:
        title, _, body = task_text.partition("\n")
        task = title[4:7]                                    # 例如 "1.2"
        nb.cells.append(md(title.replace("### ", "## ", 1)))
        if task in PREFILL:
            body = re.sub(r"\*\*Hint:\*\*[^\n]*\n", "", body)
            for old, new in PREFILL[task]:
                body = body.replace(old, new)
        # Hints often give the blank away: fold them, so students try first
        body = re.sub(r"\*\*Hint:\*\* ([^\n]+)\n",
                      r"<details><summary><b>Hint</b> (only if you are stuck)</summary>\n\n\1\n\n</details>\n", body)
        if task in FOLD_CHECK:                               # "**Success check:** …" on one line
            body = re.sub(r"\*\*Success check:\*\* ([^\n]+)\n",
                          r"<details><summary><b>Success check</b> (open after you run the cell)</summary>\n\n\1\n\n</details>\n", body)
        pos = 0
        for box in callout.finditer(body):
            add_body(body[pos:box.start()], task)
            name, inner = box.groups()
            if name.startswith("Try with AI"):              # folded, so the Core reads without AI
                add_ai_box(name, inner)
            elif name != "Answer":                           # 答案不进 workbook
                nb.cells.append(md(f"**{name}**"))
                add_body(inner, task)
            pos = box.end()
        add_body(body[pos:], task)

# ------------------------------------------------------------ after class and Advanced Study
# Everything after the Core, as on the worksheet, without the feedback QR code. Advanced Study asks
# students to write their own code: no code is given, only an empty cell before each Self-check.
core_cells = len(nb.cells)                                  # the check below runs only the Core
after = qmd.split("# After class: how today's lab helps with Assignment 2\n")[1].split("<!-- COURSE_FEEDBACK_QR:START")[0]
assignment, advanced = after.split("\n# Advanced Study\n")
links, reflection = assignment.split("**Reflection**")
nb.cells.append(md("# After class: how today's lab helps with Assignment 2\n\n" + links.strip()))
nb.cells.append(md("**Reflection**" + reflection.rstrip()))
nb.cells.append(your_answer())
intro, *investigations = re.split(r"^(?=## )", advanced, flags=re.M)
nb.cells.append(md("# Advanced Study (after class, not part of the 50 minutes)\n\n" + intro.strip()))


def add_investigation(body):
    """题目、折叠的 Hint、Ask AI；在 Self-check 前放一个空代码格给学生自己写。"""
    pos = 0
    for box in callout.finditer(body):
        add_text(body[pos:box.start()])
        name, inner = box.groups()
        if name == "Hint":
            nb.cells.append(md("<details><summary><b>Hint</b> (only if you are stuck)</summary>\n\n" + inner.strip() + "\n\n</details>"))
        else:
            nb.cells.append(md(f"**{name}**\n\n" + inner.strip()))
        pos = box.end()
    rest = body[pos:]
    check = rest.find("**Self-check")
    add_text(rest[:check])
    nb.cells.append(code_cell("# Your code here: add more cells with + Code if you need them"))
    add_text(rest[check:])                                   # Lesson 6 的 Self-check 都是文字，没有给定的检查代码


for investigation in investigations:
    title, _, body = investigation.partition("\n")
    if title.startswith("## Advanced Study log"):
        log = body.strip().replace("Copy this into a text cell at the end of your notebook.", "").strip()
        nb.cells.append(md(title + "\n\nDouble-click the cell below and fill it in."))
        nb.cells.append(md(log))
    else:
        nb.cells.append(md(title))
        add_investigation(body)
# The worksheet ends with a feedback QR code, which Colab cannot show: link to the same page instead
nb.cells.append(md("## 30-second feedback\n\nShare anonymous feedback or post a question for this lab on the "
                   "[course's Class page](https://ml.wxing.me/live/) (Give feedback, then choose Lesson 6 lab)."))

# ------------------------------------------------------------ 检查：填好空，从头执行
blank_cells = [cell for cell in nb.cells if cell.cell_type == "code" and "____" in cell.source]
assert len(blank_cells) == len(FILLS), f"{len(blank_cells)} code cells have blanks, FILLS has {len(FILLS)}"
assert not any(cell.source.startswith("**Answer") for cell in nb.cells), "an Answer box slipped into the workbook"
test = nbformat.reads(nbformat.writes(nb), as_version=4)            # a copy to fill and run
test.cells = test.cells[:core_cells]                                 # Advanced Study needs the students' own code
fills = iter(FILLS)
for cell in test.cells:
    if cell.cell_type == "code" and "____" in cell.source:
        for value in next(fills):
            cell.source = cell.source.replace("____", value, 1)
for cell in test.cells:
    cell.source = cell.source.replace(DATA_URL, LOCAL_DATA)
# 用执行本脚本的 Python 环境做内核（同 build_solution.py）
_tmp = tempfile.mkdtemp()
os.environ.setdefault("IPYTHONDIR", _tmp)
os.environ.setdefault("JUPYTER_RUNTIME_DIR", _tmp)
km = KernelManager(kernel_name="python3")
km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
NotebookClient(test, timeout=600, kernel_manager=km, resources={"metadata": {"path": str(ROOT)}}).execute()
printed = "".join(out.get("text", "") for cell in test.cells if cell.cell_type == "code" for out in cell.get("outputs", []))
missing = [value for value in EXPECT if value not in printed]
assert not missing, f"Success checks not reproduced: {missing}\n{printed[:1500]}"

nbformat.write(nb, ROOT / "lab_workbook.ipynb")
answers = sum(cell.source.startswith("**Your answer") for cell in nb.cells)
assert ai_boxes == 3, f"{ai_boxes} Try with AI boxes, expected 3"
assert DATA_URL in nbformat.writes(nb) and LOCAL_DATA not in nbformat.writes(nb), "the workbook must keep the public data URL"
print(f"lab_workbook.ipynb: {len(nb.cells)} cells, {len(blank_cells)} code cells with blanks, {answers} answer cells; all checks reproduced")
