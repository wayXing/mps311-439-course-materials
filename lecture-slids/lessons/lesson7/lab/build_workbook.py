"""从当前 Worksheet 生成学生用的 Colab workbook（lab_workbook.ipynb），并检查它能跑通：
1. 代码全部放进 notebook，学生不用复制；只留 ML 判断的空，纯 API 的空（predict_proba）直接给出；
2. 每个 Question 后面加一个“Your answer”文字格，Question first 的格子在代码之前，先写判断再运行；
3. 不放 Answer（答案只在网页 worksheet 里，折叠）；
4. 检查：把空按 build_solution.py 的 FILLS 填好，用真实 Jupyter 内核从头执行，断言 Core 的核对值。
执行检查时 MNIST 改读本地 ../../data/mnist10k.npz（同 build_solution.py），写出的 workbook 保留公开网址。
写出的 workbook 不含输出。
照 Lesson 4 的 build_workbook.py 改写（2026-10-27）。
运行：uv run --no-project --python 3.12 --with "scikit-learn==1.6.1",matplotlib,pandas,scipy,nbclient,nbformat,ipykernel python build_workbook.py
（检查时固定 scikit-learn 1.6.1，与课程其他构建一致）
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
# 不需要 ML 判断的空直接给出，对应的 Hint 一并去掉：2.1 是方法名 fit，3.1 是题面已经写出的 2 和 10
PREFILL = {
    "2.1": [("pca.____(X_digits_fit)    ", "pca.fit(X_digits_fit)     "),
            ("Fill the blank with the method that **learns** the directions. (In the workbook this is already filled in.)", "Run the cell without changing it.")],
    "3.1": [("n_components=____, random_state=0)   ", "n_components=2, random_state=0)      "),
            ("Fill the blank with the number of components, first `2`, run, then `10`, run again. (In the workbook the `2` is already filled in.)",
             "Run the cell with `2` components, then change the `2` to `10` and run it again.")],
}
# Question first 的任务：代码下面的 Success check 会泄露答案，在 workbook 里折叠起来。
# 值是 Success check 之后、不再折叠的第一段的开头（1.1 的核对值有多行）；None 表示只有一行
FOLD_CHECK = {"1.1": "**Questions:**", "3.1": None, "4.1": "**Questions:**"}
# 其余空按出现顺序的填写内容，只用于检查（与 build_solution.py 的 FILLS 相同，去掉 2.1 和 3.1）
FILLS = [["u"], ["X_digits_val"], ["40"]]
EXPECT = [
    "(5625, 784) (1875, 784) (2500, 784)",
    "angle variance rec_error sum", "0 2.966 1.398 4.364", "30 3.872 0.492 4.364", "120 0.492 3.872 4.364",
    "angle of components_[0]: 31.2", "explained_variance_: [3.874 0.49 ]", "eigvalsh(S): [0.49  3.874]",
    "[0.0955 0.072  0.0619 0.0532 0.0493]", "0.489", "0.827", "0.917", "(1, 2) 57.7",
    "10 0.885", "20 0.942", "30 0.952", "40 0.957", "50 0.956", "60 0.956", "0.962",
]
WORKSHEET_URL = "https://ml.wxing.me/materials/lesson7/lab.html"
# 课程的 MNIST 子集：检查时读本地同一个文件，写出的 workbook 保留公开网址（同 build_solution.py）
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
    md("# Lab 7 Workbook: Fewer Numbers for the Same Data\n## MPS311/439 Machine Learning\n\n"
       "The same tasks as the [Lab 7 worksheet](" + WORKSHEET_URL + "), with the code already in place: **nothing to copy**.\n\n"
       "**What you do today.** The Python is given. What is left to you are the machine-learning decisions: "
       "the blanks `____` and the questions. That is the lab. A cell with a blank stops with an error until you fill it.\n\n"
       "After the five Parts (0 to 4) come the after-class part and the **Advanced Study** (MPS439: investigation A is required). "
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
        if task in FOLD_CHECK and FOLD_CHECK[task] is None:  # "**Success check:** …" on one line
            body = re.sub(r"\*\*Success check:\*\* ([^\n]+)\n",
                          r"<details><summary><b>Success check</b> (open after you run the cell)</summary>\n\n\1\n\n</details>\n", body)
        elif task in FOLD_CHECK:                             # a Success check over several lines, up to the given paragraph
            start = body.index("**Success check:**")
            end = body.index(FOLD_CHECK[task], start)
            check = body[start + len("**Success check:**"):end].strip()
            body = (body[:start] + "<details><summary><b>Success check</b> (open after you run the cell)</summary>\n\n"
                    + check + "\n\n</details>\n\n" + body[end:])
        pos = 0
        for box in callout.finditer(body):
            add_body(body[pos:box.start()], task)
            name, inner = box.groups()
            if name.startswith("Try with AI"):              # folded, so the Core reads without AI
                add_ai_box(name, inner)
            elif name.startswith("What to take away"):      # folded: open only after answering
                nb.cells.append(md(f"<details><summary><b>{name}</b></summary>\n\n" + inner.strip() + "\n\n</details>"))
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
    if "Write the derivation in a text cell" in body:        # A, task 1
        nb.cells.append(md("**Your derivation (task 1):** *(double-click to write)*"))
    nb.cells.append(code_cell("# Your code here: add more cells with + Code if you need them"))
    rest = rest[check:]
    add_text(rest)


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
                   "[course's Class page](https://ml.wxing.me/live/) (Give feedback, then choose Lesson 7 lab)."))

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
printed = "\n".join(out.get("text", "") for cell in test.cells if cell.cell_type == "code" for out in cell.get("outputs", []))
missing = [value for value in EXPECT if value not in printed]
assert not missing, f"Success checks not reproduced: {missing}"
assert DATA_URL in nb.cells[[c.cell_type for c in nb.cells].index("code")].source, "the public data URL is missing"

nbformat.write(nb, ROOT / "lab_workbook.ipynb")
answers = sum(cell.source.startswith("**Your answer") for cell in nb.cells)
assert ai_boxes == 4, f"{ai_boxes} Try with AI boxes, expected 4"
print(f"lab_workbook.ipynb: {len(nb.cells)} cells, {len(blank_cells)} code cells with blanks, {answers} answer cells; all checks reproduced")
