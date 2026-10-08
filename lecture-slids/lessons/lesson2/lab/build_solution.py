"""Validate lab_worksheet.qmd and build lab_solution.ipynb.
1. Fill the blanks of every Core code block with the answer values below.
2. Check every Answer code line appears in the filled block.
3. Execute the filled sequence plus Advanced Study solutions in a fresh kernel.
Run from this folder: python build_solution.py
"""
import re, sys, nbformat
from nbclient import NotebookClient

FILLS = [
 ["198", "4341"], ["horsepower", "mpg"], ["0.2"], ["LinearRegression", "y_train"],
 ["100"], ["grid"], ["2", "y3"], ["mean"], ["y_pred", "0.5", "baseline_mse"],
 ["features"], ["features"], ["corr"], ["car"], ["all_features"], ["residuals"],
]

qmd = open("lab_worksheet.qmd").read()
core = qmd.split("# Advanced Study: choose one investigation")[0]
fence = re.compile(r"```python\n(.*?)```", re.S)

# walk blocks in order; remember the answer block that follows a blank block
blocks = [(m.start(), m.group(1)) for m in fence.finditer(core)]
def is_ans(pos):
    seg = core[:pos]; k = seg.rfind('title="Answer"')
    return k != -1 and "\n:::" not in seg[k:]
cells, fi, errors = [], 0, 0
i = 0
while i < len(blocks):
    pos, code = blocks[i]
    # is this an Answer block? (inside a callout-note collapse) -> skip, handled with its blank block
    if is_ans(pos):
        i += 1; continue
    if "____" in code:
        filled = code
        for v in FILLS[fi]:
            filled = filled.replace("____", v, 1)
        assert "____" not in filled, f"unfilled blank in block {fi}"
        # answer block: next block that follows an Answer callout
        j = i + 1
        ans = []
        while j < len(blocks) and is_ans(blocks[j][0]):
            ans.append(blocks[j][1]); j += 1
            break
        for a in ans:
            for line in a.strip().splitlines():
                if line.strip() and line.strip() not in filled:
                    print(f"MISMATCH block {fi}: {line!r}"); errors += 1
        fi += 1
        cells.append(filled)
    else:
        cells.append(code)
    i += 1
assert fi == len(FILLS), (fi, len(FILLS))
print("blank blocks:", fi, "mismatches:", errors)

AS = open("as_solutions.py").read().split("#%%")
cells_all = cells + [c.strip("\n") + "\n" for c in AS if c.strip()]

nb = nbformat.v4.new_notebook()
nb.cells.append(nbformat.v4.new_markdown_cell(
 "# Lab 2 Solution: Linear Regression\n\nFull code for every task, executed top to bottom. Open it only after you have tried the worksheet. "
 "In the worksheet, `DATA_URL` points to the course repository; here the same file is read from the local copy."))
for c in cells_all:
    c = c.replace('DATA_URL = "https://raw.githubusercontent.com/wayXing/mps311-439-course-materials/main/lecture-slids/lessons/lesson2/lab/auto_mpg.csv"',
                  'DATA_URL = "auto_mpg.csv"')
    # skip appendix cells
    if "BACKUP_URL" in c or 'read_csv("auto_mpg.csv")' in c: continue
    nb.cells.append(nbformat.v4.new_code_cell(c))
NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": "."}}).execute()
nbformat.write(nb, "lab_solution.ipynb")
print("solution cells:", len(nb.cells))
