"""Lab 8: build lab_solution.ipynb from lab_worksheet.qmd and check it.
Every code block before "# Advanced Study" becomes a code cell, every "One possible answer" a markdown cell after its Part.
The notebook is run top to bottom in a real Jupyter kernel; outputs are saved; the numbers the worksheet
promises ("Run, and you should see") must appear in the outputs.
Run: export IPYTHONDIR="$TMPDIR/course-ipython"; uv run --no-project --python 3.12 --with scikit-learn,matplotlib,pandas,nbclient,nbformat,ipykernel python build_solution.py
"""
from pathlib import Path
import re
import textwrap
import nbformat
from nbclient import NotebookClient
ROOT = Path(__file__).resolve().parent
qmd = (ROOT / "lab_worksheet.qmd").read_text()
body = qmd.split("# Appendix: only if the digits")[0]   # the backups are tested by test_data_loading.py, not run here
EXPECT = ["(133, 13) [44 53 36]", "0.729", "0.955", "323.6", "(5625, 784) (1875, 784) (2500, 784)", "169025.6",
          "fitting: 0.576", "validation: 0.577", "0.016 GB", "majority vote: 0.62", "majority vote: 0.592", "majority vote: 0.664",
          "validation: 0.117", "validation: 0.192", "fitting: 0.561", "validation: 0.106"]
FILLS = []   # this lab has no blanks
cells = [nbformat.v4.new_markdown_cell("# Lab 8 solution: Find Groups, and Catch It When It Goes Wrong\nAll code as in the worksheet, run from top to bottom, with one possible answer after each Part.")]
pos = 0
pat = re.compile(r"(^#{1,3} [^\n]+$)|```python\n(.*?)```|::: \{\.callout-note collapse=\"true\" title=\"One possible answer\"\}\n(.*?)\n:::|::: \{\.content-hidden\}\n(.*?)\n:::", re.S | re.M)
for m in pat.finditer(body):
    if m.group(1):
        cells.append(nbformat.v4.new_markdown_cell(m.group(1)))
    elif m.group(2) is not None:
        code = textwrap.dedent(m.group(2)).rstrip()
        if "____" in code:                                # fill the blanks with the answers, in order
            for value in FILLS.pop(0):
                code = code.replace("____", value, 1)
            assert "____" not in code
        cell = nbformat.v4.new_code_cell(code)
        cells.append(cell)
        if "groups = km.fit_predict(X_train) " in code:  # the solution also shows the changed value
            cells.append(nbformat.v4.new_markdown_cell("The same cell with `X_train_s`:"))
            cells.append(nbformat.v4.new_code_cell(code.replace("groups = km.fit_predict(X_train) ", "groups = km.fit_predict(X_train_s)")))
        if code.startswith("n = 2000"):                 # and the other two sizes of Part 3
            for n in ["500", "4000"]:
                cells.append(nbformat.v4.new_markdown_cell(f"The same cell with `n = {n}`:"))
                cells.append(nbformat.v4.new_code_cell(code.replace("n = 2000 ", f"n = {n} ".ljust(9))))
    elif m.group(3) is not None:
        cells.append(nbformat.v4.new_markdown_cell("**One possible answer.** " + m.group(3)))
    else:
        cells.append(nbformat.v4.new_markdown_cell(m.group(4)))   # Going further: one possible answer
nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
NotebookClient(nb, timeout=600, kernel_name="python3").execute()
out = ""
for c in nb.cells:
    if c.cell_type == "code":
        for o in c.get("outputs", []):
            out += o.get("evalue", "") + o.get("text", "") + str(o.get("data", {}).get("text/plain", "")) + "\n"
missing = [e for e in EXPECT if e not in out]
nbformat.write(nb, ROOT / "lab_solution.ipynb")
print("code cells:", sum(c.cell_type == "code" for c in nb.cells))
print("missing:", missing, "unused fills:", FILLS)
print(out[-1500:])
