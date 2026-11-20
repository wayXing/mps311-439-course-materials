"""Lab 7: build lab_solution.ipynb from lab_worksheet.qmd and check it.
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
EXPECT = ["(426, 30) [267 159]", "0.981", "0.452", "(5625, 784) (1875, 784) (2500, 784)", "not full rank",
          "40 0.789 0.957", "0.129", "0.957", "0.90", "0.962"]
FILLS = []   # this lab has no blanks
cells = [nbformat.v4.new_markdown_cell("# Lab 7 solution: Use PCA, and Catch It When It Goes Wrong\nAll code as in the worksheet, run from top to bottom, with one possible answer after each Part.")]
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
        if "qda.fit(X_fit, y_fit)" in code:            # QDA on the pixels: newer scikit-learn stops with an error, as the worksheet says
            cell.metadata["tags"] = ["raises-exception"]
        cells.append(cell)
        if "Z = pca.fit_transform(X_train)" in code:     # the solution also shows the changed value
            cells.append(nbformat.v4.new_markdown_cell("The same cell with `X_train_s`:"))
            cells.append(nbformat.v4.new_code_cell(code.replace("Z = pca.fit_transform(X_train) ", "Z = pca.fit_transform(X_train_s)")))
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
