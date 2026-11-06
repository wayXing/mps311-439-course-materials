"""Lab 6: build lab_solution.ipynb from lab_worksheet.qmd and check it.
Every code block before "# Advanced Study" becomes a code cell, every "One possible answer" a markdown cell after its Part.
The notebook is run top to bottom in a real Jupyter kernel; outputs are saved; the numbers the worksheet
promises ("Run, and you should see") must appear in the outputs.
Run: export IPYTHONDIR="$TMPDIR/course-ipython"; uv run --no-project --python 3.12 --with scikit-learn,matplotlib,pandas,nbclient,nbformat,ipykernel python build_solution.py
"""
from pathlib import Path
import re
import nbformat
from nbclient import NotebookClient
ROOT = Path(__file__).resolve().parent
qmd = (ROOT / "lab_worksheet.qmd").read_text()
body = qmd.split("# Advanced Study\n")[0]
EXPECT = ["(112, 4) [37 37 38]", "0.955", "37 0.946", "0.696", "1 0.625", "3 0.839 0.795", "4 0.848 0.795",
          "4 0.714", "petal width     0.968", "petal length    0.032", "0.964",
          "0.892", "petal length    0.986", "not sure (top probability below 0.95): 32", "wrong: 5", "0.947", "(535, 6) [318 217]", "0.822"]
FILLS = []   # this lab has no blanks
cells = [nbformat.v4.new_markdown_cell("# Lab 6 solution: Grow a Tree, Find What Goes Wrong, Fix It\nAll code as in the worksheet, run from top to bottom, with one possible answer after each Part.")]
pos = 0
pat = re.compile(r"(^#{1,3} [^\n]+$)|```python\n(.*?)```|::: \{\.callout-note collapse=\"true\" title=\"One possible answer\"\}\n(.*?)\n:::", re.S | re.M)
for m in pat.finditer(body):
    if m.group(1):
        cells.append(nbformat.v4.new_markdown_cell(m.group(1)))
    elif m.group(2) is not None:
        code = m.group(2).rstrip()
        if "____" in code:                                # fill the blanks with the answers, in order
            for value in FILLS.pop(0):
                code = code.replace("____", value, 1)
            assert "____" not in code
        cells.append(nbformat.v4.new_code_cell(code))
    else:
        cells.append(nbformat.v4.new_markdown_cell("**One possible answer.** " + m.group(3)))
nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
NotebookClient(nb, timeout=600, kernel_name="python3").execute()
out = ""
for c in nb.cells:
    if c.cell_type == "code":
        for o in c.get("outputs", []):
            out += o.get("text", "") + str(o.get("data", {}).get("text/plain", "")) + "\n"
missing = [e for e in EXPECT if e not in out]
nbformat.write(nb, ROOT / "lab_solution.ipynb")
print("code cells:", sum(c.cell_type == "code" for c in nb.cells))
print("missing:", missing, "unused fills:", FILLS)
print(out[-1500:])
