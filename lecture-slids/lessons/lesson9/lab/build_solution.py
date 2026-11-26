"""Lab 9: build lab_solution.ipynb from lab_worksheet.qmd and check it.
Every code block before "# Appendix" becomes a code cell, every "One possible answer" a markdown cell after its Part.
The notebook is run top to bottom in a real Jupyter kernel; outputs are saved; the numbers the worksheet
promises ("Run, and you should see") must appear in the outputs.
Run: export IPYTHONDIR="$TMPDIR/course-ipython"; uv run --no-project --python 3.12 --with tensorflow,scikit-learn,matplotlib,requests,nbclient,nbformat,ipykernel python build_solution.py
"""
from pathlib import Path
import re
import textwrap
import nbformat
from nbclient import NotebookClient
ROOT = Path(__file__).resolve().parent
qmd = (ROOT / "lab_worksheet.qmd").read_text()
body = qmd.split("# Appendix: only if MNIST")[0]   # the backups are tested by test_data_loading.py, not run here
EXPECT = ["(50000, 784) (10000, 784) (10000, 784)", "[5 0 4 1 9 2 1 3 1 4]", "118282",
          "0.9645 0.9716 0.9741 0.9743 0.97", "4 0.093",
          "1.95 0.9571 0.9568", "0.39 0.9163 0.9203", "0.25 0.9871 0.9834",
          "1 0.976", "2 0.9724", "0.976"]
cells = [nbformat.v4.new_markdown_cell("# Lab 9 solution: Train a Neural Network, and Catch It When It Goes Wrong\nAll code as in the worksheet, run from top to bottom, with one possible answer after each Part.")]
pat = re.compile(r"(^#{1,3} [^\n]+$)|```python\n(.*?)```|::: \{\.callout-note collapse=\"true\" title=\"One possible answer\"\}\n(.*?)\n:::|::: \{\.content-hidden\}\n(.*?)\n:::", re.S | re.M)
for m in pat.finditer(body):
    if m.group(1):
        cells.append(nbformat.v4.new_markdown_cell(m.group(1)))
    elif m.group(2) is not None:
        code = textwrap.dedent(m.group(2)).rstrip()
        cells.append(nbformat.v4.new_code_cell(code))
    elif m.group(3) is not None:
        cells.append(nbformat.v4.new_markdown_cell("**One possible answer.** " + m.group(3)))
    else:
        cells.append(nbformat.v4.new_markdown_cell(m.group(4)))   # Going further: one possible answer
nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
NotebookClient(nb, timeout=1200, kernel_name="python3").execute()
out = ""
for c in nb.cells:
    if c.cell_type == "code":
        for o in c.get("outputs", []):
            out += o.get("evalue", "") + o.get("text", "") + str(o.get("data", {}).get("text/plain", "")) + "\n"
missing = [e for e in EXPECT if e not in out]
nbformat.write(nb, ROOT / "lab_solution.ipynb")
print("code cells:", sum(c.cell_type == "code" for c in nb.cells))
print("missing:", missing)
print(out[-2500:])
