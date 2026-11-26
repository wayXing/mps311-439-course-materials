"""Lab 10: build lab_solution.ipynb from lab_worksheet.qmd and check it.
Every code block before the Appendix becomes a code cell, every "One possible answer" a markdown cell after its Part.
The solution also runs the values the worksheet asks students to change (Part 1: X_val and 8 filters; Part 2: other moves).
The notebook is run top to bottom in a real Jupyter kernel; outputs are saved; the numbers the worksheet
promises ("Run, and you should see") must appear in the outputs.
Run: export IPYTHONDIR="$TMPDIR/course-ipython"; uv run --no-project --python 3.12 --with scikit-learn,matplotlib,nbclient,nbformat,ipykernel,tensorflow python build_solution.py
"""
from pathlib import Path
import re
import sys
import textwrap
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
qmd = (ROOT / "lab_worksheet.qmd").read_text()
body = qmd.split("# Appendix: only if MNIST")[0]   # the backups are tested by test_data_loading.py, not run here
EXPECT = ["(50000, 28, 28, 1) (50000, 784) (10000, 28, 28, 1)",
          "Expected shape (None, 28, 28, 1), but input has incompatible shape (5, 784)",
          "12,810", "(5, 10)", "160 4640 8010 12810", "10,426",
          "output.shape=(None, 5, 5, 10)", "true 5 predicted 8 images 12", "176", "0.9866", "fully connected 118282 0.9716 0.8263", "CNN 12810 0.9824 0.9517"] + sys.argv[1:]
cells = [nbformat.v4.new_markdown_cell("# Lab 10 solution: Use a CNN, and Catch It When It Goes Wrong\n"
                                       "All code as in the worksheet, run from top to bottom, with one possible answer after each Part.")]
pat = re.compile(r"(^#{1,3} [^\n]+$)|```python\n(.*?)```|::: \{\.callout-note collapse=\"true\" title=\"One possible answer\"\}\n(.*?)\n:::|::: \{\.content-hidden\}\n(.*?)\n:::", re.S | re.M)
for m in pat.finditer(body):
    if m.group(1):
        cells.append(nbformat.v4.new_markdown_cell(m.group(1)))
    elif m.group(2) is not None:
        code = textwrap.dedent(m.group(2)).rstrip()
        cell = nbformat.v4.new_code_cell(code)
        if "X_val_flat[:5]" in code or "cnn_b.fit" in code:     # the expected errors of Part 1 and Version B
            cell.metadata["tags"] = ["raises-exception"]
        cells.append(cell)
        if "X_val_flat[:5]" in code:                            # the changed value, then 8 filters, then back to 16
            cells.append(nbformat.v4.new_markdown_cell("The same cell with `X_val`:"))
            fixed = code.replace("X_val_flat[:5]", "X_val[:5]")
            cells.append(nbformat.v4.new_code_cell(fixed))
            cells.append(nbformat.v4.new_markdown_cell("With 8 filters in the first layer (a separate name, so `cnn` keeps 16):"))
            eight = fixed.replace("Conv2D(16, 3", "Conv2D(8, 3").replace("cnn = ", "cnn8 = ").replace("cnn.summary", "cnn8.summary")
            eight = eight.split("probs = ")[0].rstrip()
            cells.append(nbformat.v4.new_code_cell(eight))
        if code.startswith("move = 2"):                         # the other moves
            for move in [0, 1, 3, 4]:
                cells.append(nbformat.v4.new_code_cell(code.replace("move = 2 ", f"move = {move} ", 1)))
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
for c in nb.cells:                                              # short print of every output, for the worksheet's numbers
    if c.cell_type == "code":
        text = " | ".join((o.get("evalue", "") or o.get("text", "")).strip().replace("\n", " / ")[-400:]
                          for o in c.get("outputs", []) if o.get("output_type") in ("stream", "error"))
        print(">>", c.source.split("\n")[0][:60], "::", text[-500:])
