"""Lab 6: build the students' Colab workbook lab_workbook.ipynb from lab_worksheet.qmd.
All code is in place (this lab has no blanks); every Observe / Explain prompt is followed by a "Your answer" text cell;
folded "One possible answer" boxes are left out (they stay on the web page). The notebook is run once in a real
kernel to check it works; the workbook is written without outputs.
Run: export IPYTHONDIR="$TMPDIR/course-ipython"; uv run --no-project --python 3.12 --with scikit-learn,matplotlib,pandas,nbclient,nbformat,ipykernel python build_workbook.py
"""
from pathlib import Path
import copy
import re
import nbformat
from nbclient import NotebookClient
ROOT = Path(__file__).resolve().parent
qmd = (ROOT / "lab_worksheet.qmd").read_text()
body = qmd.split("# Lab 6:", 1)[1].split("# Advanced Study\n")[0]
body = "# Lab 6:" + body
# drop the folded possible answers; keep Challenge and Try with AI boxes as plain text
body = re.sub(r'::: \{\.callout-note collapse="true" title="One possible answer"\}\n.*?\n:::\n', "", body, flags=re.S)
body = re.sub(r'::: \{\.callout-tip collapse="true" title="([^"]+)"\}\n(.*?)\n:::', lambda m: f"> **{m.group(1)}**\n>\n" + "\n".join("> " + l for l in m.group(2).split("\n")), body, flags=re.S)
body = body.replace("1. Open a blank Colab notebook (**File → New notebook in Drive**) and copy each code block into its own cell. Run the cells from top to bottom: later cells use names made by earlier ones.",
                    "1. This notebook already has all the code. Save a copy first (**File → Save a copy in Drive**). Run the cells from top to bottom: later cells use names made by earlier ones.")
cells = [nbformat.v4.new_markdown_cell("The same lab as a web page, with the folded answers: https://ml.wxing.me/materials/lesson6/lab.html")]
parts = re.split(r"```python\n(.*?)```", body, flags=re.S)
for k, part in enumerate(parts):
    if k % 2 == 1:
        cells.append(nbformat.v4.new_code_cell(part.rstrip()))
        continue
    text = part.strip()
    if not text:
        continue
    chunks = re.split(r"\n(?=\*\*(?:Observe|Explain and Limits)\*\*)", text)
    for c in chunks:
        c = c.strip()
        if not c:
            continue
        if c.startswith("**Observe**") or c.startswith("**Explain and Limits**"):
            para, _, rest = c.partition("\n\n")
            cells.append(nbformat.v4.new_markdown_cell(para))
            cells.append(nbformat.v4.new_markdown_cell("*Your answer:* "))
            if rest.strip():
                cells.append(nbformat.v4.new_markdown_cell(rest.strip()))
        else:
            cells.append(nbformat.v4.new_markdown_cell(c))
nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["colab"] = {"provenance": []}
check = copy.deepcopy(nb)
NotebookClient(check, timeout=600, kernel_name="python3").execute()
errors = [o for c in check.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
nbformat.write(nb, ROOT / "lab_workbook.ipynb")
print("cells:", len(nb.cells), "code:", sum(c.cell_type == "code" for c in nb.cells),
      "answer cells:", sum(c.source.startswith("*Your answer") for c in nb.cells if c.cell_type == "markdown"), "errors:", len(errors))
