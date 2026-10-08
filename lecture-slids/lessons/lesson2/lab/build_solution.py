"""从当前 Worksheet 生成带题号、文字答案和执行输出的 Solution。"""
from pathlib import Path
import re
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parent
FILLS = [
    ["198", "4341"], ["horsepower", "mpg"], ["0.2"], ["LinearRegression", "y_train"],
    ["100"], ["grid"], ["2", "y3"], ["mean"], ["y_pred", "0.5", "baseline_mse"],
    ["features"], ["features"], ["corr"], ["car"], ["all_features"], ["residuals"],
]
qmd = (ROOT / "lab_worksheet.qmd").read_text()
core = qmd.split("# Advanced Study: choose one investigation")[0]
fence = re.compile(r"```python\n(.*?)```", re.S)
nb = nbformat.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.cells.append(nbformat.v4.new_markdown_cell(
    "# Lab 2 Solution: Linear Regression\n\n"
    "This solution follows the current worksheet, task by task, including written answers. "
    "Run Core tasks from top to bottom; then choose an Advanced investigation. "
    "The data URL is the same as in the worksheet and works in a blank Colab notebook. "
    "Try the worksheet before opening these answers."
))
fi = 0
for match in re.finditer(r"^### (\d\.\d [^\n]+)\n(.*?)(?=^### |^## |\Z)", core, re.M | re.S):
    title, body = match.groups()
    answer = re.search(r'::: \{[^\n]*title="Answer"[^\n]*\}\n(.*?)\n:::', body, re.S)
    task_body = body[:answer.start()] if answer else body
    nb.cells.append(nbformat.v4.new_markdown_cell("## " + title))
    filled_blocks = []
    for code in fence.findall(task_body):
        if "____" in code:
            assert code.count("____") == len(FILLS[fi]), f"Blank count changed in {title}"
            for value in FILLS[fi]:
                code = code.replace("____", value, 1)
            fi += 1
        filled_blocks.append(code)
        nb.cells.append(nbformat.v4.new_code_cell(code))
    if answer:
        for code in fence.findall(answer.group(1)):
            for line in code.strip().splitlines():
                assert not line.strip() or any(line.strip() in block for block in filled_blocks), (
                    f"Worksheet Answer differs from filled solution in {title}: {line}"
                )
        prose = fence.sub("", answer.group(1)).strip()
        if prose:
            nb.cells.append(nbformat.v4.new_markdown_cell(prose))
assert fi == len(FILLS), (fi, len(FILLS))
nb.cells.append(nbformat.v4.new_markdown_cell(
    "# Advanced Study\n\nRun the Core tasks first. A, B and C below are worked investigations, "
    "including the worksheet self-checks. B can be run without first running A."
))
sections = (ROOT / "as_solutions.py").read_text().split("#%%")
for section in sections:
    section = section.strip()
    if not section:
        continue
    for key, title in [("A", "Solve the same regression in different ways"),
                       ("B", "Gradient descent"), ("C", "How much can a coefficient move?")]:
        if f"# ---- Advanced Study {key} ----" in section:
            nb.cells.append(nbformat.v4.new_markdown_cell(f"## {key}. {title}"))
    nb.cells.append(nbformat.v4.new_code_cell(section + "\n"))
    if "np.linalg.cond(A)" in section:
        nb.cells.append(nbformat.v4.new_markdown_cell(
            "All three solvers reproduce the sklearn fit to numerical precision. Standardising "
            "reduces the six-feature condition numbers from about 8.8e4 and 7.7e9 to 11 and 116. "
            "The nearly singular 2×2 system changes from (2, 0) to (1, 1) after a 0.0001 change "
            "in the second right-hand-side entry. Squaring the condition number in XᵀX makes "
            "numerical errors more serious; an explicit inverse is unnecessary."
        ))
    elif "standardised, eta =" in section:
        nb.cells.append(nbformat.v4.new_markdown_cell(
            "For the scalar loss, eta=0.5 reaches 3 in one step, eta=0.1 converges gradually, "
            "and eta=1.1 oscillates with growing error. On standardised car features, eta=0.1 "
            "and 500 steps agree with least squares (training MSE about 17.98; test RMSE "
            "about 4.22). Raw features overflow at eta=0.1; a much smaller step avoids overflow "
            "but converges slowly. Standardised eta=0.01 is slower and eta=0.6 diverges."
        ))
    elif "b6 = boot_hp" in section:
        nb.cells.append(nbformat.v4.new_markdown_cell(
            "The hp + weight interval remains below zero, whereas the six-feature interval "
            "contains zero. The horsepower coefficient is sensitive to the feature set and "
            "training sample, especially with strongly correlated predictors; it is not a "
            "causal effect. These bootstrap intervals describe resampling variability.\n\n"
            "## Advanced Study log\n\nRecord your chosen investigation, what you found, how AI "
            "helped and what you corrected, and your own self-check result."
        ))
# 强制使用执行本脚本的 Python 环境，避免命中其他环境的内核。
km = KernelManager(kernel_name="python3")
km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
NotebookClient(nb, timeout=300, kernel_manager=km,
               resources={"metadata": {"path": str(ROOT)}}).execute()
nbformat.write(nb, ROOT / "lab_solution.ipynb")
print(f"Validated {fi} blank blocks, all Core task answers and Advanced A/B/C; wrote {len(nb.cells)} cells.")
