"""Lab 8: test that every way a student loads data works, by running the worksheet's own code as students do.

For each data cell in lab_worksheet.qmd (the wines, the digits, and the two backups in the Appendix):
the import cell and that one cell are run in a fresh Python process, in an empty folder (like a new Colab notebook),
and the arrays they give are compared with the course's reference file lessons/data/mnist10k.npz.
Exit code 0 only if every source works and gives exactly the same images and labels.
Run before every release, and the evening before the lab:
  uv run --no-project --python 3.12 --with scikit-learn,pandas,matplotlib python test_data_loading.py
"""
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT.parents[1] / "data" / "mnist10k.npz"
qmd = (ROOT / "lab_worksheet.qmd").read_text()
blocks = re.findall(r"```python\n(.*?)```", qmd, flags=re.S)
imports = blocks[0]
assert "import urllib.request" in imports


def one(name, marker, check):
    found = [b for b in blocks if marker in b]
    assert len(found) == 1, f"{name}: expected one cell containing {marker!r}, found {len(found)}"
    code = imports + "\n" + found[0] + "\n" + check
    with tempfile.TemporaryDirectory() as folder:     # an empty folder, as in a new notebook
        start = time.time()
        run = subprocess.run([sys.executable, "-c", code], cwd=folder, capture_output=True, text=True, timeout=600)
        seconds = time.time() - start
    ok = run.returncode == 0 and "CHECK OK" in run.stdout
    print(f"{'PASS' if ok else 'FAIL'}  {name}  ({seconds:.1f} s)")
    if not ok:
        print(run.stdout[-800:], run.stderr[-1500:])
    return ok


same_digits = f"""
ref = np.load({str(REFERENCE)!r})
assert data["X"].shape == (10000, 784), data["X"].shape
assert np.array_equal(np.asarray(data["X"]), ref["X"]), "images differ from the course file"
assert np.array_equal(np.asarray(data["y"]), ref["y"]), "labels differ from the course file"
print("CHECK OK")
"""
results = [
    one("wines, Part 1 data block", "load_wine()",
        'assert X_train.shape == (133, 13) and list(np.bincount(y_train)) == [44, 53, 36]\nprint("CHECK OK")'),
    one("digits, Part 2 data block (course file on GitHub)", "DATA_URL =",
        same_digits + 'assert X_train.shape == (7500, 784) and X.max() == 1.0\n'),
    one("digits, Appendix backup 1 (jsDelivr copy)", "BACKUP_URL =", same_digits),
    one("digits, Appendix backup 2 (official MNIST file)", "MNIST_URL =", same_digits),
]
print(f"{sum(results)} of {len(results)} data sources work")
sys.exit(0 if all(results) else 1)
