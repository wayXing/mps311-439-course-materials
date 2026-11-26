"""Lab 10: test that every way a student loads MNIST works, by running the worksheet's own code as students do.

For each data source in WORKSHEET (Keras in the Part 0 set-up cell, and the two backups in the Appendix):
the set-up cell (or the import lines and the backup cell) is run in a fresh Python process, in an empty folder
and with an empty Keras cache (like a new Colab notebook), then the Part 0 data block is run, and the arrays are
compared with the course's reference file course-site/public/assignments/assignment-two/mnist.npz.
Exit code 0 only if every source works and gives exactly the same images, digits and split.
Run before every release, and the evening before the lab:
  uv run --no-project --python 3.12 --with scikit-learn,pandas,matplotlib,tensorflow python test_data_loading.py
"""
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

WORKSHEET = "lab_worksheet.qmd"
ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT.parents[3] / "course-site" / "public" / "assignments" / "assignment-two" / "mnist.npz"
qmd = (ROOT / WORKSHEET).read_text()
blocks = re.findall(r"```python\n(.*?)```", qmd, flags=re.S)
setup = blocks[0]
assert "keras.datasets.mnist.load_data()" in setup and "import urllib.request" in setup
imports = "\n".join(l for l in setup.split("\n") if l.startswith(("import ", "from ")))
data_block = [b for b in blocks if b.startswith("X = x_train.reshape")]
assert len(data_block) == 1, "expected one data block starting with X = x_train.reshape"
data_block = data_block[0]

check = f"""
ref = np.load({str(REFERENCE)!r})
assert np.array_equal(np.asarray(x_train), ref["x_train"]), "images differ from the course file"
assert np.array_equal(np.asarray(y_train), ref["y_train"]), "digits differ from the course file"
assert X_fit.shape == (50000, 28, 28, 1) and X_val.shape == (10000, 28, 28, 1) and X_fit_flat.shape == (50000, 784)
assert X.max() == 1.0 and np.array_equal(y_val, ref["y_train"][50000:])
print("CHECK OK")
"""


def one(name, code):
    with tempfile.TemporaryDirectory() as folder:      # an empty folder and an empty Keras cache, as in a new notebook
        env = dict(os.environ, KERAS_HOME=folder, TF_CPP_MIN_LOG_LEVEL="3")
        start = time.time()
        run = subprocess.run([sys.executable, "-c", code + "\n" + data_block + "\n" + check],
                             cwd=folder, env=env, capture_output=True, text=True, timeout=900)
        seconds = time.time() - start
    ok = run.returncode == 0 and "CHECK OK" in run.stdout
    print(f"{'PASS' if ok else 'FAIL'}  {name}  ({seconds:.1f} s)")
    if not ok:
        print(run.stdout[-800:], run.stderr[-1500:])
    return ok


def backup(marker):
    found = [b for b in blocks if marker in b]
    assert len(found) == 1, f"expected one cell containing {marker!r}, found {len(found)}"
    return imports + "\n" + found[0]


results = [
    one("Keras (Part 0 set-up cell)", setup),
    one("Appendix backup 1 (course website)", backup("BACKUP_URL =")),
    one("Appendix backup 2 (official file, direct)", backup("MNIST_URL =")),
]
print(f"{sum(results)} of {len(results)} data sources work")
sys.exit(0 if all(results) else 1)
