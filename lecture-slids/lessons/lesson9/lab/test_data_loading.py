"""Lab 9: test that every way a student loads MNIST works, by running the worksheet's own code as students do.

For each source in lab_worksheet.qmd (the Keras data block of Part 0, and the two backups in the Appendix),
the import cell and that source are run in a fresh Python process, in an empty folder (like a new Colab notebook).
A backup is followed by the rest of the data block, as the worksheet says. The arrays are compared with the
course's copy of the Keras file, course-site/public/assignments/assignment-two/mnist.npz.
Exit code 0 only if every source works and gives exactly the same images, labels and split.
Run before the lab to check the data downloads:
  uv run --no-project --python 3.12 --with tensorflow,scikit-learn,pandas,matplotlib,requests python test_data_loading.py
"""
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT.parents[3] / "course-site" / "public" / "assignments" / "assignment-two" / "mnist.npz"
qmd = (ROOT / "lab_worksheet.qmd").read_text()
blocks = re.findall(r"```python\n(.*?)```", qmd, flags=re.S)
imports = blocks[0]
assert "import requests" in imports
data_block = [b for b in blocks if "keras.datasets.mnist.load_data()" in b]
assert len(data_block) == 1, "expected one data block"
data_block = data_block[0]
rest_of_block = data_block.split("\n", 1)[1]              # the data block from its second line, as the Appendix says

check = f"""
ref = np.load({str(REFERENCE)!r})
assert np.array_equal(np.asarray(x_train), ref["x_train"]), "training images differ from the course file"
assert np.array_equal(np.asarray(y_train), ref["y_train"]), "training labels differ from the course file"
assert np.array_equal(np.asarray(x_test), ref["x_test"]), "test images differ from the course file"
assert np.array_equal(np.asarray(y_test), ref["y_test"]), "test labels differ from the course file"
assert X_fit.shape == (50000, 784) and X_val.shape == (10000, 784) and X_test.shape == (10000, 784)
assert X_fit.max() == 1.0 and list(y_fit[:10]) == [5, 0, 4, 1, 9, 2, 1, 3, 1, 4]
print("CHECK OK")
"""


def one(name, code):
    with tempfile.TemporaryDirectory() as folder:     # an empty folder, as in a new notebook
        start = time.time()
        run = subprocess.run([sys.executable, "-c", imports + "\n" + code + "\n" + check],
                             cwd=folder, capture_output=True, text=True, timeout=900)
        seconds = time.time() - start
    ok = run.returncode == 0 and "CHECK OK" in run.stdout
    print(f"{'PASS' if ok else 'FAIL'}  {name}  ({seconds:.1f} s)")
    if not ok:
        print(run.stdout[-800:], run.stderr[-1500:])
    return ok


backup1 = [b for b in blocks if "BACKUP_URL =" in b]
backup2 = [b for b in blocks if "fetch_openml(" in b]
assert len(backup1) == 1 and len(backup2) == 1
results = [
    one("Part 0 data block (Keras)", data_block),
    one("Appendix backup 1 (course website)", backup1[0] + "\n" + rest_of_block),
    one("Appendix backup 2 (OpenML)", backup2[0] + "\n" + rest_of_block),
]
print(f"{sum(results)} of {len(results)} data sources work")
sys.exit(0 if all(results) else 1)
