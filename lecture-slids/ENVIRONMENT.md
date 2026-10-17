# Course environment

- **Quarto** 1.8 (`/usr/local/bin/quarto` on the teacher's Mac) for Notes, slides and worksheets.
- **Python** for figures, numbers and solutions: `~/opt/anaconda3/bin/python` (NumPy, pandas, scikit-learn, Matplotlib, Playwright).
- **TensorFlow / Keras** (Lessons 9–10): anaconda's Python runs under Rosetta and cannot import TensorFlow. Use an arm64 virtual environment, `/opt/homebrew/bin/python3.12 -m venv <dir>`, then `pip install tensorflow`.
- **Notebooks**: when executing solutions with nbclient, set `IPYTHONDIR` and `JUPYTER_RUNTIME_DIR` to a temporary folder (the home `~/.ipython` link is broken).
- **Screenshots and PDF export** use the installed Google Chrome (Playwright has no bundled browser here).
- **Node.js 22** only for the website (`../course-site`) and, if ever needed, rebuilding the 2025 Slidev decks (`npm ci` here first).
- **Students** use Google Colab; data are loaded from the public mirror or scikit-learn / Keras.
