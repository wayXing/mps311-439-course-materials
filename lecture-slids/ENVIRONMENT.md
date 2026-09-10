# Course environment

## Slides and documents

- Node.js 22 (the current Slidev dependency set requires Node 22 or newer)
- npm, using the committed `package-lock.json`
- Quarto 1.8 or a compatible recent 1.x release
- A LaTeX installation only when PDF notes or worksheets need to be rendered

Install JavaScript dependencies with `npm ci`. The dependency versions in `package.json` are pinned so a fresh installation matches the tested course setup.

## Python notebooks

Google Colab is the primary student environment. For local work, Python 3.10 or 3.11 is the most compatible baseline across the existing notebooks. The course uses these main packages:

- NumPy, pandas, SciPy
- Matplotlib and seaborn
- scikit-learn
- Jupyter and ipywidgets
- XGBoost for optional advanced tree material
- TensorFlow/Keras for Weeks 10–11

Notebook metadata comes from several historical environments, so it should not be treated as a complete lock file. Before a new academic year, run at least one representative notebook from Weeks 1, 6, 10, and 11 in the intended Colab/local environment and update deprecated APIs if required.

## Verification

```sh
npm run build:course
npm run check:course
```

The build command regenerates all ten interactive slide decks and lab HTML files. The check fails when an expected source or solution notebook is missing, a notebook is invalid JSON, or a generated slide/lab is older than its source.

