# MPS311/439 Machine Learning — Slidev course

This directory is one shared Slidev project. Each teaching lesson has its own
Slidev entry file at `lessons/lessonN/lecture/slide.md`.

## Canonical course structure

```text
lessons/lessonN/
├── lecture/
│   ├── slide.md             # canonical Slidev source
│   ├── note.md              # canonical lecture-note source, when available
│   ├── demo.ipynb           # lecture demonstration, when available
│   ├── figures/             # local slide/note assets
│   ├── figure_gen.py        # figure generator, when available
│   ├── slide-export.pdf     # generated Slidev PDF
│   ├── note.html            # generated lecture note
│   ├── note.pdf             # generated lecture note
│   ├── dist/                # generated standalone Slidev build (double-clickable)
│   └── archive/             # superseded drafts and alternative designs
└── lab/
    ├── lab_worksheet.md     # canonical student worksheet source
    ├── lab_worksheet.html   # generated worksheet
    ├── lab_worksheet.pdf    # generated worksheet
    ├── lab_solution.ipynb   # canonical solution notebook
    └── archive/             # superseded lab material
```

Reading Week is a calendar break and therefore has no lesson directory. The
`foundation` directory contains pre-course Python preparation.

## Canonical Slidev entries

The lecture entries are:

- `lessons/lesson1/lecture/slide.md` through `lessons/lesson10/lecture/slide.md`

## Run, build, and export

Run these commands from this directory. Replace `lesson3` with the required lesson.

```bash
npm ci
npm run dev -- lessons/lesson3/lecture/slide.md
npm run build:standalone -- lessons/lesson3/lecture/slide.md --out dist --base './'
npm run export -- lessons/lesson3/lecture/slide.md --output lessons/lesson3/lecture/slide-export.pdf
```

To rebuild and verify all teaching lessons before publishing:

```bash
npm run build:course
npm run check:course
```

The tested toolchain and local notebook guidance are documented in `ENVIRONMENT.md`.
Use npm and the committed `package-lock.json`; the existing `bun.lock` is a
historical artefact, not the canonical dependency lock.

Lecture notes use the shared Quarto configuration in `lessons/_quarto.yml`.
Render the HTML and PDF from a lecture directory:

```bash
quarto render note.md --to html --output note.html
quarto render note.md --to pdf --output note.pdf
```

The package scripts intentionally do not choose a default lesson. Always pass an
entry file explicitly. Slidev resolves the standalone `--out` directory relative
to the entry file, while it resolves PDF `--output` from this project root; use
the full lesson path shown above for PDF exports. The standalone build embeds the
JavaScript, CSS, and local assets in `dist/index.html`, and the canonical entries
use hash routing, so that file can be opened directly without a web server. A
normal `npm run build` is a multi-file website build and should be served over
HTTP instead of opened via `file://`.

The shared standalone configuration is in `vite.config.ts`. Each lecture has a
small `vite.config.ts` forwarding file because Slidev loads Vite configuration
from the entry directory. The previous multi-file builds are preserved under
each lecture's `archive/dist_multifile/` directory.

The project path contains spaces. Slidev 52.8 encodes those spaces incorrectly
during a Vite build, so `npm install` runs `scripts/patch-slidev-paths.mjs` to
apply a small local compatibility fix to the installed CLI. The patch is
idempotent and does not modify course sources.

## Source and generated files

Edit Markdown, notebooks, figure-generation scripts, and local figure assets.
Do not edit `dist/`, exported PDFs, or generated HTML directly. Historical
versions belong in the nearest `archive/` directory. Optional advanced material
belongs in `supplementary/` rather than in a second `lessonN` directory.

## Publishing notebooks for Google Colab

This is the canonical private source. The course website's **Demo** links use
the generated public mirror at
[`wayXing/mps311-439-course-materials`](https://github.com/wayXing/mps311-439-course-materials),
not this directory directly. After changing any public lecture notebook or
course material, regenerate the mirror from `../course-site` with:

```sh
npm run sync:public
```

Then inspect, commit and push `../mps311-439-course-materials/`. Do not edit the
mirror's copied notebooks by hand. Student data, assessments, credentials,
generated PDFs/HTML, dependencies and `archive/` are private or generated and
must never enter that public repository.
