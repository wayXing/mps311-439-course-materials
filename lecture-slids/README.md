# MPS311/439 Machine Learning — course material sources

Lessons 1 and 2 use Quarto/RevealJS QMD slides. Lessons 3–10 currently retain their existing Slidev sources. New slide work should follow the approved lesson workflow and the canonical source for that lesson.

The shared visual language and classroom readability standards are defined in
[`SLIDE_DESIGN_SYSTEM.md`](./lessons/shared/design/SLIDE_DESIGN_SYSTEM.md). Agent-facing production
rules for every lesson live in [`lessons/AGENTS.md`](./lessons/AGENTS.md).

## Canonical slide entries

- Lesson 1: `lessons/lesson1/lecture/slides/slide.qmd`
- Lesson 2: `lessons/lesson2/lecture/slides/slide.qmd`
- Lessons 3–10: `lessons/lessonN/lecture/slide.md` (existing Slidev sources)

Notes, figures, labs, and solutions remain under each lesson's `lecture/` and `lab/` directories. Superseded usable versions belong in that lesson's `archive/` directory. The `foundation` directory contains pre-course Python preparation.

## Run, build, and export

For Lesson 2, render the canonical QMD and export its PDF from this directory:

```bash
python3 lessons/lesson2/lecture/slides/build_demo_embed.py
quarto render lessons/lesson2/lecture/slides/slide.qmd --to revealjs
node lessons/lesson2/lecture/slides/export_pdf.mjs
```

For an existing Slidev lesson such as Lesson 3:

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

The package scripts intentionally do not choose a default Slidev lesson. Pass an
entry file explicitly. For the legacy Slidev lessons, `--out` resolves relative
to the entry file while PDF `--output` resolves from this project root. The standalone build embeds the
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
