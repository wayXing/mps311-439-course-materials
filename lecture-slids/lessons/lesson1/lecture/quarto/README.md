# Lesson 1 Quarto lecture

This is the current Lesson 1 lecture used by the 2026/27 course website. The
earlier Slidev source at `../slide.md` and its rendered presentation remain
available as reference material; they are not overwritten by this version.

## Three-draft workflow

This version keeps three editable drafts before rendering:

1. [`01_narrative.md`](01_narrative.md) defines the lesson's central question,
   cognitive change and causal arc without deciding slide count.
2. [`02_teaching_script.md`](02_teaching_script.md) turns that arc into timed
   teaching beats, evidence, questions and transitions.
3. [`03_slide_spec.md`](03_slide_spec.md) states what students should see on
   each slide and maps those pages to the teaching script.

[`slide.qmd`](slide.qmd) is the current executable implementation of the third
draft. The three planning files are discussion drafts and can be revised. A
change to the argument or teaching sequence should start upstream; a purely
visual or technical correction can be made directly in `slide.qmd`.

It follows the course-wide visual rules in
[`../../../../SLIDE_DESIGN_SYSTEM.md`](../../../../SLIDE_DESIGN_SYSTEM.md).

The visual language follows Lessons 5–10: a photographic cover, a light canvas,
blue as the primary course colour, green for successful outcomes, red or orange
for failure and risk, and purple for advanced material. The layout is denser
than the first Quarto draft and uses executable Python cells for the evidence.

The deck convention is fixed at both ends: the opening slide uses a random
photographic cover, and the final slide is a dedicated feedback QR page. The
closing content slide and the QR slide remain separate.

`course-theme.scss` contains the reusable Quarto/RevealJS course theme.
`styles.css` contains layouts that are specific to this lesson.

## Render

From this directory:

```sh
uv sync
uv run quarto render slide.qmd
```

The rendered output is `slide.html`. Quarto executes the Python cells, embeds
the charts and outputs, and produces a self-contained RevealJS presentation.

## Data

`data/sp500_fred.csv` contains the FRED `SP500` series downloaded on
24 September 2026. The model uses observations before 2024 for fitting and the
2024 calendar year for evaluation.

Source: S&P Dow Jones Indices LLC, S&P 500 [SP500], retrieved from FRED,
Federal Reserve Bank of St. Louis.

The example is for teaching and is not investment advice.

## Cover image

`random-cover.html` requests a new image from the official Slidev curated cover
collection whenever the presentation opens. If the request fails, the title
slide keeps the local fallback at `assets/course-cover-books.webp`. The fallback
photo is by
[Roman Kraft](https://unsplash.com/@iamromankraft),
[Vintage books collection](https://unsplash.com/photos/vintage-books-collection-X1exjxxBho4).
