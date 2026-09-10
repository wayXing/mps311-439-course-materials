## Development

When starting the dev server, use background mode:

```
astro dev --background
```

Manage the background server with `astro dev stop`, `astro dev status`, and `astro dev logs`.

## Course materials and website scope

- This repository's complete course-material set is the source for Blackboard
  uploads. Keep those source files, including PDFs, intact.
- The public course website is only a lightweight, mobile-friendly entry point
  for use during class. It is not the authoritative full-material archive.
- For Vercel deployments, it is intentional to exclude PDF handouts and
  regenerated deployment artifacts. The site must retain its slides, labs,
  interactive material, and feedback page; the local full materials remain
  available for Blackboard.

## Updating course materials and deploying

When asked to update course content, locate the canonical source first in the
sibling directory `../lecture-slids/lessons/`; never edit
`course-site/public/materials/` or `course-site/dist/` directly.

- For a lecture, edit `lessons/lessonN/lecture/slide.md`, then run from
  `../lecture-slids`:
  `npm run build:standalone -- lessons/lessonN/lecture/slide.md --out dist --base './'`.
  Export `slide-export.pdf` only when the Blackboard copy also needs updating.
- For a lab worksheet, edit `lessons/lessonN/lab/lab_worksheet.md`, then render
  its HTML with Quarto. Website deployments intentionally omit PDF handouts.
- After any lecture/lab source rebuild, run from this repository:
  `npm run sync:course`, then `npm run check`.
- The public GitHub mirror used by one-click Colab is the generated sibling
  `../mps311-439-course-materials/`. When public course source, website course
  configuration, or demo notebooks change, run `npm run sync:public` here,
  inspect the mirror, then commit and push it from that directory. Never edit
  copied mirror files directly; it preserves only `.git` and regenerates the
  copied `course-site/` and `lecture-slids/` trees.
- Google Colab paths are assembled from `src/data/course.json` (`colab`) and a
  resource's notebook path. Keep those values aligned with the public mirror's
  actual layout before deploying the website.
- The mirror is strictly public: it must not contain `brief/Admin/`, student or
  assessment data, feedback administration, `.env` files, credentials,
  deployment metadata, local agent files, generated PDFs/HTML, dependencies, or
  archives. Review its staged file list before every push.
- After website source changes, or after the material sync/check above, deploy
  the production site with `vercel --prod --yes` from this repository. Confirm
  deployment is Ready and that `https://ml.wxing.me` remains its production alias.
- Changes made through `/admin/` (academic-year visibility and feedback data)
  are Supabase changes and do not require a Vercel deployment.
- Feedback QR codes use `https://ml.wxing.me`; regenerate/reinstall/recompile
  them only for a new academic year, changed session structure, or changed
  production domain—not for ordinary content edits.

## Documentation

Full documentation: https://docs.astro.build

Consult these guides before working on related tasks:

- [Adding pages, dynamic routes, or middleware](https://docs.astro.build/en/guides/routing/)
- [Working with Astro components](https://docs.astro.build/en/basics/astro-components/)
- [Using React, Vue, Svelte, or other framework components](https://docs.astro.build/en/guides/framework-components/)
- [Adding or managing content](https://docs.astro.build/en/guides/content-collections/)
- [Adding styles or using Tailwind](https://docs.astro.build/en/guides/styling/)
- [Supporting multiple languages](https://docs.astro.build/en/guides/internationalization/)
