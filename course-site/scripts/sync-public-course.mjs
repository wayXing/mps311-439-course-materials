import { cp, mkdir, rm, writeFile } from 'node:fs/promises';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const workspaceRoot = resolve(projectRoot, '..');
const publicRoot = resolve(workspaceRoot, 'mps311-439-course-materials');
const publicCourseSite = join(publicRoot, 'course-site');
const publicSlides = join(publicRoot, 'lecture-slids');
const sourceCourseSite = projectRoot;
const sourceSlides = resolve(workspaceRoot, 'lecture-slids');

const courseSiteIgnored = new Set([
  '.git', '.astro', '.openai', '.sites-artifacts', '.sites-stage', '.vercel', 'CLAUDE.md',
  'dist', 'node_modules', 'public/materials', 'supabase/.temp',
]);
const lessonIgnoredNames = new Set([
  '.DS_Store', 'archive', 'dist', 'node_modules', 'note.html', 'note.pdf',
  'note.png', 'note_files', 'slide-export.pdf', 'lab_worksheet.html',
  'lab_worksheet.pdf', 'lab_solution.html',
]);

function relativeParts(root, source) {
  return relative(root, source).split('/').filter(Boolean);
}

function shouldCopyCourseSite(source) {
  const parts = relativeParts(sourceCourseSite, source);
  if (parts.some((_, index) => courseSiteIgnored.has(parts.slice(0, index + 1).join('/')))) return false;
  return !parts.some((part) => part === '.env' || part.startsWith('.env.'));
}

function shouldCopySlides(source) {
  const parts = relativeParts(sourceSlides, source);
  if (parts.includes('admin') || parts.some((part) => lessonIgnoredNames.has(part))) return false;
  return !parts.some((part) => part.endsWith('.pdf') || part.endsWith('.webloc'));
}

await mkdir(publicRoot, { recursive: true });
await rm(publicCourseSite, { recursive: true, force: true });
await rm(publicSlides, { recursive: true, force: true });

await cp(sourceCourseSite, publicCourseSite, { recursive: true, filter: shouldCopyCourseSite });
await cp(sourceSlides, publicSlides, { recursive: true, filter: shouldCopySlides });

await writeFile(join(publicRoot, 'README.md'), `# MPS311 / MPS439 Machine Learning course materials

This repository contains the public source for the MPS311 / MPS439 Machine Learning course: teaching notes, interactive slide sources, laboratories, demonstration notebooks, and the course website source.

## Run a demonstration in Colab

Open a notebook under \`lecture-slids/lessons/*/lecture/demo.ipynb\` with Google Colab. The course website links directly to these files.

## Maintainers

This is a generated public mirror. Edit the canonical private course workspace, then run \`npm run sync:public\` from \`course-site\` before committing and pushing this repository. Do not put student, assessment, feedback, or administrative data in this repository.

## Licence

- Software in this repository is available under the [MIT License](LICENSE-CODE).
- Teaching material, including slides, notes, worksheets, notebooks, figures, and course data, is available under [CC BY-NC-SA 4.0](LICENSE-MATERIALS.md).

Files credited to third parties retain their own terms. Student, assessment, feedback, and administrative information is intentionally excluded.
`);

await writeFile(join(publicRoot, 'LICENSE-CODE'), `MIT License

Copyright (c) 2026 Wei Xing

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
`);

await writeFile(join(publicRoot, 'LICENSE-MATERIALS.md'), `# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International

Unless a file states otherwise, the teaching material in this repository — including slides, notes, worksheets, notebooks, figures, and course data — is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

You may share and adapt this material for non-commercial purposes, provided that you give appropriate credit to Dr Wei Xing and the MPS311 / MPS439 Machine Learning course, link to this licence, indicate any changes, and distribute adaptations under the same licence.

Third-party material and dependencies retain their own licence terms. This licence does not cover student work, assessment material, feedback, or administrative information.
`);

console.log(`Synced public course mirror to ${publicRoot}.`);
