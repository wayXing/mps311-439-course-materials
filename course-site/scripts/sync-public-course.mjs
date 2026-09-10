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
  '.git', '.astro', '.openai', '.sites-artifacts', '.sites-stage', '.vercel',
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

Licensing is being selected before the first public release. Do not reuse material until a licence is added.
`);

console.log(`Synced public course mirror to ${publicRoot}.`);
