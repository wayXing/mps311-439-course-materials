import { access, readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { htmlFiles, pdfLinks } from './public-html.mjs';

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const course = JSON.parse(await readFile(join(projectRoot, 'src/data/course.json'), 'utf8'));
const timetable = JSON.parse(await readFile(join(projectRoot, 'src/data/timetable.json'), 'utf8'));
const publicResources = course.lessons.flatMap((lesson) => (
  lesson.resources.filter((resource) => resource.kind !== 'pdf')
));

if (timetable.basis !== 'lesson') {
  throw new Error('Timetable must use the lesson-based format.');
}

if (timetable.events.some((event) => !('lesson' in event) || !('lectureDate' in event) || !('labDate' in event))) {
  throw new Error('Lesson timetable events must contain lesson and session-date fields.');
}

if (timetable.source !== 'MPS311 _439_TimeTable-byLessons-2026.xlsx') {
  throw new Error(`Unexpected timetable source: ${timetable.source}`);
}

const expected = [
  'dist/index.html',
  'dist/style-samples/index.html',
  'dist/timetable/index.html',
  'dist/data/course-timetable.csv',
  ...course.lessons.map((lesson) => `dist/course/${lesson.slug}/index.html`),
  ...publicResources.map((resource) => `dist/${resource.href}`),
  ...publicResources.flatMap((resource) => (resource.assets ?? []).map((asset) => `dist/${asset.target}`)),
  ...publicResources.flatMap((resource) => (
    resource.downloadHref ? [`dist/${resource.downloadHref}`] : []
  )),
];

const missing = [];
for (const path of expected) {
  await access(join(projectRoot, path)).catch(() => missing.push(path));
}

if (missing.length > 0) {
  throw new Error(`Missing generated paths:\n${missing.map((path) => `- ${path}`).join('\n')}`);
}

const pagesWithPdfLinks = [];
for await (const path of htmlFiles(join(projectRoot, 'dist'))) {
  if (pdfLinks(await readFile(path, 'utf8')).length) pagesWithPdfLinks.push(path);
}
if (pagesWithPdfLinks.length > 0) {
  throw new Error(`Public pages must not offer PDF downloads:\n${pagesWithPdfLinks.join('\n')}`);
}

const slideCount = course.lessons.reduce(
  (count, lesson) => count + lesson.resources.filter((resource) => resource.kind === 'slides').length,
  0,
);

console.log(`Validated ${course.lessons.length + 3} pages and ${publicResources.length} public resources (${slideCount} interactive slide decks).`);
