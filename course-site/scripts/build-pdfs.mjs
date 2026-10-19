// Print every current student page (slides, Notes, labs, assignment briefs) to PDF for
// students who print and annotate by hand. It only controls printing: it reads the HTML
// that `npm run sync:course` placed in public/, and never edits any course source.
//
//   npm run pdf                  all lessons, foundation and assignments
//   npm run pdf -- lesson-04     one or more slugs (also: foundation, assignments)
//
// Output: ../lecture-slids/lessons/PDF/<Folder>/*.pdf, then mirrored to Google Drive
// (set COURSE_PDF_DRIVE to override the folder, or COURSE_PDF_DRIVE=off to skip).

import { execFile, execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { homedir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { promisify } from 'node:util';

const execFileAsync = promisify(execFile);
const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const publicRoot = join(projectRoot, 'public');
const pdfRoot = resolve(projectRoot, '../lecture-slids/lessons/PDF');
const python = join(homedir(), 'opt/anaconda3/bin/python');
const chrome = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const driveDefault = join(homedir(), 'Library/CloudStorage/GoogleDrive-w.xing@sheffield.ac.uk/My Drive/MPS311_439 Machine Learning PDFs');
const drive = process.env.COURSE_PDF_DRIVE ?? driveDefault;

// puppeteer-core ships with the globally installed decktape; no extra install needed.
const globalRoot = execFileSync('npm', ['root', '-g'], { encoding: 'utf8' }).trim();
const require = createRequire(join(globalRoot, 'decktape/package.json'));
const puppeteer = require('puppeteer-core');

const course = JSON.parse(await readFile(join(projectRoot, 'src/data/course.json'), 'utf8'));
const wanted = process.argv.slice(2);

// ---------- what to print ----------
const jobs = [];
for (const lesson of course.lessons) {
  if (wanted.length && !wanted.includes(lesson.slug)) continue;
  const isFoundation = lesson.slug === 'foundation';
  const folder = isFoundation ? '00_Foundation' : `Lesson${lesson.number}`;
  const prefix = isFoundation ? 'Foundation' : `L${lesson.number}`;
  const label = isFoundation ? 'Foundation' : `Lesson ${Number(lesson.number)}`;
  const current = (kind) => lesson.resources.find((r) => r.kind === kind && r.group !== 'Archive');
  const add = (kind, type, name) => {
    const r = current(kind);
    if (r) jobs.push({ type, folder, file: `${prefix}_${name}.pdf`, src: join(publicRoot, r.href), label: `${label} · ${lesson.title}` });
  };
  add('slides', 'slides', 'slides');
  add('notes', 'page', 'notes');
  add('guide', 'page', 'python_guide');
  add('workbook', 'page', 'lab_workbook');
  add('worksheet', 'page', 'lab_worksheet');
  add('solution', 'page', 'lab_solution');
}
if (!wanted.length || wanted.includes('assignments')) {
  for (const [n, dir] of [[1, 'assignment-one'], [2, 'assignment-two']]) {
    const src = join(publicRoot, 'assignments', dir, 'brief.html');
    if (existsSync(src)) jobs.push({ type: 'page', folder: `Assignment${n}`, file: `Assignment${n}_brief.pdf`, src, label: `Assignment ${n}` });
  }
}
for (const job of jobs) {
  if (!existsSync(job.src)) throw new Error(`Missing ${job.src}; run npm run sync:course first.`);
}

// ---------- print-only CSS, injected at print time ----------
const pageCss = `
  /* Answers stay where the author put them, opened. */
  .callout-collapse.collapse { display: block !important; height: auto !important; visibility: visible !important; }
  .callout-btn-toggle, .callout-toggle { display: none !important; }
  /* Screen-only chrome. */
  #quarto-header, #quarto-sidebar, #quarto-margin-sidebar, #TOC, nav#TOC, .quarto-back-to-top,
  #quarto-back-to-top, .sidebar-toggle, #quarto-search { display: none !important; }
  #quarto-content, main.content, .page-columns { display: block !important; max-width: none !important; margin: 0 !important; padding: 0 !important; }
  body { font-size: 10.5pt; }
  /* Keep figures, tables, code and widgets on one page. */
  figure, .figure, .quarto-figure, img, svg, table, pre, .sourceCode, .callout, .cell-output-display,
  mjx-container[display="true"], iframe, canvas, .wbox { break-inside: avoid; page-break-inside: avoid; }
  h1, h2, h3, h4, h5 { break-after: avoid; page-break-after: avoid; }
  pre, code { white-space: pre-wrap !important; word-break: break-word; }
`;
// The dark cover photo would cost a lot of toner; print the title page light.
const slideCss = `
  .pdf-page:has(section.course-title) .slide-background { display: none !important; }
  section.course-title, section.course-title * { color: #111 !important; text-shadow: none !important; }
`;

async function settle(page) {
  await page.evaluate(async () => {
    if (window.MathJax?.startup?.promise) await window.MathJax.startup.promise;
    await document.fonts?.ready;
    await Promise.all([...document.images].map((img) => img.complete ? null : new Promise((r) => { img.onload = img.onerror = r; })));
  });
  await new Promise((r) => setTimeout(r, 800));
}

async function printPage(browser, job, out) {
  const page = await browser.newPage();
  await page.setViewport({ width: 1100, height: 1400 });
  await page.goto(pathToFileURL(job.src).href, { waitUntil: 'networkidle0', timeout: 120000 });
  await page.addStyleTag({ content: pageCss });
  await page.evaluate(() => document.querySelectorAll('details').forEach((d) => { d.open = true; }));
  await settle(page);
  const esc = job.label.replace(/&/g, '&amp;').replace(/</g, '&lt;');
  await page.pdf({
    path: out, format: 'A4', printBackground: true,
    margin: { top: '16mm', bottom: '16mm', left: '16mm', right: '16mm' },
    displayHeaderFooter: true,
    headerTemplate: '<span></span>',
    footerTemplate: `<div style="width:100%;font-size:7.5pt;color:#777;padding:0 16mm;display:flex;justify-content:space-between;font-family:Helvetica,Arial,sans-serif"><span>MPS311/439 Machine Learning · ${esc}</span><span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>`,
  });
  await page.close();
}

async function printSlides(browser, job, out) {
  const raw = out.replace(/\.pdf$/, '.raw.pdf');
  const page = await browser.newPage();
  await page.setViewport({ width: 1600, height: 900 });
  await page.goto(`${pathToFileURL(job.src).href}?print-pdf`, { waitUntil: 'networkidle0', timeout: 120000 });
  await page.waitForFunction(() => window.Reveal?.isReady?.() && document.querySelector('.pdf-page'), { timeout: 60000 });
  await page.addStyleTag({ content: slideCss });
  await settle(page);
  await page.pdf({ path: raw, printBackground: true, preferCSSPageSize: true });
  await page.close();
  await execFileAsync(python, [join(projectRoot, 'scripts/slides-handout.py'), raw, out, `MPS311/439 Machine Learning · ${job.label} · Slides`]);
  await rm(raw);
}

// ---------- run ----------
const browser = await puppeteer.launch({ executablePath: chrome, headless: true, args: ['--allow-file-access-from-files'] });
const report = [];
try {
  for (const job of jobs) {
    const dir = join(pdfRoot, job.folder);
    await mkdir(dir, { recursive: true });
    const out = join(dir, job.file);
    process.stdout.write(`${job.folder}/${job.file} ... `);
    await (job.type === 'slides' ? printSlides : printPage)(browser, job, out);
    const { stdout } = await execFileAsync(python, ['-c', 'import sys,pypdf;print(len(pypdf.PdfReader(sys.argv[1]).pages))', out]);
    console.log(`${stdout.trim()} pages`);
    report.push(`${job.folder}/${job.file}\t${stdout.trim()} pages`);
  }
} finally {
  await browser.close();
}

await writeFile(join(pdfRoot, 'README.txt'), [
  'MPS311/439 Machine Learning — printable PDFs (English).',
  'Generated by course-site/scripts/build-pdfs.mjs (npm run pdf). Do not edit by hand;',
  'change the course source, rebuild, run npm run sync:course, then npm run pdf.',
  'Slides are printed three per A4 page with lines for notes; answers in Notes are shown in place.',
  '',
  `Last run: ${new Date().toISOString()} (files rebuilt in that run):`,
  ...report,
  '',
].join('\n'));

if (drive !== 'off') {
  if (existsSync(dirname(drive))) {
    await mkdir(drive, { recursive: true });
    // --inplace keeps each Drive file (and its link) when it is replaced.
    await execFileAsync('rsync', ['-a', '--inplace', '--delete', '--exclude', 'README.txt', '--exclude', '.*', `${pdfRoot}/`, `${drive}/`]);
    console.log(`Mirrored to Google Drive: ${drive}`);
  } else {
    console.log(`Google Drive folder not found (${dirname(drive)}); skipped the Drive copy.`);
  }
}
