import { execFile } from 'node:child_process';
import { cp, mkdir, mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';
import { removePublicPdfLinks } from './public-html.mjs';

const execFileAsync = promisify(execFile);

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const lessonRoot = resolve(projectRoot, '../lecture-slids/lessons');
const outputRoot = join(projectRoot, 'public/materials');
const course = JSON.parse(await readFile(join(projectRoot, 'src/data/course.json'), 'utf8'));

await rm(outputRoot, { recursive: true, force: true });
await mkdir(outputRoot, { recursive: true });

let copied = 0;
let rendered = 0;

for (const lesson of course.lessons) {
  for (const resource of lesson.resources) {
    if (resource.group === 'Archive' && !resource.source.split('/').includes('uos-2025')) {
      throw new Error(`Only complete UoS 2025 archives may be published: ${resource.source}`);
    }
    const source = join(lessonRoot, resource.source);
    const target = join(projectRoot, 'public', resource.target);

    await stat(source).catch(() => {
      throw new Error(`Missing course resource: ${source}`);
    });
    await mkdir(dirname(target), { recursive: true });

    if (resource.render === 'quarto') {
      const renderRoot = await mkdtemp(join(tmpdir(), 'course-demo-'));
      const notebookName = basename(source);
      const renderSource = join(renderRoot, notebookName);
      const notebook = JSON.parse(await readFile(source, 'utf8'));

      // A markdown cell beginning with `---` is a valid visual divider in Jupyter,
      // but Quarto interprets it as YAML front matter. Normalise only the temporary
      // render copy; the source and downloadable notebook remain unchanged.
      for (const cell of notebook.cells ?? []) {
        if (cell.cell_type === 'markdown' && Array.isArray(cell.source) && cell.source[0] === '---\n') {
          cell.source[0] = '***\n';
        }
      }

      await writeFile(renderSource, JSON.stringify(notebook));
      try {
        await execFileAsync('quarto', [
          'render',
          renderSource,
          '--to',
          'html',
          '--no-execute',
          '--output-dir',
          target,
          '--quiet',
        ]);
      }
      finally {
        await rm(renderRoot, { recursive: true, force: true });
      }
      await cp(source, join(target, notebookName));
      rendered += 1;
    }
    else if (resource.directory) {
      await cp(source, target, {
        recursive: true,
        filter: (path) => !path.endsWith('/404.html') && !path.endsWith('/_redirects'),
      });
    }
    else {
      await cp(source, target);
    }

    if (resource.kind === 'worksheet') {
      const feedbackQrSource = join(dirname(source), 'feedback-qr.svg');
      const feedbackQrTarget = join(dirname(target), 'feedback-qr.svg');
      await stat(feedbackQrSource).catch(() => {
        throw new Error(`Missing lab feedback QR code: ${feedbackQrSource}`);
      });
      await mkdir(dirname(feedbackQrTarget), { recursive: true });
      await cp(feedbackQrSource, feedbackQrTarget);
    }

    if (resource.downloadSource) {
      if (!resource.downloadTarget) {
        throw new Error(`Missing downloadTarget for ${resource.downloadSource}`);
      }
      const downloadSource = join(lessonRoot, resource.downloadSource);
      const downloadTarget = join(projectRoot, 'public', resource.downloadTarget);
      await stat(downloadSource).catch(() => {
        throw new Error(`Missing downloadable resource: ${downloadSource}`);
      });
      await mkdir(dirname(downloadTarget), { recursive: true });
      await cp(downloadSource, downloadTarget);
    }
    copied += 1;
  }
}

// Quarto's alternate-format controls belong to the Blackboard archive. Strip
// PDF links only from the website copies, including links labelled PDF whose
// generated href incorrectly points to another HTML file.
const removedPdfLinks = await removePublicPdfLinks(outputRoot);
console.log(`Synced ${copied} public course resources (${rendered} notebooks rendered as HTML; ${removedPdfLinks} PDF links removed).`);

// Publish only the approved student brief and its three datasets. Assessment
// sources and teacher notes remain outside the public GitHub mirror.
const assignmentRoot = resolve(projectRoot, '../assessment/2026/assignment1');
const assignmentOutput = join(projectRoot, 'public/assignments/assignment-one');
await rm(join(projectRoot, 'public/assignments'), { recursive: true, force: true });
if (await stat(assignmentRoot).then(() => true).catch(() => false)) {
  await mkdir(assignmentOutput, { recursive: true });
  await cp(join(assignmentRoot, 'assignment_1_california_en_v10.html'), join(assignmentOutput, 'brief.html'));
  for (const name of ['train.csv', 'test.csv', 'full.csv']) {
    await cp(join(assignmentRoot, 'data', name), join(assignmentOutput, name));
  }
  await removePublicPdfLinks(assignmentOutput);
  console.log('Synced Assignment 1 student brief and datasets.');
}
