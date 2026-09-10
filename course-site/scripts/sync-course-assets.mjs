import { execFile } from 'node:child_process';
import { cp, mkdir, mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';

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

console.log(`Synced ${copied} public course resources (${rendered} notebooks rendered as HTML).`);
