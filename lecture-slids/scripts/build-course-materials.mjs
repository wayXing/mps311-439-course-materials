import { execFile } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';
import { lessonPaths, teachingLessons } from './course-lessons.mjs';

const execFileAsync = promisify(execFile);
const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');

for (const lesson of teachingLessons) {
  const { quarto, slideSource: slide, labSource: lab } = lessonPaths(lesson);

  console.log(`Building Lesson ${lesson} slides...`);
  if (quarto) {
    await execFileAsync('quarto', [
      'render', slide, '--to', 'revealjs', '--quiet',
    ], { cwd: projectRoot, maxBuffer: 20 * 1024 * 1024 });
  } else {
    await execFileAsync('npm', [
      'run', 'build:standalone', '--', slide,
      '--out', 'dist', '--base', './',
    ], { cwd: projectRoot, maxBuffer: 20 * 1024 * 1024 });
  }

  console.log(`Rendering Lesson ${lesson} lab...`);
  await execFileAsync('quarto', [
    'render',
    lab,
    '--to',
    'html',
    '--quiet',
  ], { cwd: projectRoot, maxBuffer: 20 * 1024 * 1024 });
}

console.log(`Built ${teachingLessons.length} slide decks and ${teachingLessons.length} lab worksheets.`);
