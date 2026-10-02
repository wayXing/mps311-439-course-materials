import { readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const write = process.argv.includes('--write');
if (process.argv.some((arg) => arg !== '--write' && arg !== process.argv[0] && arg !== process.argv[1])) {
  throw new Error('Usage: node scripts/update-feedback-qr-copy.mjs [--write]');
}

const lessons = resolve(dirname(fileURLToPath(import.meta.url)), '../../lecture-slids/lessons');
const start = '<!-- COURSE_FEEDBACK_QR:START -->';
const end = '<!-- COURSE_FEEDBACK_QR:END -->';
const oldLecture = 'Scan this code to share anonymous feedback or post a question for this lecture.';
const newLecture = 'What would help you learn better next time?';
const oldLab = /Scan the code to share anonymous feedback or post a question for this lab\. It opens the correct \*\*Lesson \d{2} lab\*\* record automatically\./g;
const newLab = "What would help you learn better next time?\n\nScan to share anonymous feedback on today's lab.";
const changes = [];

for (let number = 2; number <= 10; number++) {
  for (const kind of ['lecture', 'lab']) {
    const path = resolve(lessons, `lesson${number}`, kind, kind === 'lecture' ? 'slide.md' : 'lab_worksheet.md');
    const source = await readFile(path, 'utf8');
    if (source.split(start).length !== 2 || source.split(end).length !== 2) {
      throw new Error(`Expected exactly one marked QR section: ${path}`);
    }
    const before = source.indexOf(start) + start.length;
    const after = source.indexOf(end);
    if (after <= before) throw new Error(`Invalid QR markers: ${path}`);
    const section = source.slice(before, after);
    let updated;
    if (kind === 'lecture') {
      const old = `<p class="text-xl mb-4">${oldLecture}</p>`;
      const replacement = `<p class="text-3xl mb-4">${newLecture}</p>\n\n<p class="text-2xl mb-4">Scan to share anonymous feedback on today's lecture.</p>`;
      if (section.includes(replacement) && !section.includes(old)) updated = section;
      else {
        if (section.split(old).length !== 2) throw new Error(`Expected one lecture prompt: ${path}`);
        updated = section.replace(old, replacement);
      }
    } else {
      const matches = [...section.matchAll(oldLab)];
      if (section.includes(newLab) && matches.length === 0) updated = section;
      else {
        if (matches.length !== 1) throw new Error(`Expected one lab prompt: ${path}`);
        updated = section.replace(oldLab, newLab);
      }
    }
    if (updated !== section) changes.push({ path, result: source.slice(0, before) + updated + source.slice(after) });
  }
}

if (write) {
  for (const { path, result } of changes) await writeFile(path, result, 'utf8');
}
for (const { path } of changes) console.log(`${write ? 'Updated' : 'Would update'} ${path}`);
console.log(`${write ? 'Updated' : 'Ready to update'} ${changes.length} files; Lesson 1 excluded.`);
