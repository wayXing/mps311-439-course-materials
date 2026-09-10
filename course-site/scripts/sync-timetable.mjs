import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const sourceName = 'MPS311_439_TimeTable-byLessons-2026.csv';
const source = resolve(projectRoot, `../${sourceName}`);
const jsonTarget = join(projectRoot, 'src/data/timetable.json');
const csvTarget = join(projectRoot, 'public/data/course-timetable.csv');

function parseCsv(input) {
  const rows = [];
  let row = [];
  let field = '';
  let quoted = false;

  for (let index = 0; index < input.length; index += 1) {
    const character = input[index];

    if (quoted) {
      if (character === '"' && input[index + 1] === '"') {
        field += '"';
        index += 1;
      }
      else if (character === '"') {
        quoted = false;
      }
      else {
        field += character;
      }
    }
    else if (character === '"') {
      quoted = true;
    }
    else if (character === ',') {
      row.push(field);
      field = '';
    }
    else if (character === '\n') {
      row.push(field);
      rows.push(row);
      row = [];
      field = '';
    }
    else if (character !== '\r') {
      field += character;
    }
  }

  if (field || row.length) {
    row.push(field);
    rows.push(row);
  }
  return rows;
}

function outcomes(value) {
  return value
    .split('\n')
    .map((item) => item.replace(/^\s*•\s*/, '').trim())
    .filter(Boolean);
}

const csv = await readFile(source, 'utf8');
const [headers, ...rows] = parseCsv(csv);
const records = rows.map((values) => Object.fromEntries(headers.map((header, index) => [
  header.replace(/^\uFEFF/, '').trim(),
  values[index]?.trim() ?? '',
])));

const events = records
  .filter((record) => record.Contents || record.Theme || record['Assessment info'])
  .map((record) => ({
    lesson: record.Lesson ? Number(record.Lesson) : null,
    type: record.Type || (record['Assessment info'] ? 'Assessment' : 'Course event'),
    title: record.Contents || record.Theme || record['Assessment info'],
    theme: record.Theme,
    coreOutcomes: outcomes(record['Core Learning Outcomes (Required for all students)'] ?? ''),
    advancedOutcomes: outcomes(record['Advanced Learning Outcomes (for MPS439 students)'] ?? ''),
    assessment: record['Assessment info'],
    lectureDate: record['Lecture date'],
    labDate: record['Lab date'],
  }));

const timetable = {
  source: 'MPS311 _439_TimeTable-byLessons-2026.xlsx',
  academicYear: '2026/27',
  basis: 'lesson',
  events,
};

await mkdir(dirname(jsonTarget), { recursive: true });
await mkdir(dirname(csvTarget), { recursive: true });
await writeFile(jsonTarget, `${JSON.stringify(timetable, null, 2)}\n`);
await copyFile(source, csvTarget);

console.log(`Synced ${events.length} timetable events.`);
