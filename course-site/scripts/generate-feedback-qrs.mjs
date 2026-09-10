import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { renderSVG } from 'uqr';
import sharp from 'sharp';
import course from '../src/data/course.json' with { type: 'json' };

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const projectDirectory = resolve(scriptDirectory, '..');
const materialsDirectory = resolve(projectDirectory, '..', 'lecture-slids', 'lessons');
const defaultYear = course.academicYear.replace(/[^0-9]+/g, '-').replace(/^-|-$/g, '');
const argumentsList = process.argv.slice(2);

const option = (name) => {
  const index = argumentsList.indexOf(name);
  return index === -1 ? undefined : argumentsList[index + 1];
};

const usage = () => {
  console.error(
    'Usage: npm run feedback:qr -- --base-url https://course.example.edu [--year 2026-27] [--output-dir public/feedback-qr/2026-27]',
  );
};

if (argumentsList.includes('--help')) {
  usage();
  process.exit(0);
}

const baseUrlValue = option('--base-url');
if (!baseUrlValue) {
  usage();
  process.exit(1);
}

let baseUrl;
try {
  baseUrl = new URL(baseUrlValue);
} catch {
  console.error('--base-url must be a full https URL.');
  process.exit(1);
}

if (baseUrl.protocol !== 'https:') {
  console.error('--base-url must use https.');
  process.exit(1);
}

baseUrl.pathname = baseUrl.pathname.replace(/\/$/, '');
baseUrl.search = '';
baseUrl.hash = '';

const academicYear = option('--year') ?? defaultYear;
if (!/^\d{4}-\d{2}$/.test(academicYear)) {
  console.error('--year must use the form YYYY-YY, for example 2026-27.');
  process.exit(1);
}

const outputDirectory = resolve(
  projectDirectory,
  option('--output-dir') ?? `public/feedback-qr/${academicYear}`,
);
const teachingLessons = course.lessons.filter((lesson) => lesson.type === 'teaching');
const sessions = teachingLessons.flatMap((lesson) => [
  { lesson: lesson.number, type: 'lecture', title: lesson.title },
  { lesson: lesson.number, type: 'lab', title: lesson.title },
]);

await mkdir(outputDirectory, { recursive: true });

const manifest = await Promise.all(
  sessions.map(async (session) => {
    const sessionKey = `${academicYear}-lesson-${session.lesson}-${session.type}`;
    const feedbackUrl = new URL('/feedback/', baseUrl);
    feedbackUrl.searchParams.set('year', academicYear);
    feedbackUrl.searchParams.set('session', sessionKey);
    const filename = `lesson-${session.lesson}-${session.type}.svg`;
    const pngFilename = `lesson-${session.lesson}-${session.type}.png`;
    const svg = renderSVG(feedbackUrl.toString(), { ecc: 'M', border: 2 });
    const png = await sharp(Buffer.from(svg)).png().toBuffer();
    await Promise.all([
      writeFile(resolve(outputDirectory, filename), svg, 'utf8'),
      writeFile(resolve(outputDirectory, pngFilename), png),
    ]);
    const lessonDirectory = resolve(materialsDirectory, `lesson${Number(session.lesson)}`, session.type);
    const lessonAssets = session.type === 'lecture'
      ? [
          resolve(lessonDirectory, 'feedback-qr.svg'),
          resolve(lessonDirectory, 'public', 'feedback-qr.svg'),
        ]
      : [resolve(lessonDirectory, 'feedback-qr.svg')];
    await Promise.all(lessonAssets.map(async (lessonAsset) => {
      await mkdir(dirname(lessonAsset), { recursive: true });
      await writeFile(lessonAsset, svg, 'utf8');
    }));
    if (session.type === 'lab') {
      await writeFile(resolve(lessonDirectory, 'feedback-qr.png'), png);
    }
    return {
      filename,
      pngFilename,
      session: sessionKey,
      label: `Lesson ${session.lesson} · ${session.type === 'lecture' ? 'Lecture' : 'Lab'} — ${session.title}`,
      url: feedbackUrl.toString(),
    };
  }),
);

await writeFile(
  resolve(outputDirectory, 'manifest.json'),
  `${JSON.stringify({ academicYear, baseUrl: baseUrl.toString(), sessions: manifest }, null, 2)}\n`,
  'utf8',
);

console.log(`Generated ${manifest.length} feedback QR codes in ${outputDirectory}`);
manifest.forEach(({ filename, url }) => console.log(`- ${filename}: ${url}`));
