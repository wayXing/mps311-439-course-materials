import { readFile, stat } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { teachingLessons } from './course-lessons.mjs';

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const problems = [];

async function fileStat(relativePath) {
  return stat(join(projectRoot, relativePath)).catch(() => {
    problems.push(`Missing: ${relativePath}`);
    return null;
  });
}

for (const lesson of teachingLessons) {
  const slideSource = lesson <= 2
    ? `lessons/lesson${lesson}/lecture/slides/slide.qmd`
    : `lessons/lesson${lesson}/lecture/slide.md`;
  const slideBuild = lesson <= 2
    ? `lessons/lesson${lesson}/lecture/slides/slide.html`
    : `lessons/lesson${lesson}/lecture/dist/index.html`;
  const labSource = `lessons/lesson${lesson}/lab/lab_worksheet.${lesson === 2 ? "qmd" : "md"}`;
  const labBuild = `lessons/lesson${lesson}/lab/lab_worksheet.html`;
  const solution = `lessons/lesson${lesson}/lab/lab_solution.ipynb`;

  const [slideSourceStat, slideBuildStat, labSourceStat, labBuildStat, solutionStat] = await Promise.all([
    fileStat(slideSource),
    fileStat(slideBuild),
    fileStat(labSource),
    fileStat(labBuild),
    fileStat(solution),
  ]);

  if (slideSourceStat && slideBuildStat && slideBuildStat.mtimeMs < slideSourceStat.mtimeMs) {
    problems.push(`Stale slide build: ${slideBuild}`);
  }
  if (labSourceStat && labBuildStat && labBuildStat.mtimeMs < labSourceStat.mtimeMs) {
    problems.push(`Stale lab build: ${labBuild}`);
  }
  if (solutionStat) {
    try {
      JSON.parse(await readFile(join(projectRoot, solution), 'utf8'));
    }
    catch {
      problems.push(`Invalid notebook JSON: ${solution}`);
    }
  }
}

if (problems.length > 0) {
  throw new Error(`Course material check failed:\n${problems.map((problem) => `- ${problem}`).join('\n')}`);
}

console.log(`Validated ${teachingLessons.length} slide decks, lab worksheets, and solution notebooks.`);
