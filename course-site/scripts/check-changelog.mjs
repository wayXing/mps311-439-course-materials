// Checks src/data/changelog.json, the public changelog shown at /changelog/.
// 1. The data is well formed: newest release first, unique versions, known change types.
// 2. Every release is recorded: if the public mirror (../mps311-439-course-materials) has commits
//    dated after the newest changelog release, the check fails until a release entry is added.
//    Where the mirror is not checked out (for example on the deployment server), step 2 is skipped.
import { execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const projectRoot = join(dirname(fileURLToPath(import.meta.url)), '..');
const mirror = join(projectRoot, '..', 'mps311-439-course-materials');
const changelog = JSON.parse(await readFile(join(projectRoot, 'src/data/changelog.json'), 'utf8'));
const types = new Set(['feedback', 'added', 'changed', 'fixed']);
const problems = [];

const releases = changelog.releases ?? [];
if (releases.length === 0) problems.push('There are no releases.');
const seen = new Set();
releases.forEach((release, index) => {
  const name = `v${release.version}`;
  if (seen.has(release.version)) problems.push(`${name} appears twice.`);
  seen.add(release.version);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(release.date ?? '')) problems.push(`${name} has no YYYY-MM-DD date.`);
  if (index > 0 && release.date > releases[index - 1].date) problems.push(`${name} is newer than the release above it; keep the newest first.`);
  if (!release.title) problems.push(`${name} has no title.`);
  if (!release.items?.length) problems.push(`${name} lists no changes.`);
  for (const item of release.items ?? []) {
    if (!types.has(item.type)) problems.push(`${name} has an unknown change type "${item.type}".`);
    if (!item.text?.trim()) problems.push(`${name} has an empty change.`);
  }
});

const latest = releases[0];
if (latest && existsSync(join(mirror, '.git'))) {
  const unrecorded = execFileSync('git', ['-C', mirror, 'log', '--format=%ad %s', '--date=short'], { encoding: 'utf8' })
    .split('\n')
    .filter((line) => line && line.slice(0, 10) > latest.date);
  if (unrecorded.length) {
    problems.push(
      `The public mirror has ${unrecorded.length} commit(s) after the latest changelog release v${latest.version} (${latest.date}). `
      + 'Add a release to src/data/changelog.json before publishing:\n  ' + unrecorded.join('\n  '),
    );
  }
}

if (problems.length) {
  console.error(`Changelog check failed:\n- ${problems.join('\n- ')}`);
  process.exit(1);
}
console.log(`Changelog OK: latest release v${latest.version} (${latest.date}), ${releases.length} releases.`);
