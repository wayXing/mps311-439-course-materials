// Collects what a release is about to make public, so public-release-reviewer reads only the changes.
//   node scripts/collect-release-changes.mjs <out.txt>   after `npm run sync:public`, before pushing
//   node scripts/collect-release-changes.mjs --record     after the release is live
// Two sources of public content:
// 1. The public mirror (../mps311-439-course-materials): its working tree against its last commit,
//    which is the last release. Covers the site source (course.json, changelog, pages) and the
//    lesson sources. Notebooks are compared by cell text only; binary files are only listed.
// 2. Files the site serves that are not in the mirror (public/, except public/materials, which is
//    rendered from mirror sources): compared by content hash with .release-manifest.json, written
//    by --record. HTML is reduced to its visible text.
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { readFile, readdir, writeFile } from 'node:fs/promises';
import { dirname, extname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const projectRoot = join(dirname(fileURLToPath(import.meta.url)), '..');
const mirror = join(projectRoot, '..', 'mps311-439-course-materials');
const publicRoot = join(projectRoot, 'public');
const manifestPath = join(projectRoot, '.release-manifest.json');
const binary = new Set(['.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg', '.ico', '.pdf', '.npz', '.zip', '.mp4', '.woff', '.woff2']);

const git = (...args) => execFileSync('git', ['-C', mirror, ...args], { encoding: 'utf8', maxBuffer: 1 << 28 });

async function walk(dir) {
  const out = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) out.push(...await walk(path));
    else out.push(path);
  }
  return out;
}

async function siteHashes() {
  const files = (await walk(publicRoot)).filter((path) => !relative(publicRoot, path).startsWith('materials/'));
  const hashes = {};
  for (const path of files) hashes[relative(projectRoot, path)] = createHash('sha256').update(await readFile(path)).digest('hex');
  return hashes;
}

const visibleText = (html) => html
  .replace(/<(script|style)[\s\S]*?<\/\1>/gi, ' ')
  .replace(/<[^>]+>/g, ' ')
  .replace(/&nbsp;/g, ' ').replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>')
  .replace(/[ \t]+/g, ' ').replace(/\n\s*\n+/g, '\n').trim();

// Data tables: the header and first rows are enough to see what columns are published.
const dataPreview = (csv) => {
  const lines = csv.split('\n').filter(Boolean);
  return `${lines.slice(0, 4).join('\n')}\n… ${lines.length - 1} data rows in total`;
};

const cellText = (json) => {
  try {
    return JSON.parse(json).cells.map((cell, i) => `--- cell ${i + 1} (${cell.cell_type})\n${[].concat(cell.source).join('')}`).join('\n');
  } catch { return json; }
};

if (process.argv[2] === '--record') {
  await writeFile(manifestPath, JSON.stringify(await siteHashes(), null, 2) + '\n');
  console.log(`Recorded ${manifestPath}`);
  process.exit(0);
}

const outPath = process.argv[2];
if (!outPath) {
  console.error('Usage: node scripts/collect-release-changes.mjs <out.txt> | --record');
  process.exit(1);
}
const sections = [];

// 1. Public mirror against its last commit.
const changed = git('status', '--porcelain', '--untracked-files=all').split('\n').filter(Boolean)
  .map((line) => ({ status: line.slice(0, 2).trim(), path: line.slice(3).replace(/^.* -> /, '') }));
for (const { status, path } of changed) {
  const ext = extname(path).toLowerCase();
  const head = `## Mirror: ${path} (${status === '??' ? 'new' : status === 'D' ? 'deleted' : 'changed'})`;
  if (status === 'D' || binary.has(ext)) { sections.push(`${head}\n(binary or deleted file, not shown)`); continue; }
  if (ext === '.ipynb') {
    const before = status === '??' ? '' : cellText(git('show', `HEAD:${path}`));
    const after = cellText(await readFile(join(mirror, path), 'utf8'));
    const oldLines = new Set(before.split('\n'));
    const added = after.split('\n').filter((line) => !oldLines.has(line));
    sections.push(`${head}\nLines new in this notebook's cells:\n${added.join('\n')}`);
    continue;
  }
  const body = ext === '.csv'
    ? dataPreview(await readFile(join(mirror, path), 'utf8'))
    : status === '??'
      ? await readFile(join(mirror, path), 'utf8')
      : git('diff', '--unified=2', 'HEAD', '--', path);
  sections.push(`${head}\n${body}`);
}

// 2. Site files outside the mirror, against the last recorded release.
const previous = existsSync(manifestPath) ? JSON.parse(await readFile(manifestPath, 'utf8')) : {};
const current = await siteHashes();
for (const [path, hash] of Object.entries(current)) {
  if (previous[path] === hash) continue;
  const ext = extname(path).toLowerCase();
  const head = `## Site: ${path} (${previous[path] ? 'changed' : 'new since last recorded release'})`;
  if (binary.has(ext)) { sections.push(`${head}\n(binary file, not shown)`); continue; }
  const text = await readFile(join(projectRoot, path), 'utf8');
  sections.push(`${head}\n${ext === '.html' ? visibleText(text) : ext === '.csv' ? dataPreview(text) : text}`);
}

const intro = changed.length || sections.length
  ? `Public changes in this release: ${changed.length} mirror file(s), ${sections.length - changed.length} site file(s).`
  : 'No public changes found. Did you run npm run sync:public first?';
await writeFile(outPath, `${intro}\n\n${sections.join('\n\n')}\n`);
console.log(`${intro}\nWritten to ${outPath}`);
