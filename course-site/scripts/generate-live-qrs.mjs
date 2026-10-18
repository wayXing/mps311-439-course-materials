// The course's one QR code, pointing at the live classroom page (/live/).
// Written to public/live-qr/course.svg for the lecturer's screen and copied next to every
// lesson's slide deck as live-qr.svg. Run again only if the domain changes.
import { copyFile, mkdir, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { renderSVG } from 'uqr';

const siteDirectory = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const liveUrl = 'https://ml.wxing.me/live/';
const outputDirectory = resolve(siteDirectory, 'public/live-qr');
await mkdir(outputDirectory, { recursive: true });
const target = resolve(outputDirectory, 'course.svg');
await writeFile(target, renderSVG(liveUrl, { border: 1 }));

for (let lesson = 1; lesson <= 10; lesson += 1) {
  const deckDirectory = resolve(siteDirectory, `../lecture-slids/lessons/lesson${lesson}/lecture/slides`);
  if (existsSync(deckDirectory)) await copyFile(target, resolve(deckDirectory, 'live-qr.svg'));
}
console.log(`QR code for ${liveUrl}`);
