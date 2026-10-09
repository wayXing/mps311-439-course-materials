import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { dirname, join, resolve, extname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-chromium';

const here = dirname(fileURLToPath(import.meta.url));
const stem = process.argv[2] ?? 'slide_v2';
if (!/^slide(?:_v\d+)?$/.test(stem)) throw new Error('Invalid slide source name');
const source = await readFile(join(here, `${stem}.qmd`), 'utf8');
const expectedSlides = (source.match(/^## /gm) ?? []).length + 1;
const mimeTypes = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript',
  '.css': 'text/css', '.woff2': 'font/woff2', '.webp': 'image/webp', '.svg': 'image/svg+xml' };
const server = createServer(async (request, response) => {
  const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
  const path = resolve(here, `.${pathname}`);
  if (!path.startsWith(`${here}/`)) {
    response.writeHead(403).end();
    return;
  }
  try {
    const body = await readFile(path);
    response.writeHead(200, { 'Content-Type': mimeTypes[extname(path)] ?? 'application/octet-stream' });
    response.end(body);
  } catch {
    response.writeHead(404).end();
  }
});
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));

const browser = await chromium.launch({
  executablePath: process.env.LESSON2_CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: true,
  args: ['--no-sandbox'],
});
try {
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
  await page.goto(`http://127.0.0.1:${server.address().port}/${stem}.html?print-pdf`, { waitUntil: 'networkidle' });
  await page.waitForFunction(() => window.Reveal?.isReady?.(), { timeout: 30000 });
  await page.waitForFunction(() => [...document.querySelectorAll('.math')].every((el) => el.querySelector('mjx-container, .MathJax')), { timeout: 30000 });
  if (await page.locator('mjx-merror, [data-mjx-error]').count()) throw new Error('Math rendering failed');
  await page.waitForTimeout(500);
  const count = await page.locator('.reveal .slides section.slide, .reveal .slides section.course-title').count();
  if (count !== expectedSlides) throw new Error(`Expected ${expectedSlides} slides; rendered ${count}`);
  await page.pdf({ path: join(here, `${stem}.pdf`), printBackground: true, preferCSSPageSize: true });
  console.log(`Exported ${count} Quarto/RevealJS slides to ${stem}.pdf`);
} finally {
  await browser.close();
  await new Promise((resolve) => server.close(resolve));
}
