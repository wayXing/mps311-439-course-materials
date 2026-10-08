import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-chromium';

const here = dirname(fileURLToPath(import.meta.url));
const html = await readFile(join(here, 'slide.html'));
const server = createServer((request, response) => {
  if (request.url?.startsWith('/slide.html')) {
    response.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    response.end(html);
  } else {
    response.writeHead(404);
    response.end();
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
  await page.goto(`http://127.0.0.1:${server.address().port}/slide.html?print-pdf`, { waitUntil: 'networkidle' });
  await page.waitForFunction(() => window.Reveal?.isReady?.(), { timeout: 30000 });
  await page.waitForTimeout(1500);
  const count = await page.locator('.reveal .slides section.slide, .reveal .slides section.course-title').count();
  if (count !== 21) throw new Error(`Expected 21 slides; rendered ${count}`);
  await page.pdf({ path: join(here, 'slide.pdf'), printBackground: true, preferCSSPageSize: true });
  console.log(`Exported ${count} Quarto/RevealJS slides to slide.pdf`);
} finally {
  await browser.close();
  await new Promise((resolve) => server.close(resolve));
}
