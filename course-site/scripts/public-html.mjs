import { readdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

// Skip inline JavaScript/CSS: expressions such as `value<a.property` are not
// HTML links, and embedded icon styles must not be mistaken for PDF controls.
const anchors = /<script\b[^>]*>[\s\S]*?<\/script\s*>|<style\b[^>]*>[\s\S]*?<\/style\s*>|(<a(?=[\s>])[^>]*>[\s\S]*?<\/a\s*>)/gi;

export function isPdfLink(anchor) {
  const href = anchor.match(/\bhref\s*=\s*["']([^"']*)["']/i)?.[1] ?? '';
  const label = anchor.replace(/<[^>]*>/g, '').trim();
  return /\.pdf(?:[?#]|$)/i.test(href)
    || /\bbi-file-pdf(?:-fill)?\b/i.test(anchor)
    || /^(?:download\s+)?pdf$/i.test(label);
}

export function pdfLinks(html) {
  return [...html.matchAll(anchors)].map((match) => match[1]).filter((anchor) => anchor && isPdfLink(anchor));
}

function withoutPdfLinks(html) {
  return html
    .replace(/<div\b[^>]*class=["']quarto-alternate-formats["'][^>]*>[\s\S]*?<\/div>/gi, (formats) => {
      const filtered = formats.replace(/<li\b[^>]*>[\s\S]*?<\/li>/gi, (item) => (
        pdfLinks(item).length ? '' : item
      ));
      return /<li\b/i.test(filtered) ? filtered : '';
    })
    .replace(anchors, (match, anchor) => anchor && isPdfLink(anchor) ? '' : match);
}

export async function* htmlFiles(directory) {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) yield* htmlFiles(path);
    else if (entry.isFile() && entry.name.endsWith('.html')) yield path;
  }
}

export async function removePublicPdfLinks(directory) {
  let removed = 0;
  for await (const path of htmlFiles(directory)) {
    const html = await readFile(path, 'utf8');
    const filtered = withoutPdfLinks(html);
    removed += pdfLinks(html).length;
    if (filtered !== html) await writeFile(path, filtered);
  }
  return removed;
}
