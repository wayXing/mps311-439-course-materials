import { readdir, rm, stat } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const output = join(root, 'dist');

async function removePdfs(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  for (const entry of entries) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) {
      await removePdfs(path);
    } else if (entry.isFile() && entry.name.toLowerCase().endsWith('.pdf')) {
      await rm(path);
    }
  }
}

try {
  if ((await stat(output)).isDirectory()) await removePdfs(output);
} catch (error) {
  if (error && typeof error === 'object' && 'code' in error && error.code === 'ENOENT') {
    throw new Error('Expected the Astro build output in dist before removing deployment PDFs.');
  }
  throw error;
}
