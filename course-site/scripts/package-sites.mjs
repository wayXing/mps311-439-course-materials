import { cp, mkdir, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const astroDist = join(root, 'dist');
const hostingConfig = join(root, '.openai/hosting.json');
const stage = join(root, '.sites-stage');
const sitesDist = join(stage, 'dist');
const client = join(sitesDist, 'client');
const server = join(sitesDist, 'server');
const omitPdfs = process.argv.includes('--omit-pdfs');
const shellOnly = process.argv.includes('--shell-only');
const archive = join(
  root,
  `.sites-artifacts/course-site${shellOnly ? '-shell-preview' : omitPdfs ? '-preview' : ''}.tgz`,
);

async function exists(path) {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

if (!(await exists(join(astroDist, 'index.html')))) {
  throw new Error('Run the Astro production build before packaging the site.');
}

if (!(await exists(hostingConfig))) {
  throw new Error('Missing .openai/hosting.json. Create or connect the Sites project first.');
}

const worker = `function withSecurityHeaders(response) {
  const headers = new Headers(response.headers);
  headers.set("x-content-type-options", "nosniff");
  headers.set("referrer-policy", "strict-origin-when-cross-origin");
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

async function asset(env, request, path) {
  if (!env.ASSETS) {
    return new Response("Sites asset binding missing", { status: 500 });
  }
  const url = new URL(request.url);
  url.pathname = path;
  return env.ASSETS.fetch(new Request(url, request));
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const originalPath = url.pathname;
    const candidates = [originalPath];

    if (originalPath === "/") {
      candidates.push("/index.html");
    } else if (originalPath.endsWith("/")) {
      candidates.push(originalPath + "index.html");
    } else if (!originalPath.split("/").at(-1)?.includes(".")) {
      candidates.push(originalPath + "/index.html");
    }

    for (const path of candidates) {
      const response = await asset(env, request, path);
      if (response.ok) return withSecurityHeaders(response);
    }

    return new Response("Page not found", {
      status: 404,
      headers: { "content-type": "text/plain; charset=utf-8" },
    });
  },
};
`;

await rm(stage, { recursive: true, force: true });
await mkdir(client, { recursive: true });
await mkdir(server, { recursive: true });
await mkdir(dirname(archive), { recursive: true });

await cp(astroDist, client, {
  recursive: true,
  filter: (source) =>
    !source.endsWith('/.DS_Store')
    && !(omitPdfs && source.endsWith('.pdf'))
    && !(shellOnly && source.includes('/materials')),
});
await mkdir(join(sitesDist, '.openai'), { recursive: true });
await writeFile(
  join(sitesDist, '.openai/hosting.json'),
  await readFile(hostingConfig),
);
await writeFile(join(server, 'index.js'), worker);

const tar = spawnSync('tar', ['-C', stage, '-czf', archive, 'dist'], {
  cwd: root,
  stdio: 'inherit',
});
if (tar.status !== 0) throw new Error('Could not create the Sites archive.');

const listing = spawnSync('tar', ['-tzf', archive], {
  cwd: root,
  encoding: 'utf8',
});
if (listing.status !== 0) throw new Error('Could not validate the Sites archive.');

for (const required of [
  'dist/server/index.js',
  'dist/client/index.html',
  'dist/client/feedback/index.html',
  'dist/.openai/hosting.json',
]) {
  if (!listing.stdout.split('\n').includes(required)) {
    throw new Error(`Sites archive is missing ${required}.`);
  }
}

console.log(archive);
