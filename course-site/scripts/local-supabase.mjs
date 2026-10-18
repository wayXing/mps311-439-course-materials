// Local stand-in for the Supabase project, for testing the classroom pages before release.
//
// It needs two containers (see "Local test stack" in AGENTS.md): Postgres with every file in
// supabase/migrations applied, and PostgREST on port 54330. This script then serves, on
// port 54321, what the site expects from Supabase:
//   /rest/v1/*                    -> PostgREST (anon role, or service_role for the edge function)
//   /functions/v1/course-feedback -> supabase/functions/course-feedback/index.ts, run in Node
// The admin password for the local stack is read from LOCAL_ADMIN_PASSWORD in .env.localstack.
//
// Run: node --experimental-strip-types scripts/local-supabase.mjs
import { createHmac, createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const siteDirectory = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const restUrl = 'http://localhost:54330';
const port = 54321;
const jwtSecret = 'local-only-jwt-secret-for-the-course-site-test-stack-0001';

const env = Object.fromEntries(
  readFileSync(resolve(siteDirectory, '.env.localstack'), 'utf8')
    .split('\n')
    .filter((line) => line.includes('=') && !line.trim().startsWith('#'))
    .map((line) => [line.slice(0, line.indexOf('=')).trim(), line.slice(line.indexOf('=') + 1).trim()]),
);

if (!env.LOCAL_ADMIN_PASSWORD) {
  console.error('Set LOCAL_ADMIN_PASSWORD in .env.localstack first.');
  process.exit(1);
}

const base64Url = (text) => Buffer.from(text).toString('base64url');
const signJwt = (role) => {
  const head = base64Url(JSON.stringify({ alg: 'HS256', typ: 'JWT' }));
  const body = base64Url(JSON.stringify({ role, exp: 4102444800 /* fixed, so tokens survive a restart */ }));
  const signature = createHmac('sha256', jwtSecret).update(`${head}.${body}`).digest('base64url');
  return `${head}.${body}.${signature}`;
};
const anonJwt = signJwt('anon');
const serviceJwt = signJwt('service_role');

// The edge function reads these through Deno.env and registers itself through Deno.serve.
let handler;
globalThis.Deno = {
  env: {
    get: (name) => ({
      SUPABASE_URL: `http://localhost:${port}`,
      SUPABASE_SERVICE_ROLE_KEY: serviceJwt,
      COURSE_ADMIN_PASSWORD_HASH: createHash('sha256').update(env.LOCAL_ADMIN_PASSWORD ?? '').digest('hex'),
    })[name],
  },
  serve: (fn) => { handler = fn; },
};
await import(pathToFileURL(resolve(siteDirectory, 'supabase/functions/course-feedback/index.ts')).href);

// Lower-case names, so they replace the edge function's own CORS headers instead of doubling them.
const cors = {
  'access-control-allow-origin': '*',
  'access-control-allow-headers': 'apikey, authorization, content-type, prefer',
  'access-control-allow-methods': 'GET, POST, OPTIONS',
};

const readBody = (request) => new Promise((done) => {
  const chunks = [];
  request.on('data', (chunk) => chunks.push(chunk));
  request.on('end', () => done(Buffer.concat(chunks)));
});

createServer(async (request, response) => {
  try {
    await serve(request, response);
  } catch (error) {
    // A restarting database or PostgREST must not stop the stand-in.
    response.writeHead(502, cors);
    response.end(JSON.stringify({ error: String(error) }));
  }
}).listen(port, '127.0.0.1', () => console.log(`Local Supabase stand-in on http://localhost:${port}`));

async function serve(request, response) {
  const url = new URL(request.url, `http://localhost:${port}`);
  const body = await readBody(request);
  const headers = { ...request.headers };

  if (url.pathname.startsWith('/functions/v1/course-feedback')) {
    const edgeRequest = new Request(url, {
      method: request.method,
      // Any local port may call in; the function itself only allows its listed origins.
      headers: { ...headers, origin: 'http://localhost:4321', 'x-forwarded-for': '127.0.0.1' },
      body: ['GET', 'HEAD'].includes(request.method) ? undefined : body,
    });
    const edgeResponse = await handler(edgeRequest);
    response.writeHead(edgeResponse.status, { ...Object.fromEntries(edgeResponse.headers), ...cors });
    response.end(Buffer.from(await edgeResponse.arrayBuffer()));
    return;
  }

  if (request.method === 'OPTIONS') {
    response.writeHead(204, cors);
    response.end();
    return;
  }

  if (url.pathname.startsWith('/rest/v1/')) {
    // Supabase maps the API key to a role; here the service key gets service_role, anything else anon.
    const token = headers.apikey === serviceJwt || headers.authorization === `Bearer ${serviceJwt}` ? serviceJwt : anonJwt;
    const forwarded = await fetch(`${restUrl}${url.pathname.slice('/rest/v1'.length)}${url.search}`, {
      method: request.method,
      headers: {
        'content-type': headers['content-type'] ?? 'application/json',
        prefer: headers.prefer ?? '',
        authorization: `Bearer ${token}`,
      },
      body: ['GET', 'HEAD'].includes(request.method) ? undefined : body,
    });
    response.writeHead(forwarded.status, { ...cors, 'content-type': forwarded.headers.get('content-type') ?? 'application/json' });
    response.end(Buffer.from(await forwarded.arrayBuffer()));
    return;
  }

  response.writeHead(404, cors);
  response.end('not found');
}
