declare const Deno: {
  env: { get(name: string): string | undefined };
  serve(handler: (request: Request) => Response | Promise<Response>): void;
};

const allowedOrigins = new Set([
  'https://ml.wxing.me',
  'https://mps311-439-machine-learning.vercel.app',
  'https://mps311-439-machine-learning.weivern.chatgpt.site',
  'http://localhost:4321',
  'http://127.0.0.1:4321',
]);

const defaultOrigin = 'https://ml.wxing.me';
const adminPasswordHash = '0cf14b319e3ea5f83b48509ff27b06e71dc010eb330c327a67c0497c7186e53e';
const adminTokenLifetimeSeconds = 12 * 60 * 60;

const encoder = new TextEncoder();
const decoder = new TextDecoder();

const responseHeaders = (origin: string | null) => ({
  'Access-Control-Allow-Origin': origin && allowedOrigins.has(origin)
    ? origin
    : defaultOrigin,
  'Access-Control-Allow-Headers': 'apikey, content-type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
  'Content-Type': 'application/json; charset=utf-8',
  'Vary': 'Origin',
});

const jsonResponse = (
  body: Record<string, unknown>,
  status: number,
  origin: string | null,
) => new Response(JSON.stringify(body), {
  status,
  headers: responseHeaders(origin),
});

const keyedHash = async (secret: string, value: string) => {
  const key = await crypto.subtle.importKey(
    'raw',
    encoder.encode(secret),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign'],
  );
  const signature = await crypto.subtle.sign('HMAC', key, encoder.encode(value));
  return Array.from(new Uint8Array(signature))
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('');
};

const sha256 = async (value: string) => {
  const digest = await crypto.subtle.digest('SHA-256', encoder.encode(value));
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('');
};

const constantTimeEquals = (left: string, right: string) => {
  let difference = left.length ^ right.length;
  const limit = Math.max(left.length, right.length);
  for (let index = 0; index < limit; index += 1) {
    difference |= (left.charCodeAt(index) || 0) ^ (right.charCodeAt(index) || 0);
  }
  return difference === 0;
};

const base64UrlEncode = (value: string) => {
  const bytes = encoder.encode(value);
  let binary = '';
  bytes.forEach((byte) => { binary += String.fromCharCode(byte); });
  return btoa(binary).replaceAll('+', '-').replaceAll('/', '_').replace(/=+$/, '');
};

const base64UrlDecode = (value: string) => {
  const padded = value.replaceAll('-', '+').replaceAll('_', '/')
    + '='.repeat((4 - (value.length % 4)) % 4);
  const binary = atob(padded);
  return decoder.decode(Uint8Array.from(binary, (character) => character.charCodeAt(0)));
};

const issueAdminToken = async (secret: string) => {
  const payload = JSON.stringify({ role: 'course-admin', exp: Math.floor(Date.now() / 1000) + adminTokenLifetimeSeconds });
  return `${base64UrlEncode(payload)}.${await keyedHash(secret, payload)}`;
};

const hasValidAdminToken = async (value: unknown, secret: string) => {
  if (typeof value !== 'string') return false;
  const [encodedPayload, signature, extra] = value.split('.');
  if (!encodedPayload || !signature || extra || !/^[a-f0-9]{64}$/.test(signature)) return false;
  try {
    const payload = base64UrlDecode(encodedPayload);
    const expectedSignature = await keyedHash(secret, payload);
    if (!constantTimeEquals(signature, expectedSignature)) return false;
    const parsed = JSON.parse(payload);
    return parsed?.role === 'course-admin'
      && Number.isInteger(parsed?.exp)
      && parsed.exp > Math.floor(Date.now() / 1000);
  } catch {
    return false;
  }
};

const clientIp = (request: Request) => {
  const forwarded = request.headers.get('x-forwarded-for');
  if (forwarded) return forwarded.split(',')[0].trim();
  return request.headers.get('cf-connecting-ip')
    ?? request.headers.get('x-real-ip')
    ?? 'unknown';
};

const callRpc = async (
  projectUrl: string,
  serviceRoleKey: string,
  functionName: string,
  body: Record<string, unknown>,
) => {
  const response = await fetch(`${projectUrl}/rest/v1/rpc/${functionName}`, {
    method: 'POST',
    headers: {
      apikey: serviceRoleKey,
      Authorization: `Bearer ${serviceRoleKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    let message = 'submission_failed';
    try {
      const error = await response.json();
      if (typeof error?.message === 'string') message = error.message;
    } catch {}
    throw new Error(message);
  }

  const text = await response.text();
  return text ? JSON.parse(text) : null;
};

Deno.serve(async (request: Request) => {
  const origin = request.headers.get('origin');

  if (request.method === 'OPTIONS') {
    return new Response(null, { status: 204, headers: responseHeaders(origin) });
  }

  if (request.method !== 'POST') {
    return jsonResponse({ error: 'Method not allowed.' }, 405, origin);
  }

  if (origin && !allowedOrigins.has(origin)) {
    return jsonResponse({ error: 'This site is not allowed to submit.' }, 403, origin);
  }

  const contentLength = Number(request.headers.get('content-length') ?? 0);
  if (contentLength > 12_000) {
    return jsonResponse({ error: 'Submission is too large.' }, 413, origin);
  }

  const projectUrl = Deno.env.get('SUPABASE_URL');
  const serviceRoleKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY');
  if (!projectUrl || !serviceRoleKey) {
    return jsonResponse({ error: 'The feedback service is unavailable.' }, 503, origin);
  }

  let payload: Record<string, unknown>;
  try {
    const rawBody = await request.text();
    if (rawBody.length > 12_000) {
      return jsonResponse({ error: 'Submission is too large.' }, 413, origin);
    }
    payload = JSON.parse(rawBody);
  } catch {
    return jsonResponse({ error: 'Invalid submission.' }, 400, origin);
  }

  // Simple honeypot: real users never see or fill this field.
  if (typeof payload.website === 'string' && payload.website.trim()) {
    return jsonResponse({ ok: true }, 200, origin);
  }

  const action = payload.action;

  if (action === 'admin_login') {
    const password = typeof payload.password === 'string' ? payload.password : '';
    if (!constantTimeEquals(await sha256(password), adminPasswordHash)) {
      return jsonResponse({ error: 'Incorrect password.' }, 401, origin);
    }
    return jsonResponse({ ok: true, token: await issueAdminToken(serviceRoleKey) }, 200, origin);
  }

  if (
    action === 'admin_overview'
    || action === 'admin_create_year'
    || action === 'admin_delete_year'
    || action === 'admin_set_year_visibility'
  ) {
    if (!await hasValidAdminToken(payload.admin_token, serviceRoleKey)) {
      return jsonResponse({ error: 'Your admin session has expired. Please sign in again.' }, 401, origin);
    }

    const yearKey = typeof payload.year_key === 'string' ? payload.year_key : null;
    try {
      if (action === 'admin_overview') {
        const overview = await callRpc(projectUrl, serviceRoleKey, 'admin_course_overview', {
          p_year_key: yearKey,
        });
        return jsonResponse({ ok: true, overview }, 200, origin);
      }

      if (!yearKey) {
        return jsonResponse({ error: 'An academic year is required.' }, 400, origin);
      }

      if (action === 'admin_set_year_visibility') {
        if (typeof payload.is_visible !== 'boolean') {
          return jsonResponse({ error: 'A visibility setting is required.' }, 400, origin);
        }
        await callRpc(projectUrl, serviceRoleKey, 'admin_set_course_year_visibility', {
          p_year_key: yearKey,
          p_is_visible: payload.is_visible,
        });
      } else {
        await callRpc(
          projectUrl,
          serviceRoleKey,
          action === 'admin_create_year' ? 'admin_create_course_year' : 'admin_delete_course_year',
          { p_year_key: yearKey },
        );
      }

      const overview = await callRpc(projectUrl, serviceRoleKey, 'admin_course_overview', {
        p_year_key: null,
      });
      return jsonResponse({ ok: true, overview }, 200, origin);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'admin_request_failed';
      if (message.includes('academic-year name') || message.includes('already exists')) {
        return jsonResponse({ error: message }, 400, origin);
      }
      if (message.includes('duplicate key')) {
        return jsonResponse({ error: 'That academic year already exists.' }, 409, origin);
      }
      return jsonResponse({ error: 'Could not update the academic years.' }, 500, origin);
    }
  }

  const contextKey = typeof payload.context_key === 'string' ? payload.context_key : '';

  if (!/^[a-z0-9][a-z0-9-]{2,79}$/.test(contextKey)) {
    return jsonResponse({ error: 'Invalid class session.' }, 400, origin);
  }

  if (action === 'summary') {
    try {
      const summary = await callRpc(
        projectUrl,
        serviceRoleKey,
        'get_course_feedback_summary',
        { p_context_key: contextKey },
      );
      return jsonResponse({ ok: true, summary }, 200, origin);
    } catch {
      return jsonResponse({ error: 'Could not load the class snapshot.' }, 500, origin);
    }
  }

  const deviceId = typeof payload.device_id === 'string' ? payload.device_id : '';
  if (!/^[a-zA-Z0-9-]{20,100}$/.test(deviceId)) {
    return jsonResponse({ error: 'Invalid browser token.' }, 400, origin);
  }

  const deviceKeyHash = await keyedHash(
    serviceRoleKey,
    `${String(action)}:${contextKey}:${deviceId}`,
  );
  const ipKeyHash = await keyedHash(serviceRoleKey, `ip:${clientIp(request)}`);

  try {
    if (action === 'feedback') {
      await callRpc(projectUrl, serviceRoleKey, 'submit_course_feedback_v2', {
        p_context_key: contextKey,
        p_difficulty: payload.difficulty,
        p_pace: payload.pace,
        p_content_amount: payload.content_amount,
        p_comment: payload.comment,
        p_device_key_hash: deviceKeyHash,
        p_ip_key_hash: ipKeyHash,
      });
      return jsonResponse({ ok: true, mode: 'upserted' }, 200, origin);
    }

    if (action === 'question') {
      const id = await callRpc(projectUrl, serviceRoleKey, 'submit_course_question_v1', {
        p_context_key: contextKey,
        p_display_name: payload.display_name,
        p_question: payload.question,
        p_device_key_hash: deviceKeyHash,
        p_ip_key_hash: ipKeyHash,
      });
      return jsonResponse({ ok: true, id }, 200, origin);
    }

    return jsonResponse({ error: 'Unknown submission type.' }, 400, origin);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'submission_failed';
    if (message.includes('rate_limit_device')) {
      return jsonResponse({ error: 'Please wait a little before submitting again.' }, 429, origin);
    }
    if (message.includes('rate_limit_ip')) {
      return jsonResponse({ error: 'This network is receiving many submissions. Please try again shortly.' }, 429, origin);
    }
    if (message.includes('must be') || message.includes('Invalid')) {
      return jsonResponse({ error: message }, 400, origin);
    }
    return jsonResponse({ error: 'Could not save this submission. Please try again.' }, 500, origin);
  }
});
