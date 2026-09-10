alter table private.course_feedback
  add column if not exists submission_key_hash text;

create unique index if not exists course_feedback_context_submission_key_idx
  on private.course_feedback (context_key, submission_key_hash)
  where submission_key_hash is not null;

create table private.course_submission_events (
  id bigint generated always as identity primary key,
  action text not null,
  context_key text not null,
  device_key_hash text not null,
  ip_key_hash text not null,
  created_at timestamptz not null default now(),

  constraint course_submission_events_action
    check (action in ('feedback', 'question')),
  constraint course_submission_events_context_key_format
    check (context_key ~ '^[a-z0-9][a-z0-9-]{2,79}$'),
  constraint course_submission_events_device_hash_format
    check (device_key_hash ~ '^[a-f0-9]{64}$'),
  constraint course_submission_events_ip_hash_format
    check (ip_key_hash ~ '^[a-f0-9]{64}$')
);

create index course_submission_events_device_window_idx
  on private.course_submission_events (
    action,
    context_key,
    device_key_hash,
    created_at desc
  );

create index course_submission_events_ip_window_idx
  on private.course_submission_events (
    action,
    context_key,
    ip_key_hash,
    created_at desc
  );

alter table private.course_submission_events enable row level security;
alter table private.course_submission_events force row level security;

create table public.course_questions (
  id bigint generated always as identity primary key,
  context_key text not null,
  display_name text not null,
  question text not null,
  is_visible boolean not null default true,
  created_at timestamptz not null default now(),

  constraint course_questions_context_key_format
    check (context_key ~ '^[a-z0-9][a-z0-9-]{2,79}$'),
  constraint course_questions_display_name_length
    check (char_length(display_name) between 1 and 40),
  constraint course_questions_question_length
    check (char_length(question) between 1 and 1000)
);

create index course_questions_context_created_idx
  on public.course_questions (context_key, created_at desc)
  where is_visible;

alter table public.course_questions enable row level security;

revoke all on table public.course_questions from public, anon, authenticated;
grant select on table public.course_questions to anon, authenticated;

create policy "Visible class questions are public"
  on public.course_questions
  for select
  to anon, authenticated
  using (is_visible);

revoke execute on function public.submit_course_feedback(
  text,
  integer,
  integer,
  integer,
  text
) from public, anon, authenticated;

create or replace function public.submit_course_feedback_v2(
  p_context_key text,
  p_difficulty integer,
  p_pace integer,
  p_content_amount integer,
  p_comment text,
  p_device_key_hash text,
  p_ip_key_hash text
)
returns void
language plpgsql
security definer
set search_path = ''
as $$
declare
  normalized_context text := lower(btrim(p_context_key));
  normalized_comment text := nullif(btrim(p_comment), '');
begin
  if normalized_context is null
    or normalized_context !~ '^[a-z0-9][a-z0-9-]{2,79}$' then
    raise exception using errcode = '22023', message = 'Invalid feedback session.';
  end if;

  if p_difficulty not between 1 and 10
    or p_pace not between 1 and 10
    or p_content_amount not between 1 and 10 then
    raise exception using errcode = '22023', message = 'Ratings must be between 1 and 10.';
  end if;

  if normalized_comment is not null
    and char_length(normalized_comment) > 2000 then
    raise exception using errcode = '22023', message = 'Suggestion must be 2,000 characters or fewer.';
  end if;

  if p_device_key_hash !~ '^[a-f0-9]{64}$'
    or p_ip_key_hash !~ '^[a-f0-9]{64}$' then
    raise exception using errcode = '22023', message = 'Invalid submission token.';
  end if;

  if (
    select count(*)
    from private.course_submission_events
    where action = 'feedback'
      and context_key = normalized_context
      and device_key_hash = p_device_key_hash
      and created_at > now() - interval '1 minute'
  ) >= 3 then
    raise exception using errcode = 'P0001', message = 'rate_limit_device';
  end if;

  if (
    select count(*)
    from private.course_submission_events
    where action = 'feedback'
      and context_key = normalized_context
      and ip_key_hash = p_ip_key_hash
      and created_at > now() - interval '1 minute'
  ) >= 40 then
    raise exception using errcode = 'P0001', message = 'rate_limit_ip';
  end if;

  insert into private.course_submission_events (
    action,
    context_key,
    device_key_hash,
    ip_key_hash
  ) values (
    'feedback',
    normalized_context,
    p_device_key_hash,
    p_ip_key_hash
  );

  insert into private.course_feedback (
    context_key,
    difficulty,
    pace,
    content_amount,
    comment,
    submission_key_hash
  ) values (
    normalized_context,
    p_difficulty::smallint,
    p_pace::smallint,
    p_content_amount::smallint,
    normalized_comment,
    p_device_key_hash
  )
  on conflict (context_key, submission_key_hash)
    where submission_key_hash is not null
  do update set
    difficulty = excluded.difficulty,
    pace = excluded.pace,
    content_amount = excluded.content_amount,
    comment = excluded.comment,
    created_at = now();
end;
$$;

create or replace function public.submit_course_question_v1(
  p_context_key text,
  p_display_name text,
  p_question text,
  p_device_key_hash text,
  p_ip_key_hash text
)
returns bigint
language plpgsql
security definer
set search_path = ''
as $$
declare
  normalized_context text := lower(btrim(p_context_key));
  normalized_name text := coalesce(nullif(btrim(p_display_name), ''), 'Anonymous');
  normalized_question text := nullif(btrim(p_question), '');
  inserted_id bigint;
begin
  if normalized_context is null
    or normalized_context !~ '^[a-z0-9][a-z0-9-]{2,79}$' then
    raise exception using errcode = '22023', message = 'Invalid class session.';
  end if;

  if char_length(normalized_name) > 40 then
    raise exception using errcode = '22023', message = 'Display name must be 40 characters or fewer.';
  end if;

  if normalized_question is null or char_length(normalized_question) > 1000 then
    raise exception using errcode = '22023', message = 'Question must be between 1 and 1,000 characters.';
  end if;

  if p_device_key_hash !~ '^[a-f0-9]{64}$'
    or p_ip_key_hash !~ '^[a-f0-9]{64}$' then
    raise exception using errcode = '22023', message = 'Invalid submission token.';
  end if;

  if (
    select count(*)
    from private.course_submission_events
    where action = 'question'
      and context_key = normalized_context
      and device_key_hash = p_device_key_hash
      and created_at > now() - interval '10 minutes'
  ) >= 3 then
    raise exception using errcode = 'P0001', message = 'rate_limit_device';
  end if;

  if (
    select count(*)
    from private.course_submission_events
    where action = 'question'
      and context_key = normalized_context
      and ip_key_hash = p_ip_key_hash
      and created_at > now() - interval '10 minutes'
  ) >= 60 then
    raise exception using errcode = 'P0001', message = 'rate_limit_ip';
  end if;

  insert into private.course_submission_events (
    action,
    context_key,
    device_key_hash,
    ip_key_hash
  ) values (
    'question',
    normalized_context,
    p_device_key_hash,
    p_ip_key_hash
  );

  insert into public.course_questions (
    context_key,
    display_name,
    question
  ) values (
    normalized_context,
    normalized_name,
    normalized_question
  )
  returning id into inserted_id;

  return inserted_id;
end;
$$;

revoke all on function public.submit_course_feedback_v2(
  text,
  integer,
  integer,
  integer,
  text,
  text,
  text
) from public, anon, authenticated;
grant execute on function public.submit_course_feedback_v2(
  text,
  integer,
  integer,
  integer,
  text,
  text,
  text
) to service_role;

revoke all on function public.submit_course_question_v1(
  text,
  text,
  text,
  text,
  text
) from public, anon, authenticated;
grant execute on function public.submit_course_question_v1(
  text,
  text,
  text,
  text,
  text
) to service_role;

comment on table public.course_questions is
  'Public, instructor-facing class questions. Display names are user supplied and unverified.';
comment on table private.course_submission_events is
  'Short rate-limit event history. Contains keyed hashes only; raw IP addresses are never stored.';
comment on function public.submit_course_feedback_v2(
  text,
  integer,
  integer,
  integer,
  text,
  text,
  text
) is 'Service-only anonymous feedback writer with device and network rate limits.';
comment on function public.submit_course_question_v1(
  text,
  text,
  text,
  text,
  text
) is 'Service-only public class-question writer with device and network rate limits.';

notify pgrst, 'reload schema';
