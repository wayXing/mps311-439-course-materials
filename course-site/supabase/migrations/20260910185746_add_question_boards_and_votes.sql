create table public.course_question_boards (
  context_key text primary key,
  status text not null default 'archived',
  opened_at timestamptz,
  archived_at timestamptz,
  created_at timestamptz not null default now(),
  constraint course_question_boards_context_key_format check (context_key ~ '^[a-z0-9][a-z0-9-]{2,79}$'),
  constraint course_question_boards_status check (status in ('live', 'archived')),
  constraint course_question_boards_lifecycle check (
    (status = 'live' and opened_at is not null and archived_at is null)
    or (status = 'archived' and archived_at is not null)
  )
);

insert into public.course_question_boards (context_key, status, opened_at, archived_at)
select distinct context_key, 'archived', min(created_at), now()
from public.course_questions
group by context_key
on conflict (context_key) do nothing;

alter table public.course_question_boards enable row level security;
revoke all on table public.course_question_boards from public, anon, authenticated;
grant select on table public.course_question_boards to anon, authenticated;
create policy "Question boards are publicly readable" on public.course_question_boards for select to anon, authenticated using (true);

alter table public.course_questions
  add column vote_count integer not null default 0,
  add column is_answered boolean not null default false,
  add column answered_at timestamptz,
  add constraint course_questions_vote_count_nonnegative check (vote_count >= 0);
create index course_questions_context_ranking_idx on public.course_questions (context_key, is_answered, vote_count desc, created_at asc) where is_visible;

create table private.course_question_votes (
  question_id bigint not null references public.course_questions(id) on delete cascade,
  voter_key_hash text not null,
  created_at timestamptz not null default now(),
  primary key (question_id, voter_key_hash),
  constraint course_question_votes_hash_format check (voter_key_hash ~ '^[a-f0-9]{64}$')
);
alter table private.course_question_votes enable row level security;
alter table private.course_question_votes force row level security;

alter table private.course_submission_events drop constraint course_submission_events_action;
alter table private.course_submission_events add constraint course_submission_events_action check (action in ('feedback', 'question', 'vote'));

create or replace function public.open_course_question_board(p_context_key text)
returns void language plpgsql security definer set search_path = '' as $$
declare normalized_context text := lower(btrim(p_context_key));
begin
  if normalized_context is null or normalized_context !~ '^[a-z0-9][a-z0-9-]{2,79}$' then
    raise exception using errcode = '22023', message = 'Invalid class session.';
  end if;
  update public.course_question_boards set status = 'archived', archived_at = now()
  where status = 'live' and context_key <> normalized_context;
  insert into public.course_question_boards (context_key, status, opened_at, archived_at)
  values (normalized_context, 'live', now(), null)
  on conflict (context_key) do update set status = 'live', opened_at = now(), archived_at = null;
end;
$$;

create or replace function public.archive_course_question_board(p_context_key text)
returns void language plpgsql security definer set search_path = '' as $$
begin
  update public.course_question_boards set status = 'archived', archived_at = now()
  where context_key = lower(btrim(p_context_key)) and status = 'live';
end;
$$;

create or replace function public.submit_course_question_v2(
  p_context_key text, p_display_name text, p_question text, p_device_key_hash text, p_ip_key_hash text
) returns bigint language plpgsql security definer set search_path = '' as $$
declare
  normalized_context text := lower(btrim(p_context_key));
  normalized_name text := coalesce(nullif(btrim(p_display_name), ''), 'Anonymous');
  normalized_question text := nullif(btrim(p_question), '');
  inserted_id bigint;
begin
  if normalized_context is null or normalized_context !~ '^[a-z0-9][a-z0-9-]{2,79}$' then raise exception using errcode = '22023', message = 'Invalid class session.'; end if;
  if not exists (select 1 from public.course_question_boards where context_key = normalized_context and status = 'live') then raise exception using errcode = 'P0001', message = 'board_closed'; end if;
  if char_length(normalized_name) > 40 then raise exception using errcode = '22023', message = 'Display name must be 40 characters or fewer.'; end if;
  if normalized_question is null or char_length(normalized_question) > 1000 then raise exception using errcode = '22023', message = 'Question must be between 1 and 1,000 characters.'; end if;
  if p_device_key_hash !~ '^[a-f0-9]{64}$' or p_ip_key_hash !~ '^[a-f0-9]{64}$' then raise exception using errcode = '22023', message = 'Invalid submission token.'; end if;
  if (select count(*) from private.course_submission_events where action = 'question' and context_key = normalized_context and device_key_hash = p_device_key_hash and created_at > now() - interval '10 minutes') >= 3 then raise exception using errcode = 'P0001', message = 'rate_limit_device'; end if;
  if (select count(*) from private.course_submission_events where action = 'question' and context_key = normalized_context and ip_key_hash = p_ip_key_hash and created_at > now() - interval '10 minutes') >= 60 then raise exception using errcode = 'P0001', message = 'rate_limit_ip'; end if;
  insert into private.course_submission_events (action, context_key, device_key_hash, ip_key_hash) values ('question', normalized_context, p_device_key_hash, p_ip_key_hash);
  insert into public.course_questions (context_key, display_name, question) values (normalized_context, normalized_name, normalized_question) returning id into inserted_id;
  return inserted_id;
end;
$$;

create or replace function public.toggle_course_question_vote_v1(
  p_question_id bigint, p_context_key text, p_voter_key_hash text, p_ip_key_hash text
) returns table (voted boolean, vote_count integer) language plpgsql security definer set search_path = '' as $$
declare normalized_context text := lower(btrim(p_context_key)); current_vote_count integer;
begin
  if p_voter_key_hash !~ '^[a-f0-9]{64}$' or p_ip_key_hash !~ '^[a-f0-9]{64}$' then raise exception using errcode = '22023', message = 'Invalid vote token.'; end if;
  if not exists (select 1 from public.course_question_boards where context_key = normalized_context and status = 'live') then raise exception using errcode = 'P0001', message = 'board_closed'; end if;
  if (select count(*) from private.course_submission_events where action = 'vote' and context_key = normalized_context and device_key_hash = p_voter_key_hash and created_at > now() - interval '1 minute') >= 30 then raise exception using errcode = 'P0001', message = 'rate_limit_device'; end if;
  if (select count(*) from private.course_submission_events where action = 'vote' and context_key = normalized_context and ip_key_hash = p_ip_key_hash and created_at > now() - interval '1 minute') >= 300 then raise exception using errcode = 'P0001', message = 'rate_limit_ip'; end if;
  select questions.vote_count into current_vote_count from public.course_questions questions where questions.id = p_question_id and questions.context_key = normalized_context and questions.is_visible for update;
  if not found then raise exception using errcode = '22023', message = 'Question is unavailable.'; end if;
  insert into private.course_submission_events (action, context_key, device_key_hash, ip_key_hash) values ('vote', normalized_context, p_voter_key_hash, p_ip_key_hash);
  if exists (select 1 from private.course_question_votes where question_id = p_question_id and voter_key_hash = p_voter_key_hash) then
    delete from private.course_question_votes where question_id = p_question_id and voter_key_hash = p_voter_key_hash;
    update public.course_questions questions set vote_count = greatest(questions.vote_count - 1, 0) where questions.id = p_question_id returning questions.vote_count into current_vote_count;
    return query select false, current_vote_count;
  else
    insert into private.course_question_votes (question_id, voter_key_hash) values (p_question_id, p_voter_key_hash);
    update public.course_questions questions set vote_count = questions.vote_count + 1 where questions.id = p_question_id returning questions.vote_count into current_vote_count;
    return query select true, current_vote_count;
  end if;
end;
$$;

create or replace function public.admin_set_course_question_answered(p_question_id bigint, p_is_answered boolean)
returns void language plpgsql security definer set search_path = '' as $$
begin update public.course_questions set is_answered = p_is_answered, answered_at = case when p_is_answered then now() else null end where id = p_question_id; end;
$$;
create or replace function public.admin_set_course_question_visibility(p_question_id bigint, p_is_visible boolean)
returns void language plpgsql security definer set search_path = '' as $$
begin update public.course_questions set is_visible = p_is_visible where id = p_question_id; end;
$$;

create or replace function public.admin_course_overview(p_year_key text default null)
returns jsonb language sql security definer set search_path = '' as $$
  select jsonb_build_object(
    'years', coalesce((select jsonb_agg(jsonb_build_object('key', years.year_key, 'label', years.label, 'isVisible', years.is_visible, 'feedbackCount', (select count(*)::integer from private.course_feedback feedback where feedback.context_key like years.year_key || '-%'), 'questionCount', (select count(*)::integer from public.course_questions questions where questions.context_key like years.year_key || '-%')) order by years.year_key desc) from public.course_academic_years years), '[]'::jsonb),
    'boards', coalesce((select jsonb_agg(to_jsonb(board_rows)) from (select boards.context_key as "contextKey", boards.status, boards.opened_at as "openedAt", boards.archived_at as "archivedAt", (select count(*)::integer from public.course_questions questions where questions.context_key = boards.context_key) as "questionCount" from public.course_question_boards boards where p_year_key is null or boards.context_key like lower(btrim(p_year_key)) || '-%' order by (boards.status = 'live') desc, coalesce(boards.opened_at, boards.archived_at, boards.created_at) desc) board_rows), '[]'::jsonb),
    'feedback', coalesce((select jsonb_agg(to_jsonb(feedback_rows)) from (select feedback.context_key as "contextKey", feedback.difficulty, feedback.pace, feedback.content_amount as "contentAmount", feedback.comment, feedback.created_at as "createdAt" from private.course_feedback feedback where p_year_key is null or feedback.context_key like lower(btrim(p_year_key)) || '-%' order by feedback.created_at desc limit 1000) feedback_rows), '[]'::jsonb),
    'questions', coalesce((select jsonb_agg(to_jsonb(question_rows)) from (select questions.id, questions.context_key as "contextKey", questions.display_name as "displayName", questions.question, questions.is_visible as "isVisible", questions.is_answered as "isAnswered", questions.vote_count as "voteCount", questions.created_at as "createdAt" from public.course_questions questions where p_year_key is null or questions.context_key like lower(btrim(p_year_key)) || '-%' order by questions.created_at desc limit 1000) question_rows), '[]'::jsonb)
  );
$$;

revoke all on function public.submit_course_question_v2(text, text, text, text, text) from public, anon, authenticated;
revoke all on function public.toggle_course_question_vote_v1(bigint, text, text, text) from public, anon, authenticated;
revoke all on function public.open_course_question_board(text) from public, anon, authenticated;
revoke all on function public.archive_course_question_board(text) from public, anon, authenticated;
revoke all on function public.admin_set_course_question_answered(bigint, boolean) from public, anon, authenticated;
revoke all on function public.admin_set_course_question_visibility(bigint, boolean) from public, anon, authenticated;
revoke all on function public.admin_course_overview(text) from public, anon, authenticated;
grant execute on function public.submit_course_question_v2(text, text, text, text, text) to service_role;
grant execute on function public.toggle_course_question_vote_v1(bigint, text, text, text) to service_role;
grant execute on function public.open_course_question_board(text) to service_role;
grant execute on function public.archive_course_question_board(text) to service_role;
grant execute on function public.admin_set_course_question_answered(bigint, boolean) to service_role;
grant execute on function public.admin_set_course_question_visibility(bigint, boolean) to service_role;
grant execute on function public.admin_course_overview(text) to service_role;
notify pgrst, 'reload schema';
