create schema if not exists private;

revoke all on schema private from public;
revoke all on schema private from anon, authenticated;

create table private.course_feedback (
  id bigint generated always as identity primary key,
  context_key text not null,
  difficulty smallint not null,
  pace smallint not null,
  content_amount smallint not null,
  comment text,
  created_at timestamptz not null default now(),

  constraint course_feedback_context_key_format
    check (context_key ~ '^[a-z0-9][a-z0-9-]{2,79}$'),
  constraint course_feedback_difficulty_range
    check (difficulty between 1 and 10),
  constraint course_feedback_pace_range
    check (pace between 1 and 10),
  constraint course_feedback_content_amount_range
    check (content_amount between 1 and 10),
  constraint course_feedback_comment_length
    check (comment is null or char_length(comment) <= 2000)
);

create index course_feedback_context_created_idx
  on private.course_feedback (context_key, created_at desc);

alter table private.course_feedback enable row level security;
alter table private.course_feedback force row level security;

create or replace function public.submit_course_feedback(
  p_context_key text,
  p_difficulty integer,
  p_pace integer,
  p_content_amount integer,
  p_comment text default null
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
    raise exception using
      errcode = '22023',
      message = 'Invalid feedback session.';
  end if;

  if p_difficulty not between 1 and 10
    or p_pace not between 1 and 10
    or p_content_amount not between 1 and 10 then
    raise exception using
      errcode = '22023',
      message = 'Ratings must be between 1 and 10.';
  end if;

  if normalized_comment is not null
    and char_length(normalized_comment) > 2000 then
    raise exception using
      errcode = '22023',
      message = 'Comment must be 2,000 characters or fewer.';
  end if;

  insert into private.course_feedback (
    context_key,
    difficulty,
    pace,
    content_amount,
    comment
  )
  values (
    normalized_context,
    p_difficulty::smallint,
    p_pace::smallint,
    p_content_amount::smallint,
    normalized_comment
  );
end;
$$;

create or replace function public.get_course_feedback_summary(
  p_context_key text
)
returns jsonb
language sql
stable
security definer
set search_path = ''
as $$
  with filtered as (
    select difficulty, pace, content_amount
    from private.course_feedback
    where context_key = lower(btrim(p_context_key))
  ),
  summary as (
    select
      count(*)::integer as response_count,
      round(avg(difficulty)::numeric, 2) as difficulty_average,
      round(avg(pace)::numeric, 2) as pace_average,
      round(avg(content_amount)::numeric, 2) as content_amount_average
    from filtered
  ),
  difficulty_counts as (
    select scores.score, count(filtered.difficulty)::integer as response_count
    from generate_series(1, 10) as scores(score)
    left join filtered on filtered.difficulty = scores.score
    group by scores.score
  ),
  pace_counts as (
    select scores.score, count(filtered.pace)::integer as response_count
    from generate_series(1, 10) as scores(score)
    left join filtered on filtered.pace = scores.score
    group by scores.score
  ),
  content_counts as (
    select scores.score, count(filtered.content_amount)::integer as response_count
    from generate_series(1, 10) as scores(score)
    left join filtered on filtered.content_amount = scores.score
    group by scores.score
  )
  select jsonb_build_object(
    'contextKey', lower(btrim(p_context_key)),
    'responseCount', summary.response_count,
    'difficulty', jsonb_build_object(
      'average', summary.difficulty_average,
      'distribution', (
        select jsonb_agg(response_count order by score)
        from difficulty_counts
      )
    ),
    'pace', jsonb_build_object(
      'average', summary.pace_average,
      'distribution', (
        select jsonb_agg(response_count order by score)
        from pace_counts
      )
    ),
    'contentAmount', jsonb_build_object(
      'average', summary.content_amount_average,
      'distribution', (
        select jsonb_agg(response_count order by score)
        from content_counts
      )
    )
  )
  from summary;
$$;

revoke all on function public.submit_course_feedback(text, integer, integer, integer, text)
  from public;
revoke all on function public.get_course_feedback_summary(text)
  from public;

grant usage on schema public to anon, authenticated;
grant execute on function public.submit_course_feedback(text, integer, integer, integer, text)
  to anon, authenticated;
grant execute on function public.get_course_feedback_summary(text)
  to anon, authenticated;

notify pgrst, 'reload schema';
