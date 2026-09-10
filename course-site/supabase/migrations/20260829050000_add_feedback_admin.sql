create table public.course_academic_years (
  year_key text primary key,
  label text not null,
  created_at timestamptz not null default now(),

  constraint course_academic_years_key_format
    check (year_key ~ '^[0-9]{4}-[0-9]{2}$'),
  constraint course_academic_years_label_format
    check (label ~ '^[0-9]{4}/[0-9]{2}$')
);

alter table public.course_academic_years enable row level security;

revoke all on table public.course_academic_years from public, anon, authenticated;
grant select on table public.course_academic_years to anon, authenticated;

create policy "Academic years are publicly selectable"
  on public.course_academic_years
  for select
  to anon, authenticated
  using (true);

insert into public.course_academic_years (year_key, label)
values
  ('2026-27', '2026/27'),
  ('2025-26', '2025/26')
on conflict (year_key) do nothing;

create or replace function public.admin_course_overview(
  p_year_key text default null
)
returns jsonb
language sql
security definer
set search_path = ''
as $$
  select jsonb_build_object(
    'years', coalesce((
      select jsonb_agg(
        jsonb_build_object(
          'key', years.year_key,
          'label', years.label,
          'feedbackCount', (
            select count(*)::integer
            from private.course_feedback feedback
            where feedback.context_key like years.year_key || '-%'
          ),
          'questionCount', (
            select count(*)::integer
            from public.course_questions questions
            where questions.context_key like years.year_key || '-%'
          )
        )
        order by years.year_key desc
      )
      from public.course_academic_years years
    ), '[]'::jsonb),
    'feedback', coalesce((
      select jsonb_agg(to_jsonb(feedback_rows))
      from (
        select
          feedback.context_key as "contextKey",
          feedback.difficulty,
          feedback.pace,
          feedback.content_amount as "contentAmount",
          feedback.comment,
          feedback.created_at as "createdAt"
        from private.course_feedback feedback
        where p_year_key is null or feedback.context_key like lower(btrim(p_year_key)) || '-%'
        order by feedback.created_at desc
        limit 1000
      ) feedback_rows
    ), '[]'::jsonb),
    'questions', coalesce((
      select jsonb_agg(to_jsonb(question_rows))
      from (
        select
          questions.context_key as "contextKey",
          questions.display_name as "displayName",
          questions.question,
          questions.is_visible as "isVisible",
          questions.created_at as "createdAt"
        from public.course_questions questions
        where p_year_key is null or questions.context_key like lower(btrim(p_year_key)) || '-%'
        order by questions.created_at desc
        limit 1000
      ) question_rows
    ), '[]'::jsonb)
  );
$$;

create or replace function public.admin_create_course_year(
  p_year_key text
)
returns void
language plpgsql
security definer
set search_path = ''
as $$
declare
  normalized_key text := lower(btrim(p_year_key));
  start_year integer;
  end_year integer;
begin
  if normalized_key !~ '^[0-9]{4}-[0-9]{2}$' then
    raise exception using errcode = '22023', message = 'Use an academic year such as 2027-28.';
  end if;

  start_year := substring(normalized_key from 1 for 4)::integer;
  end_year := substring(normalized_key from 6 for 2)::integer;
  if end_year <> ((start_year + 1) % 100) then
    raise exception using errcode = '22023', message = 'The academic year must cover consecutive years.';
  end if;

  insert into public.course_academic_years (year_key, label)
  values (normalized_key, start_year::text || '/' || lpad(end_year::text, 2, '0'));
end;
$$;

create or replace function public.admin_delete_course_year(
  p_year_key text
)
returns void
language plpgsql
security definer
set search_path = ''
as $$
begin
  delete from public.course_academic_years
  where year_key = lower(btrim(p_year_key));
end;
$$;

revoke all on function public.admin_course_overview(text) from public, anon, authenticated;
revoke all on function public.admin_create_course_year(text) from public, anon, authenticated;
revoke all on function public.admin_delete_course_year(text) from public, anon, authenticated;

grant execute on function public.admin_course_overview(text) to service_role;
grant execute on function public.admin_create_course_year(text) to service_role;
grant execute on function public.admin_delete_course_year(text) to service_role;

comment on function public.admin_course_overview(text) is
  'Service-only instructor reader for raw anonymous feedback and class questions.';
comment on function public.admin_create_course_year(text) is
  'Service-only instructor action that exposes an academic year in the feedback selector.';
comment on function public.admin_delete_course_year(text) is
  'Service-only instructor action that removes an academic year from the feedback selector without deleting records.';

notify pgrst, 'reload schema';
