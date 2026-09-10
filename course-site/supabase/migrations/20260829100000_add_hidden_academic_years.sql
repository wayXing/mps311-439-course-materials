alter table public.course_academic_years
  add column is_visible boolean not null default true;

alter policy "Academic years are publicly selectable"
  on public.course_academic_years
  using (is_visible);

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
          'isVisible', years.is_visible,
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

create or replace function public.admin_set_course_year_visibility(
  p_year_key text,
  p_is_visible boolean
)
returns void
language plpgsql
security definer
set search_path = ''
as $$
begin
  update public.course_academic_years
  set is_visible = p_is_visible
  where year_key = lower(btrim(p_year_key));
end;
$$;

revoke all on function public.admin_set_course_year_visibility(text, boolean) from public, anon, authenticated;
grant execute on function public.admin_set_course_year_visibility(text, boolean) to service_role;

comment on function public.admin_set_course_year_visibility(text, boolean) is
  'Service-only instructor action that hides or shows a year in the public feedback selector.';
