alter table public.course_academic_years
  drop constraint course_academic_years_key_format,
  drop constraint course_academic_years_label_format;

alter table public.course_academic_years
  add constraint course_academic_years_key_format
    check (year_key ~ '^[a-z0-9][a-z0-9-]{2,79}$'),
  add constraint course_academic_years_label_length
    check (char_length(btrim(label)) between 2 and 80);

create or replace function public.admin_create_course_year(
  p_year_key text
)
returns void
language plpgsql
security definer
set search_path = ''
as $$
declare
  normalized_label text := btrim(p_year_key);
  generated_key text;
  key_suffix text;
begin
  if normalized_label is null or char_length(normalized_label) not between 2 and 80 then
    raise exception using errcode = '22023', message = 'The academic-year name must be between 2 and 80 characters.';
  end if;

  if exists (
    select 1
    from public.course_academic_years years
    where lower(years.label) = lower(normalized_label)
  ) then
    raise exception using errcode = '23505', message = 'That academic-year name already exists.';
  end if;

  generated_key := trim(both '-' from regexp_replace(lower(normalized_label), '[^a-z0-9]+', '-', 'g'));
  if char_length(generated_key) < 3 then
    generated_key := 'year-' || substring(md5(normalized_label) from 1 for 12);
  end if;

  if char_length(generated_key) > 67 then
    key_suffix := substring(md5(normalized_label) from 1 for 12);
    generated_key := left(generated_key, 67) || '-' || key_suffix;
  end if;

  if exists (select 1 from public.course_academic_years years where years.year_key = generated_key) then
    generated_key := left(generated_key, 67) || '-' || substring(md5(normalized_label) from 1 for 12);
  end if;

  insert into public.course_academic_years (year_key, label)
  values (generated_key, normalized_label);
end;
$$;

comment on function public.admin_create_course_year(text) is
  'Service-only instructor action that creates a freely named student-facing academic-year option.';
