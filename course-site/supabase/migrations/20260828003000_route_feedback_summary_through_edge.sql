revoke execute on function public.get_course_feedback_summary(text)
  from public, anon, authenticated;
grant execute on function public.get_course_feedback_summary(text)
  to service_role;

comment on function public.get_course_feedback_summary(text) is
  'Service-only aggregate ratings reader. Written suggestions are never returned.';

notify pgrst, 'reload schema';
