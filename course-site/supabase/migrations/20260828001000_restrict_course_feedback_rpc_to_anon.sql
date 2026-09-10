revoke execute on function public.submit_course_feedback(text, integer, integer, integer, text)
  from authenticated;
revoke execute on function public.get_course_feedback_summary(text)
  from authenticated;

comment on table private.course_feedback is
  'Anonymous course feedback. Raw rows are intentionally kept outside the Data API.';
comment on function public.submit_course_feedback(text, integer, integer, integer, text) is
  'Intentional anonymous RPC. Validates a bounded payload and writes only to private.course_feedback.';
comment on function public.get_course_feedback_summary(text) is
  'Intentional anonymous RPC. Returns aggregate ratings only and never returns written comments.';

notify pgrst, 'reload schema';
