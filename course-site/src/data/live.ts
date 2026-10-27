import liveJson from './live.json';
import { course } from './course';

export interface LivePoll {
  key: string;
  kind?: 'choice' | 'text';
  slide: string;
  prompt: string;
  options?: string[];
  showLive?: boolean;
}

export interface LiveLesson {
  sections: string[];
  polls: LivePoll[];
}

const configured = liveJson.lessons as Record<string, LiveLesson>;
const workshops = liveJson.workshops as Record<string, LiveLesson & { label: string; title: string }>;

// Every teaching lesson can be started live; lessons without a configuration have no polls yet.
export const liveLessons = course.lessons
  .filter((lesson) => lesson.type === 'teaching')
  .map((lesson) => ({
    slug: lesson.slug,
    label: lesson.label,
    title: lesson.title,
    sections: configured[lesson.slug]?.sections ?? [],
    polls: configured[lesson.slug]?.polls ?? [],
    workshop: false,
  }));

// The optional assessment workshops sit in the same menu. The live system stores them as a
// 'lecture' session (it knows only lecture and lab); the pages call them workshops.
liveLessons.push(...Object.entries(workshops).map(([slug, workshop]) => ({
  slug,
  label: workshop.label,
  title: workshop.title,
  sections: workshop.sections,
  polls: workshop.polls,
  workshop: true,
})));

export const labSections = liveJson.labSections as string[];

// The end-of-class "least clear" vote lists a lesson's parts plus "Everything was clear": at most 8 options.
for (const [slug, lesson] of Object.entries({ ...configured, ...workshops })) {
  if (lesson.sections.length > 7) throw new Error(`live.json: ${slug} has more than 7 sections.`);
}
if (labSections.length > 7) throw new Error('live.json: labSections has more than 7 entries.');

export const academicYearKey = course.academicYear.replace(/[^0-9]+/g, '-').replace(/^-|-$/g, '');

// Supabase project. A local test stack overrides it with PUBLIC_SUPABASE_URL (see scripts/local-supabase.mjs).
export const supabaseUrl = import.meta.env.PUBLIC_SUPABASE_URL || 'https://whxpzfruhtshrwiksezg.supabase.co';
if (import.meta.env.PROD && /localhost|127\.0\.0\.1/.test(supabaseUrl)) {
  throw new Error('PUBLIC_SUPABASE_URL points at a local test stack; unset it for a production build.');
}
export const publishableKey = 'sb_publishable_GEKCtohy16BxHMc8_fji6w_TbcrlbPd';
