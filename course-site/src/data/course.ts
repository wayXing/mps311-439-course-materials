import courseJson from './course.json';

export type Accent = 'coral' | 'blue' | 'sage' | 'gold' | 'violet';
export type LessonType = 'preparation' | 'teaching';
export type ResourceKind = 'slides' | 'pdf' | 'notes' | 'notebook' | 'workbook' | 'worksheet' | 'solution' | 'guide' | 'data';

export interface CourseResource {
  label: string;
  description: string;
  kind: ResourceKind;
  group: string;
  source: string;
  target: string;
  href: string;
  directory?: boolean;
  downloadSource?: string;
  downloadTarget?: string;
  downloadHref?: string;
  render?: 'quarto';
}

export interface CourseLesson {
  slug: string;
  number: string;
  label: string;
  title: string;
  shortTitle: string;
  summary: string;
  accent: Accent;
  type: LessonType;
  topics: string[];
  note?: string;
  resources: CourseResource[];
}

export interface CourseData {
  code: string;
  title: string;
  academicYear: string;
  instructor: string;
  description: string;
  colab?: {
    repository: string;
    branch?: string;
    notebookRoot?: string;
  };
  lessons: CourseLesson[];
}

export const course = courseJson as CourseData;
