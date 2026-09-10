import timetableJson from './timetable.json';

export interface TimetableEvent {
  lesson: number | null;
  type: string;
  title: string;
  theme: string;
  coreOutcomes: string[];
  advancedOutcomes: string[];
  assessment: string;
  lectureDate: string;
  labDate: string;
}

export interface TimetableData {
  source: string;
  academicYear: string;
  basis: 'lesson';
  events: TimetableEvent[];
}

export const timetable = timetableJson as TimetableData;
