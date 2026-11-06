import changelogJson from './changelog.json';

export type ChangeType = 'improvement' | 'new' | 'fix';

export interface ChangelogItem {
  type: ChangeType;
  text: string;
}

export interface ChangelogRelease {
  version: string;
  date: string;
  dateLabel?: string;
  title: string;
  items: ChangelogItem[];
}

export interface ChangelogData {
  commitment: string[];
  releases: ChangelogRelease[];
}

export const changelog = changelogJson as ChangelogData;
export const currentVersion = changelog.releases[0].version;
