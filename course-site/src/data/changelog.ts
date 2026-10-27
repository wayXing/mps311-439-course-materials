import changelogJson from './changelog.json';

export type ChangeType = 'feedback' | 'added' | 'changed' | 'fixed';

export interface ChangelogRelease {
  version: string;
  date: string;
  dateLabel?: string;
  title: string;
  items: { type: ChangeType; text: string }[];
}

export interface ChangelogData {
  commitment: string[];
  releases: ChangelogRelease[];
}

// Newest release first; the site badge shows releases[0].
export const changelog = changelogJson as ChangelogData;
export const currentVersion = changelog.releases[0].version;
