export const teachingLessons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

// A lesson listed here is being redesigned: until its new deck and worksheet replace them,
// the site keeps serving the UoS 2025 version from the lesson's archive.
const servedFromArchive = {};

// Lessons 1 to 3 use Quarto decks in lecture/slides/; Lessons 4 to 10 still use Slidev (lecture/slide.md).
const quartoDecks = { 1: 'slide', 2: 'slide_v2', 3: 'slide' };
// Lessons whose lab worksheet is a Quarto .qmd file (the others are .md).
const qmdWorksheets = new Set([2, 3]);

// Source and build paths of the slide deck and lab worksheet that the site currently serves.
export function lessonPaths(lesson) {
  const deck = quartoDecks[lesson];
  const quarto = deck !== undefined;
  const base = servedFromArchive[lesson] ?? `lessons/lesson${lesson}`;
  return {
    quarto,
    slideSource: quarto ? `${base}/lecture/slides/${deck}.qmd` : `${base}/lecture/slide.md`,
    slideBuild: quarto ? `${base}/lecture/slides/${deck}.html` : `${base}/lecture/dist/index.html`,
    labSource: `${base}/lab/lab_worksheet.${qmdWorksheets.has(lesson) ? 'qmd' : 'md'}`,
    labBuild: `${base}/lab/lab_worksheet.html`,
    solution: `${base}/lab/lab_solution.ipynb`,
  };
}
