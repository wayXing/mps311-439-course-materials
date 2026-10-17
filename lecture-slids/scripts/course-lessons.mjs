export const teachingLessons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

// A lesson listed here is being redesigned: until its new deck and worksheet replace them,
// the site keeps serving the UoS 2025 version from the lesson's archive.
const servedFromArchive = {};

// All ten lessons use Quarto decks in lecture/slides/.
const quartoDecks = { 1: 'slide', 2: 'slide_v2', 3: 'slide', 4: 'slide', 5: 'slide', 6: 'slide', 7: 'slide', 8: 'slide', 9: 'slide', 10: 'slide' };
// Lessons whose lab worksheet is a Quarto .qmd file (the others are .md).
const qmdWorksheets = new Set([2, 3, 4, 5, 6, 7, 8, 9, 10]);

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
