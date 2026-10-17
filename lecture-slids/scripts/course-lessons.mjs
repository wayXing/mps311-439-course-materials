export const teachingLessons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

// A lesson listed here is being redesigned: until its new deck and worksheet replace them,
// the site keeps serving the UoS 2025 version from the lesson's archive.
const servedFromArchive = {
  5: 'lessons/lesson5/archive/uos-2025', 6: 'lessons/lesson6/archive/uos-2025', 7: 'lessons/lesson7/archive/uos-2025',
  8: 'lessons/lesson8/archive/uos-2025', 9: 'lessons/lesson9/archive/uos-2025', 10: 'lessons/lesson10/archive/uos-2025',
};  // released one lesson at a time, a week before class: see reviews/PENDING_RELEASES.md

// Lessons whose served deck is a Quarto deck in lecture/slides/ (the 2025 archives of Lessons 5–10 are Slidev).
const quartoDecks = { 1: 'slide', 2: 'slide_v2', 3: 'slide', 4: 'slide' };
// Lessons whose lab worksheet is a Quarto .qmd file (the others are .md).
const qmdWorksheets = new Set([2, 3, 4]);

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
