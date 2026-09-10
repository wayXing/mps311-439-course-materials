import { readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import course from '../src/data/course.json' with { type: 'json' };

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const lessonsDirectory = resolve(scriptDirectory, '..', '..', 'lecture-slids', 'lessons');
const academicYear = course.academicYear.replace(/[^0-9]+/g, '-').replace(/^-|-$/g, '');
const startMarker = '<!-- COURSE_FEEDBACK_QR:START -->';
const endMarker = '<!-- COURSE_FEEDBACK_QR:END -->';

const replaceMarkedSection = (source, section) => {
  const expression = new RegExp(`${startMarker}[\\s\\S]*?${endMarker}`, 'g');
  return expression.test(source) ? source.replace(expression, section) : `${source.trimEnd()}\n\n${section}\n`;
};

const lectureSection = (lesson) => `${startMarker}
---
layout: center
class: text-center
---

# 30-second feedback

<p class="text-xl mb-4">Scan this code to share anonymous feedback or post a question for this lecture.</p>

<img src="./feedback-qr.svg" alt="Feedback QR code for Lesson ${lesson.number} lecture" class="w-44 mx-auto rounded-lg shadow" />

<p class="text-sm mt-4 opacity-70">MPS311/439 · ${academicYear} · Lesson ${lesson.number} lecture</p>
${endMarker}`;

const labSection = (lesson) => `${startMarker}
---

## 30-second feedback

Scan the code to share anonymous feedback or post a question for this lab. It opens the correct **Lesson ${lesson.number} lab** record automatically.

![Feedback QR code for Lesson ${lesson.number} lab](./feedback-qr.png){fig-align="center" width="180px"}
${endMarker}`;

const teachingLessons = course.lessons.filter((lesson) => lesson.type === 'teaching');

for (const lesson of teachingLessons) {
  const lessonDirectory = resolve(lessonsDirectory, `lesson${Number(lesson.number)}`);
  const slidePath = resolve(lessonDirectory, 'lecture', 'slide.md');
  const labPath = resolve(lessonDirectory, 'lab', 'lab_worksheet.md');

  let slides = await readFile(slidePath, 'utf8');
  const slidesHadLegacyCode = /(?:\.\/|\.\.\/\.\.\/|\.\.\/\.\.\/\.\.\/)?QR_Code_for_real_time_feedback\.png/.test(slides);
  slides = slides.replace(/(?:\.\.\/)*QR_Code_for_real_time_feedback\.png/g, 'feedback-qr.svg');
  if (!slidesHadLegacyCode) slides = replaceMarkedSection(slides, lectureSection(lesson));
  await writeFile(slidePath, slides, 'utf8');

  let lab = await readFile(labPath, 'utf8');
  const labHadLegacyCode = /QR_Code_for_real_time_feedback\.png/.test(lab);
  lab = lab.replace(/(?:\.\.\/)*QR_Code_for_real_time_feedback\.png/g, 'feedback-qr.svg');
  if (!labHadLegacyCode) lab = replaceMarkedSection(lab, labSection(lesson));
  await writeFile(labPath, lab, 'utf8');
}

console.log(`Installed session-specific feedback QR codes for ${teachingLessons.length} lectures and labs.`);
