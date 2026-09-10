# MPS311 / MPS439 Course Site

Astro + TypeScript 课程网站。网站负责课程导航与资源展示；每个 Lesson 的 slides 继续由相邻的 `lecture-slids` Slidev 项目生成。

## 常用命令

```sh
npm ci
npm run dev
npm run build
npm run check
```

- `npm run dev`：同步课程资源和 timetable CSV，并启动本地网站。
- `npm run build`：同步资源与 timetable，并输出静态网站到 `dist/`。
- `npm run package:sites`：把成功构建的静态网站打包为 Sites 托管产物。
- `npm run check`：运行 Astro/TypeScript 检查，并确认所有页面和公开资源都已生成。
- `npm run sync:course`：只从 `../lecture-slids/lessons` 更新公开材料。
- `npm run sync:timetable`：从课程根目录的 `MPS311_439_TimeTable-byLessons-2026.csv` 更新 2026 秋季按 Lesson 编排的课程计划和下载文件。
- `npm run feedback:qr -- --base-url https://ml.wxing.me`：为全部 lecture/lab 生成按学年和课次区分的二维码，并写入网站与本地课程材料。
- `npm run feedback:install`：在各 lecture/lab 源文件中添加或更新二维码区块；随后重建 Slidev 与 Quarto 资料并运行 `npm run sync:course`。

课程 Demo 的源文件仍是每周 lecture 文件夹中的 `.ipynb`。同步时会用已安装的 Quarto 生成无需执行代码的静态 HTML 阅读版，并同时保留原始 Notebook 下载；因此运行 `dev` 或 `build` 的环境需要能够调用 `quarto`。

## Google Colab 与公开课程镜像

网站的 **Demo** 按钮直接打开 Google Colab，地址由 `src/data/course.json` 的 `colab`
配置和每个 resource 的 notebook 路径组成。当前公开源为
[`wayXing/mps311-439-course-materials`](https://github.com/wayXing/mps311-439-course-materials)
的 `main` 分支；静态 HTML 阅读版与 `.ipynb` 下载仍由本网站提供。

不要直接编辑 `../mps311-439-course-materials/` 中复制出来的文件。课程源或此配置改动后，
从本目录运行 `npm run sync:public`，再在该镜像目录审核、提交并推送。脚本只重建镜像中的
`course-site/` 与 `lecture-slids/`，保留其 `.git`；它刻意排除教学管理资料、密钥、部署配置、
生成 PDF/HTML、依赖和 archive。公开代码使用 MIT，课程材料使用 CC BY-NC-SA 4.0。

## 更新课程内容并上线

课程资料的**唯一源文件**在相邻的 `../lecture-slids/lessons/`；不要直接编辑
`public/materials/` 或 `dist/`。更新后按改动类型处理：

| 改动 | 所在位置 | 必要操作 |
| --- | --- | --- |
| Lecture slides | `lecture-slids/lessons/lessonN/lecture/slide.md` | 重新构建该 Lesson 的 Slidev standalone HTML；如 Blackboard 也要更新，再导出 PDF。 |
| Lab worksheet | `lecture-slids/lessons/lessonN/lab/lab_worksheet.md` | 用 Quarto 重新渲染该 Lesson 的 HTML；PDF 只在 Blackboard 版本需要时生成。 |
| 课程 Demo notebook | `lecture-slids/lessons/lessonN/lecture/*.ipynb` | 同步网站时会由 Quarto 生成静态 HTML。 |
| 网站文案、布局或样式 | `course-site/src/` | 不需要重新编译 slides/labs。 |
| 学年、反馈记录 | `/admin/` | 后台操作立即生效，不需要网站部署。 |

以 Lesson 3 lecture 为例，完整且安全的上线流程：

```sh
# 1. 编辑 canonical source 后，生成学生在网站上查看的交互式 slides
cd ../lecture-slids
npm run build:standalone -- lessons/lesson3/lecture/slide.md --out dist --base './'

# 2. 可选：为 Blackboard 更新 PDF；网站不会部署 PDF
npm run export -- lessons/lesson3/lecture/slide.md --output lessons/lesson3/lecture/slide-export.pdf

# 对 lab 修改，改用：quarto render lessons/lesson3/lab/lab_worksheet.md --to html

# 3. 同步生成资料到网站、验证，然后部署正式域名
cd ../course-site
npm run sync:course
npm run check
vercel --prod --yes
```

只修改网站自身时可省略前两步：

```sh
cd course-site
npm run check
vercel --prod --yes
```

部署前必须运行 `npm run sync:course`（课程材料有改动时）和 `npm run check`。
Vercel 的正式别名是 `https://ml.wxing.me`；部署成功后无需另行修改 DNS。二维码只在
新增学年、调整课次结构或更换正式域名时重新生成，不是每次修改内容都需要更新。

## Feedback QR codes

二维码固定使用公开正式域名 `https://ml.wxing.me`，绝不使用受保护的 Vercel preview 地址。每个码编码 `year` 和 `session`，例如 `2026-27-lesson-03-lab`；扫码会自动打开相应学年和课次的反馈页。新学年在更新 `course.json` 后，以 `--year YYYY-YY` 重新运行生成器，并重新安装、编译和同步课程资料即可。

## 需要修改的核心文件

- `src/data/course.json`：课程 Lesson、标题、介绍、主题和公开资源的唯一清单。
- `src/styles/theme.css`：全站颜色、字体、圆角、阴影等视觉变量。
- `src/styles/global.css`：具体页面和组件的布局样式。
- `src/pages/index.astro`：首页内容。
- `src/pages/course/[slug].astro`：所有 Lesson 共用的详情页模板。
- `src/pages/timetable.astro`：教学、workshop 和 assessment 日程页。

## 课程材料

`public/materials/` 和 `dist/` 都是生成内容，不应手工编辑，也不会提交到版本控制。每次启动或构建网站时，脚本会根据 `src/data/course.json` 自动复制学生可见的 slides、讲义、demo、lab worksheet 和 lab solutions。当前每个 Lesson 的 solution notebook 会同时发布为网页阅读版和 `.ipynb` 下载；是否公开由 `src/data/course.json` 中对应的 solution resource 决定。

## 当前范围

网站目前包含静态课程材料、课表，以及按 lecture/lab 区分的课程互动页面。学生可以公开提交希望在课堂上解答的问题，并使用任意显示名；评分和课堂改进建议通过独立通道匿名提交。反馈数值汇总对学生可见，文字建议只保存在教师可访问的 Supabase 后台。写入请求经过轻量限流，同一浏览器对同一课次重复提交反馈时只更新原记录。
