# Lesson 5–10 Lab 对课表学习目标的覆盖与用时审计

日期：2026-10-27。只读审计，没有改任何课程材料。

## 依据与做法

- 学习目标：`course-site/src/data/timetable.json` 中各讲的 `coreOutcomes`（原文照录）。
- Lab：各讲 `lab/lab_worksheet.qmd` 的 Core 部分（Part 0 到最后一个 Part）。**Try with AI 框、After class、Advanced Study 不算“Lab 里亲手练到”**：Try with AI 写明“if you finish early, or after class”，基础弱的学生做不到。
- 判定：**有** = Core 里有亲手运行或填空、并有问题让学生回答；**部分** = 碰到了，但没有让学生做这条目标要求的那件事；**无** = Core 里没有。
- 出处：缺的部分写明在 Note（`lecture/note_en.qmd` 的节号）、Slides（`lecture/slides/slide.qmd`）或 Advanced 的哪里出现。
- 用时：按 Lesson 4 的新做法估，即学生打开**代码已放好的 Colab workbook**，只填空、运行、写答案，不复制代码；以基础最弱但认真跟做的学生为准。估法：读说明和 Question first 约 2 分钟，填一个空 1–2 分钟，每个书面问题约 1.5–2 分钟，加上 Colab 实际运行时间，开头读 “What this lab is for / How to work” 约 3 分钟。

**一个前提要先说明：** Lesson 5–10 目前**都还没有 workbook**（`lab/` 下没有 `build_workbook.py` 和 `lab_workbook.ipynb`，网站 `course.json` 只有 Lesson 1 和 4 的 workbook），worksheet 里仍写“Open a blank Colab notebook … Copy each code block into a cell”。各讲 Part 标注的分钟数加起来是 42–46 分钟，是按复制代码估的；一旦像 Lesson 4 那样改成 workbook，每讲都会掉到 35 分钟上下。所以下面的“用时”是按改成 workbook 之后估的。

---

## Lesson 5：Discriminative analysis（LDA、QDA）

| 目标 | Lab 里有没有 | 在哪 | 建议 |
|---|---|---|---|
| Use sklearn's LinearDiscriminantAnalysis and QuadraticDiscriminantAnalysis | 有 | Part 2.1（填两个类名）、3.1–3.3（QDA、`reg_param`） | 不改 |
| Explain why LDA assumes equal covariances | 部分 | Lab 只在 2.1 的注释里写 “one shared spread: straight boundary”，2.1 问题 1 数了 465 个数。“共用一个 spread → 边界是直线、要估的数少”这层没有让学生看见或回答。Note §4（含 “Why does one shared spread always give a straight line?”）、Note Advanced Study 1；Lab Advanced B 推导 w、b；直线 vs 椭圆边界的图只在 Part 3 之后的**第三个 Try with AI** 里 | 把第三个 Try with AI 升为 Core 一节（见下） |
| Describe when LDA vs QDA is appropriate | 部分 | 2.1 的 Question first（“QDA 在讲课的二维数据上最好，这里也会吗？”）和 3.1 QDA 在 784 像素上失败，都碰到了；但学生没有在同一处看到“QDA 赢”的情形，只看到它输。Note §4（二维数据 0.719 vs 0.867）、§6、§9 第 4 条 | 同上一节一起补：二维数据上 QDA 明显赢，接 2.1 的 30 维结果对照 |
| Compare LDA vs logistic regression performance | 有 | Part 2.1（0.979 / 0.960 / 0.958），3.3 Success check 对照 0.908 / 0.832 | 不改 |

**用时（workbook）：约 35–37 分钟，明显不足 45。** Part 0 约 3，1.1 约 4，1.2 约 8，2.1 约 8，3.1 约 4（含 30 秒运行），3.2 约 6（含 1 分钟运行），3.3 约 2，开头 3。

**最小补法：** 在 Part 2 前加 **“1.3 One shared spread, or one each”**（或放在 2.1 之前作 2.1，原 2.1 顺延）。照 Lesson 4 的 3.2 样板：

- 代码给全：第三个 Try with AI 里现成的二维数据（`X_two`、`y_two`），再加画两条 0.5 边界的 contour 代码（现在要学生问 AI 才能画）。
- 只留一个概念空：`lda_two = ____().fit(...)` 填哪个模型是“共用一个 spread”。
- Question first：“左图共用一个 spread，右图每类一个。哪一张的边界是直线？”
- 运行后问两三个问题：(1) 两个模型的准确率（约 0.719 / 0.867），为什么共用 spread 只能画直线（Note §4 的一句话即可）；(2) 这里 QDA 赢，下一节 30 个特征时它还会赢吗？——把原 2.1 的 Question first 接上，学生就同时看到“该用 QDA”和“不该用 QDA”的两种情形。
- 加约 7–8 分钟，总计约 43–45 分钟。Try with AI 里这一框相应删去。

---

## Lesson 6：Decision Tree

| 目标 | Lab 里有没有 | 在哪 | 建议 |
|---|---|---|---|
| Build decision trees using sklearn's DecisionTreeClassifier | 有 | 1.1、2.2、3.1 | 不改 |
| Explain how splitting criteria (Gini, entropy) work | 部分：Gini 有，entropy 无 | Gini：1.2 手算并有两个问题。Entropy：Note §2 只一句（“Its shape is very close to Gini, and it usually picks the same cut point”）；Note Advanced Study 1 第 3 题换成 entropy；Lab Advanced A 不含 entropy；Slides 讲者备注写“熵……Note 里一句” | 在 1.1 末加一行 `criterion="entropy"` 对照（见下） |
| Describe when trees overfit and how to prevent it | 有 | 2.2（20 个深度，训练到 1.0、验证停在 0.78，三个问题）、3.1 | 不改 |
| Interpret tree structure and feature importance | 部分：只读了根节点；feature importance 无 | Core 只打印 `tree_.feature[0]`、`tree_.threshold[0]`。树的图（`plot_tree`）在 Part 1 的 Try with AI；`feature_importances_` 在 Part 3 之后的 Try with AI。Note §7（F5 右图，mean concave points 0.72）和 §10 第 4 条（相关特征）；Slides “Less accurate here, but you can **read** it” 页 | 加一节读树（见下） |

**用时（workbook）：约 32–34 分钟，明显不足 45，是六讲里最短的。** Part 0 约 4，1.1 约 3，1.2 约 8，2.1 约 2，2.2 约 9（含约 30 秒运行），3.1 约 4，开头 3。

**最小补法（两处，合计约 10–12 分钟，总计约 43–45）：**

1. **1.1 加一个对照（约 2 分钟）。** 1.1 后面给一行代码全写好的 `DecisionTreeClassifier(max_depth=1, criterion="entropy", random_state=0)`，打印根节点特征和切点。Question first：“换成 entropy，第一刀会换地方吗？”运行后一问：为什么两种标准常选同一刀（Note §2 的形状图）。Success check 的数要跑一次再定。
2. **新加 “3.2 Read the tree”（约 8–10 分钟）。** 代码给全：在 30 个特征的肿块上拟合 depth-5 的树（Note §7 用的就是这棵），打印前 5 个 `feature_importances_` 和名字，并用 `plot_tree` 画 depth-2 的那棵（把两个 Try with AI 合成一节）。只留一个概念空：`tree.____` 填 `feature_importances_`，或者留“左下叶子判成哪类”。Question first：“你猜树最依赖哪一个测量？”运行后问：(1) 0.72 是什么意思；(2) 沿着 depth-2 的树，给一个 radius 14、texture 22 的肿块走一遍，判成什么；(3) 半径和周长高度相关，树只用了其中一个，另一个的 importance 接近 0，这能说明它和恶性无关吗？（Note §10 第 4 条）

---

## Lesson 7：Principal component analysis (PCA)

| 目标 | Lab 里有没有 | 在哪 | 建议 |
|---|---|---|---|
| Use sklearn's PCA for dimensionality reduction | 有 | 1.1、2.1、3.1、3.2 | 不改 |
| Explain why PCA finds directions of maximum variance | 部分 | 1.2 问题 2 对比“10 个像素 3.7% vs 10 个成分 48.9%”，2.1 看重建误差，都碰到“方差大＝留得多”；但学生没有亲手看“方差最大的方向＝重建误差最小的方向”。Note §3（二维点云转方向）、§4；Note Advanced Study 1 的 Exercise（勾股关系）；Lab Advanced A 第 1 题推导 | 加一节二维点云（见下） |
| Describe what eigenvalues represent | 无 | Lab Core 一次也没出现 “eigenvalue”，只用 `explained_variance_ratio_`，没有说它是“特征值 ÷ 特征值之和”，也没有用 `explained_variance_`。Note 只在 Advanced Study 1 出现（`λ = 投影方差`，`ratio = λ_j / Σλ_i`）；Lab Advanced A 第 2–4 题用 `np.linalg.eigh`。Slides 里没有找到 eigenvalue | 同一节里补上：特征值＝沿这个方向的方差 |
| Determine optimal number of components | 有 | 1.2（累计 90%）和 3.1（按 QDA 验证准确率选 k=40） | 不改 |

**用时（workbook）：约 30–33 分钟，明显不足 45。** Part 0 约 2，1.1 约 4，1.2 约 5，2.1 约 6，3.1 约 8（6 次 PCA+QDA，约半分钟），3.2 约 2，开头 3。

**最小补法：** 在 Part 1 前（或 1.1 与 1.2 之间）加 **“1.x Variance along a direction”**，约 10 分钟，总计约 41–43 分钟：

- 代码给全：Note §3 的二维点云；对 0°–180° 每隔 15° 的单位方向 `u`，打印投影方差和重建误差（两者相加是常数，Note Advanced Study 1 的 Exercise）；再拟合 `PCA()`，打印 `components_[0]` 的角度和 `explained_variance_`。
- 只留一个概念空：投影方差那一行留 `np.var(X_c @ ____, ddof=1)`，填方向 `u`；或者留“方差最大的那个角度”让学生读出来填进去。
- Question first：“哪一个方向把点云保留得最多？它的重建误差会最大还是最小？”
- 运行后问：(1) 方差最大的角度和 `components_[0]` 的角度一致吗？(2) 那个最大方差和 `explained_variance_[0]` 一样吗？——这就是第一个**特征值**：沿第一个成分的方差；(3) 回到 MNIST：`explained_variance_ratio_[0] = 0.0955` 用特征值怎么说？

---

## Lesson 8：k-means and Hierarchical Clustering

| 目标 | Lab 里有没有 | 在哪 | 建议 |
|---|---|---|---|
| Apply K-means clustering using sklearn | 有 | 1.1、2.1–2.2、3.1 | 不改 |
| Explain why K-means finds spherical clusters | 无 | Lab 全部在 MNIST 上做，没有一个“形状不圆”的例子。Note §4 “Where K-means Fails”（两环 62.5%、两带 51.3%，“两个中心之间的边界是直线”）；Slides “Where it fails: Two rings, two strips”“The rings are linked, not centred” | 加一节两环（见下） |
| Describe when hierarchical vs K-means is better | 部分 | Part 4 只演示了 single linkage 在数字上**失败**（0.115），Ward 与 K-means 接近。学生没有看到“层次聚类赢 K-means”的情形，4.1 的 Question first 只是引用课上的两环。Note §4–§5、§7（大数据要先抽样）；Slides 同上 | 与上一行同一节补 |
| Choose optimal K using elbow method | 无 | 只在 Part 4 之后的 Try with AI（k=2–20 画 inertia）。Note §3 末（“look for the bend, which here is at 3”）；Slides “More groups always lower … Look for the bend” 页 | 把这个 Try with AI 升为 Core（见下） |

**用时（workbook）：约 32–34 分钟，明显不足 45。** Part 0 约 2，1.1 约 5，2.1 约 4，2.2 约 7，3.1 约 5，4.1 约 6（两次各 10–20 秒），开头 3。

**最小补法（两处，合计约 12 分钟，总计约 44–46）：** 这是缺口最多的一讲（两条“无”、一条“部分”）。

1. **新加 “Part 1 前：Two rings”（约 7 分钟）。** 代码给全：`make_circles`（或 Note §4 的两环数据）上分别跑 `KMeans(n_clusters=2)` 和 `AgglomerativeClustering(n_clusters=2, linkage="single")`，并排画两张着色图，打印各自的多数票准确率。只留一个概念空：`linkage=____`。Question first：“K-means 会把两环分成内外两圈，还是左右两半？”运行后问：(1) 两种方法各认对多少（约 0.625 vs 1.0，数字跑一次再定）；(2) 两个环的中心在哪里？为什么 K-means 只能切出直边的块（＝它找的是围着一个中心的“圆团”）；(3) 那为什么 Part 4 里 single 在数字上反而失败？——这样学生两头都看到，“什么时候用哪一个”就有了依据。
2. **把 elbow 的 Try with AI 升为 Core “1.2 How many groups?”（约 5 分钟）。** 代码给全；为了省时间改为 k=2–15、`n_init=3`（约 20–30 秒，需实测）。Question first：“inertia 会在 k=10 出现明显拐点吗？”运行后问：(1) 有没有清楚的拐点？(2) 没有拐点时，这里为什么仍用 k=10？（Note §3 末）

---

## Lesson 9：Deep learning

| 目标 | Lab 里有没有 | 在哪 | 建议 |
|---|---|---|---|
| Implementing neural networks using Keras | 有 | 1.1、1.3、2.1、3.1 | 不改 |
| Explain why we need multiple layers | 部分 | Part 2 去掉中间的 activation，看到 118,282 个权重也只有 0.920 ≈ 逻辑回归 0.9277，回答了“为什么中间要有弯”。但“为什么要有隐藏层、几个单元能围住什么”是课上用两环讲的（Note §2 三条线围三角形、§3 widget：1/2/3/4 个单元 0.69/0.82/0.97/0.997），Lab 里没有亲手做；MNIST 上也没有对照 0 层与有隐藏层。Note Advanced Study 3 第 3 题（更宽或更深） | 加一节两环上的小网络（见下） |

**用时（workbook）：约 35–38 分钟，不足 45。** Part 0 约 2（下载 11 MB），1.1 约 5（三个带引号的空，基础弱的学生容易出错），1.2 约 4，1.3 约 5（约半分钟训练），2.1 约 5（约半分钟训练），3.1 约 7（约 1 分钟训练），Part 4 约 4，开头 3。

**最小补法：** 在 Part 1 前加 **“Part 1: Lines around the ring”**，约 8 分钟，总计约 43–46 分钟：

- 代码给全：课上的两环数据（2 个特征、几百个点），Keras 建 `Dense(h, activation="sigmoid") → Dense(1, activation="sigmoid")`，对 h = 1、2、4 各训练一次（几百个点，每次几秒），打印准确率；再给一个画边界的函数，画 h=1 和 h=4 两张。
- 只留一个概念空：隐藏单元数 `h`，或输出层的 activation。
- Question first：“只有 1 个隐藏单元时，边界能是什么形状？”
- 运行后问：(1) 准确率从 1 到 4 个单元怎么变（应接近 Note §3 的 0.69 → 0.997，数要跑一次再定）；(2) 每个单元画了一条什么线？4 条线一起围出了什么？(3) 接着 Part 2：如果把中间的 sigmoid 去掉，4 个单元还能围住吗？——这样“为什么要隐藏层”和“为什么要 activation”前后连上。

---

## Lesson 10：Deep learning Convolutional network

| 目标 | Lab 里有没有 | 在哪 | 建议 |
|---|---|---|---|
| Understand how a convolutional layer works | 部分 | 2.1 填 `Conv2D(16, 3)`，2.2 手算权重数（共享权重），3.1 平移测试，都在讲卷积的**效果**。但“一个 3×3 filter 滑过图像、乘加得到 feature map”这件事本身，学生没有亲手做或看到。Note §2（Horizontal / Vertical edge 的 widget，7 和 1）；Slides “A small patch, everywhere: Slide a filter over a digit”；学到的 filter 只在 2.3 后的 Try with AI | 加一节手设 filter（见下） |
| Implementing a CNN using Keras | 有 | 2.1、2.3、3.1、4.1 | 不改 |

**用时（workbook）：约 37–40 分钟，不足 45。** Part 0 约 2，1.1 约 2（约 20 秒训练），1.2 约 5，2.1 约 5，2.2 约 4，2.3 约 4（约 1 分钟训练），3.1 约 5，4.1 约 4，4.2 约 4，开头 3。

**最小补法：** 在 Part 2 前加 **“1.3 One filter, by hand”**，约 7 分钟，总计约 44–47 分钟：

- 代码给全：取一张验证集里的 7，定义 Note §2 的 “Horizontal edge” filter（上行 1、中行 0、下行 −1），用 `scipy.signal.correlate2d(..., mode="valid")`（或给 `Conv2D(1, 3)` 设好权重）算出 26×26 的 feature map，并排画原图、Horizontal edge、Vertical edge 三张图，打印 feature map 的形状。
- 只留一个概念空：Vertical edge filter 的 9 个数（照 Horizontal 的写法转 90°），或 feature map 的形状 `(____, ____)`。
- Question first：“Horizontal edge 的 feature map 在 7 的哪里最黑？”
- 运行后问：(1) 为什么 28×28 的图得到 26×26？(2) 换成数字 1，两张 feature map 哪张亮？(3) 2.1 里的 16 个 filter 和这个有什么不同？（答：权重是学出来的）

---

## 汇总：哪几讲要改、优先级

**共同问题：** 六讲改成 workbook 后，Core 都会在 30–40 分钟之间，全部明显不足 45 分钟；而且每讲都有一条到两条学习目标只在 Note、Slides 或 Try with AI 里出现。好在两件事可以一起解决：每讲加一节 Lesson 4 3.2 式的“代码给全、只留一个概念空、Question first、运行后两三个问题”，正好补上缺的目标，也补上约 8–12 分钟。

另外，这六讲都还没有 workbook，worksheet 的 “How to work” 第 1–2 条仍写复制代码，`build_workbook.py` 也要逐讲加；这一步应在补节之后做，否则 workbook 要生成两次。

| 优先级 | 讲 | 缺什么 | workbook 用时 | 补法 |
|---|---|---|---|---|
| 1 | Lesson 8 | 两条**无**（spherical、elbow），一条部分（何时用层次聚类） | 约 32–34 | 加两环一节 + elbow 升为 Core |
| 1 | Lesson 7 | 一条**无**（eigenvalue），一条部分（why max variance） | 约 30–33（最短之一） | 加二维点云一节，一次补两条 |
| 2 | Lesson 6 | feature importance **无**，entropy 无，读树只读根节点 | 约 32–34 | 加 “Read the tree” + entropy 一行对照 |
| 2 | Lesson 5 | 两条部分（equal covariances、何时 LDA/QDA） | 约 35–37 | 二维 LDA/QDA 边界从 Try with AI 升为 Core |
| 3 | Lesson 10 | 一条部分（卷积层怎样工作） | 约 37–40 | 加手设 filter 一节 |
| 3 | Lesson 9 | 一条部分（why multiple layers，只做了 activation 一半） | 约 35–38 | 加两环小网络一节 |

排序理由：课表上明写、Lab 完全没做的（“无”）排最前，Lesson 4 的 ROC/AUC 就是这一类；只做了一半的排后。Lesson 9、10 的学习目标只有两条，Lab 已经做了大半，缺口最小。

**还要实测的地方：** 上面新节里写的数字（两环的准确率、entropy 的切点、elbow 的运行时间、二维点云的角度）都没有跑过，补节时要先在 Colab 跑一次定 Success check；各讲用时也最好在 workbook 生成后按 Lesson 4 的做法实测一次。补完后按 `course-reviewer` 的 L6、L7 再审。
