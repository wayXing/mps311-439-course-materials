# Auto 教学数据来源

`auto_mpg.csv` 从 UCI Machine Learning Repository 的 Auto MPG 原始文件提取，保留 `mpg`、`cylinders`、`displacement`、`horsepower`、`weight`、`acceleration`、`model_year` 七列，不含 `origin` 与 `car name`。原始文件有 398 行，其中 6 行的 `horsepower` 为 `?`；删除这 6 行后保留 392 行。没有填补缺失值，也没有对数值做标准化或其他改动。

- 数据来源：https://archive.ics.uci.edu/dataset/9/auto
- 原始文件：https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data
- 数据引用：Quinlan, R. (1993). Auto MPG [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5859H
- UCI 页面标注许可：CC BY 4.0。

此数据记录历史车辆的城市工况燃油经济性。`mpg` 越高表示单位燃油可行驶距离越远。课程只用它分析同来源样本中的预测关联，不作现代车辆性能或马力因果效应的结论。

学生代码从公开仓库读取：`https://raw.githubusercontent.com/wayXing/mps311-439-course-materials/main/lecture-slids/lessons/lesson2/lab/auto_mpg.csv`。发布前需把本文件夹的 `auto_mpg.csv` 推送到该仓库，并用学生代码实测。
