# Lesson 3 Note 的数据来源

- `auto_mpg.csv`：与 `lesson2/lab/auto_mpg.csv` 相同（来源与核对见 `lesson2/lab/DATA_SOURCE.md`）。
- `bike_hourly.csv`：UCI Bike Sharing Dataset 的逐小时数据（Capital Bikeshare，华盛顿，2011 到 2012 年，17379 条；Fanaee-T 与 Gama，2013）。经 OpenML（`fetch_openml("Bike_Sharing_Demand", version=2)`）于 2026-10-15 获取，只保留 `hour` 与 `count` 两列。
- 加州房价：scikit-learn 自带的 `fetch_california_housing`（1990 年美国人口普查，20640 个街区组），由 `build_numbers.py` 直接读取，不另存文件。
