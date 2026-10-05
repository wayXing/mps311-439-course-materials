---
pagetitle: "Lab 1: Getting Started with Data and AI"
---

# Lab 1: Getting Started with Data and AI
## MPS311/439 — Machine Learning
**Lesson 1 Lab Session | Core duration: 50 minutes**

## What this lab is for

This first lab is a low-pressure return to Python. You may be opening a notebook for the first time, or you may have learned Python before and forgotten some of it. Both are expected.

Today you will practise three things:

1. load a dataset into a notebook;
2. inspect it and make a few simple plots;
3. use AI when you are stuck, then check whether its suggestion works.

You are **not** expected to build a machine-learning model today. Most code is provided. Your job is to run it, complete small gaps, make simple changes, and understand the output.

## Choose how you want to work

Both routes complete the same lab:

- **Guided route — recommended if Colab or Python feels unfamiliar:** open the **Lab workbook** from the course website. It contains the task structure, setup code, small gaps, and spaces for your answers.
- **From-scratch route — if you prefer to type everything yourself:** create a blank Colab notebook and use this worksheet as your guide. Copy or type each code example as you reach it.

The workbook is not the solution. It does not contain completed answers or final outputs. Use the worked solution only after attempting the tasks, to check your work.

### If you are stuck

Use the same routine throughout this lab:

1. Read the final line of the error message.
2. Check spelling, quotation marks, brackets, and whether the cell above has been run.
3. If the problem remains, give an AI assistant:
   - what you are trying to do;
   - the relevant code;
   - the complete error message.
4. Try the smallest suggested change and check the result.

Do not ask only “fix my code”. Ask the AI to explain what was wrong so that you can recognise the problem next time.

## Setup: open, run, and save (5 minutes)

### If you are using Google Colab for the first time

1. On the course website, select **Lab workbook**. Colab will open in a new browser tab.
2. Sign in to your Google account if Colab asks you to.
3. Select **File → Save a copy in Drive**. Work in the copied notebook so that your changes are saved.
4. Run a code cell by selecting the round **▶** button on its left. You can also press **Shift + Enter**.
5. Run the setup cell and wait until you see `Dataset loaded successfully.` The dataset is loaded automatically from the public course repository.

Colab saves your Drive copy automatically. You can also select **File → Save** before leaving.

The main setup is intentionally short. It uses a public data address that has been tested with `pandas`. If it does not work, continue to the backup appendix at the end of this worksheet.

Run the following cell without changing it.

```python
import matplotlib.pyplot as plt
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/wayXing/mps311-439-course-materials/main/lecture-slids/lessons/lesson1/lab/bike_rentals.csv"
df = pd.read_csv(DATA_URL)

print("Dataset loaded successfully.")
print("Rows and columns:", df.shape)
```

**Success check:** You should see `Dataset loaded successfully.` followed by the number of rows and columns.

Now confirm that you can edit and re-run a cell.

```python
message = "My Lesson 1 notebook is working."
print(message)
```

Change the message, run the cell again, and save your own copy of the notebook.

> **Python reminder:** Text inside quotation marks is a string. `print(...)` displays a value.

## Part 1: Meet the dataset (8 minutes)

### 1.1 Look at the first rows

Complete the method name and run the cell.

```python
df.____()
```

**Hint:** Use `head`.

> **Python reminder:** A method belongs to an object. We call it with a dot and round brackets: `object.method()`.

This is a synthetic teaching dataset of hourly bicycle rentals. Its patterns are for practising analysis and are not evidence about a real city. One row represents one hour. The columns are:

- `date`: calendar date;
- `day`: day of the week;
- `hour`: hour from 0 to 23;
- `temperature_c`: temperature in degrees Celsius;
- `humidity_pct`: relative humidity;
- `wind_speed_kmh`: wind speed;
- `weather`: Clear, Cloudy, or Rain;
- `is_weekend`: 1 at weekends, otherwise 0;
- `rentals`: bicycles rented during that hour.

**Question:** What does one row represent?

> ______________________________________________________________________

### 1.2 Check the size and column names

Fill in the two attributes.

```python
print("Rows and columns:", df.____)
print("Column names:", df.____.tolist())
```

**Hints:** Use `shape` and `columns`.

> **Python reminder:** Attributes such as `df.shape` do not use round brackets. Methods such as `df.head()` do.

Record the result:

> Rows: __________  Columns: __________

## Part 2: Inspect the data (8 minutes)

### 2.1 Data types and missing entries

Run the complete command below.

```python
df.info()
```

**Success check:** You should see every column name, its number of non-missing values, and its data type.

Which columns appear to have missing entries?

> ______________________________________________________________________

### 2.2 Simple numerical summary

Complete and run the method.

```python
df.____()
```

**Hint:** Use `describe`.

Find the minimum, mean, and maximum for `rentals`.

> Minimum: __________  Mean: __________  Maximum: __________

**Question:** Would the mean alone describe both quiet and busy hours well? Use the minimum and maximum to explain your answer.

> ______________________________________________________________________

## Part 3: Check data quality and use AI (10 minutes)

### 3.1 Count missing values

Fill in the method name.

```python
missing_values = df.____().sum()
print(missing_values)
```

**Hint:** Use `isna`.

### 3.2 Count duplicate rows

Fill in the method name.

```python
duplicate_rows = df.____().sum()
print("Duplicate rows:", duplicate_rows)
```

**Hint:** Use `duplicated`.

For today, use the provided cleaning code. In a real project, removing data would require more thought.

```python
df_clean = df.drop_duplicates().dropna().copy()

print("Original shape:", df.shape)
print("Clean shape:   ", df_clean.shape)
```

### 3.3 Diagnose an error with AI

The following code contains a common mistake. Run it and read the error.

```python
df_clean["weather_type"].value_counts()
```

Ask an AI assistant:

> I am trying to count the rows in each weather category in a pandas DataFrame. My code is `df_clean["weather_type"].value_counts()`. The error is pasted below. Explain the cause and show the smallest correction. The available columns are: date, day, hour, temperature_c, humidity_pct, wind_speed_kmh, weather, is_weekend, rentals.

Paste the complete error after the prompt. Then make the correction and run the code again.

**What was wrong?**

> ______________________________________________________________________

**How did you check that the correction worked?**

> ______________________________________________________________________

## Part 4: Make three simple plots (15 minutes)

### 4.1 First plot: run a complete example

Run this code to see the distribution of hourly rentals.

```python
plt.figure(figsize=(7, 4))
plt.hist(df_clean["rentals"], bins=15, edgecolor="white")
plt.xlabel("Hourly bicycle rentals")
plt.ylabel("Number of hours")
plt.title("Distribution of hourly rentals")
plt.show()
```

**Question:** Are most hours near the lower, middle, or upper end of the rental range?

> ______________________________________________________________________

### 4.2 Second plot: complete a group comparison

Calculate the average rentals for each weather category.

```python
rentals_by_weather = df_clean.groupby("____")["____"].mean()
print(rentals_by_weather)
```

**Hints:** Group by `weather` and calculate the mean of `rentals`.

Now complete the plot type.

```python
rentals_by_weather.plot(kind="____", color="#4C78A8")
plt.xlabel("Weather")
plt.ylabel("Average hourly rentals")
plt.title("Average rentals by weather")
plt.show()
```

**Hint:** Use `bar`.

**Question:** Which weather category has the highest average rentals?

> ______________________________________________________________________

### 4.3 Third plot: modify a successful template

This code calculates average rentals for every hour of the day.

```python
rentals_by_hour = df_clean.groupby("hour")["rentals"].mean()

rentals_by_hour.plot(kind="line", marker="o", color="#E07A5F")
plt.xlabel("Hour of day")
plt.ylabel("Average rentals")
plt.title("Average rentals through the day")
plt.show()
```

Run it once. Then change the line colour or marker and run it again.

**Question:** At approximately which hours are rentals highest?

> ______________________________________________________________________

## Part 5: Finish and reflect (4 minutes)

Write one simple observation that is directly supported by your output.

> The data show that ____________________________________________________

Write one sentence about how AI helped you today.

> AI helped me __________________________________________________________

Before finishing, make sure that you can:

- run and re-run a notebook cell;
- load and preview a CSV file;
- inspect columns and simple summaries;
- complete and modify a small plotting example;
- use an error message to ask AI for focused help;
- check whether the proposed fix works.

# Advanced Study — choose one investigation

MPS311 students are encouraged to explore. MPS439 students should complete at least one investigation below. Allow about 30 minutes for a first investigation; finding and analysing a new dataset may take longer and can be continued after the lab. You may let AI write the code. You are responsible for the question, checks and interpretation. No machine-learning model is required.

## Breadth A: Direct AI to build a richer visual analysis

**Question:** Does the overall hourly rental pattern hide differences between weekdays and weekends?

Ask AI to propose two complementary views: a grouped hourly line plot and a box plot of rentals by weekday/weekend. Give it the actual column names and explain that `is_weekend` is 0 for weekdays and 1 for weekends. Ask it to show the number of observations in each group as well as the plots. Do not accept a plot merely because it looks impressive.

Start with this prompt, then refine it:

> I have a synthetic hourly bicycle-rental dataset in `df_clean`. Columns are date, day, hour, temperature_c, humidity_pct, wind_speed_kmh, weather, is_weekend, rentals. Does the overall hourly pattern hide weekday/weekend differences? Suggest two complementary plots and explain what each answers. Generate pandas/matplotlib code for Colab, include group counts, label units, and do not invent results. Use only these columns.

Before running the code, check its column names and grouping. After running it, check one plotted value against a printed pandas summary. Then ask AI for one useful revision, such as separate panels or a heatmap of mean rentals by day and hour. Explain whether the revision makes the comparison clearer.

**Deliver:** two plots, group counts, one numerical check, one revision with its reason, and a conclusion that explains what the overall mean hides. A box plot shows spread; a mean curve does not.

## Breadth B: Find data you want to investigate

Find a small tabular dataset about a topic you care about, such as sport, transport, energy or entertainment. Start at [Kaggle Datasets](https://www.kaggle.com/datasets) or [data.gov.uk](https://www.data.gov.uk/search). Search for your topic plus `CSV`. Read the dataset description and original source before downloading. Choose one CSV with documented columns, a numerical variable and a meaningful grouping or time variable. If finding suitable data takes more than ten minutes, continue with Breadth A and return to this later.

Download and unzip the CSV if necessary. In Colab, open **Files → Upload**, upload the CSV, and ask AI to help load it with pandas. Give AI the column names and a few non-sensitive sample rows. Check the row count, types, missing entries and units before analysing it; ask about unfamiliar codes rather than guessing.

Write your own question before requesting plots. Ask AI for two different plots that help answer it, then run and inspect the code. Revise at least one suggestion and check one plotted quantity against a numerical summary.

**Deliver:** the dataset page and original source, what one row represents, units for the variables used, your question, two plots, one verified value, and a short conclusion with one limitation. Explain why each plot helps answer your question. Keep the notebook runnable from the uploaded CSV; record its filename. There is no single model answer for this investigation.

## Depth C: Audit a claim and a cleaning decision

**Claim to investigate:** “Clear weather causes more bicycle rentals.”

Use the course data. First calculate mean rentals and group counts by weather. Then compare weather categories separately within weekdays and weekends. You may ask AI to write the grouped analysis and plot, but check its calculations yourself. Does the grouped view strengthen, qualify or weaken the original description? Explain why neither view establishes causation.

Now investigate the provided cleaning rule. Compare two versions: (1) remove duplicates and all rows with missing values; (2) remove duplicates and only rows missing the variables needed for the weather/rentals comparison. Print the number of rows retained and weather means for both. Do missing temperature, humidity or wind values prevent this particular calculation? Explain which cleaning rule you would use for this question and why. Do not claim the better rule must apply to every analysis.

**Deliver:** a table or plot with group counts, the two cleaning comparisons, one checked value, a revised evidence-based claim, and one unresolved limitation. An unchanged conclusion is acceptable if you support it with the comparison.

## Your investigation record

For whichever option you choose, finish with:

- **Question and choice:** what did you investigate, and why this data/view?
- **Evidence:** what do your outputs show?
- **AI audit:** what did you check or change, and how?
- **Conclusion and limit:** what can you say, and what remains unsupported?

Save your code and outputs in your own notebook. The aim is a defensible analysis, not a large number of plots.

## Appendix: only if the data did not load

Do not use this section if the main setup worked.

First try the independent CDN copy:

```python
BACKUP_URL = "https://cdn.jsdelivr.net/gh/wayXing/mps311-439-course-materials@main/lecture-slids/lessons/lesson1/lab/bike_rentals.csv"
df = pd.read_csv(BACKUP_URL)
```

If both websites are unavailable, download `bike_rentals.csv` from the course materials and use Colab's **Files → Upload** button. Then run:

```python
df = pd.read_csv("bike_rentals.csv")
```

## 30-second feedback

Scan the feedback QR code and submit one thing that helped you today, one point that remained difficult, or one question for the next session.

![](feedback-qr.png){fig-alt="Feedback QR code for Lesson 1 lab" width="35%"}
