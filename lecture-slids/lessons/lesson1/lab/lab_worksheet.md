---
pagetitle: "Lab 1: From Data to Insight"
---

# Lab 1: From Data to Insight
## MPS311/439 - Machine Learning
**Lesson 1 Lab Session | Duration: 50 minutes**

---

## Introduction

Welcome to your first machine learning lab. Before we train a model, we need to understand the data in front of us. This process is called **exploratory data analysis (EDA)**.

EDA helps us answer four questions:

1. What does each row and column represent?
2. Is the data complete and plausible?
3. What patterns can we see?
4. What can we responsibly conclude from those patterns?

Today you will investigate hourly bicycle rentals. By the end of the lab, you should be able to inspect a dataset, identify simple data-quality issues, choose a suitable basic visualisation, and communicate one evidence-based finding.

### Core learning outcomes

By the end of the core lab, you should be able to:

- load a CSV file into a pandas DataFrame;
- inspect rows, columns, data types, and summary statistics;
- identify missing values and duplicate rows;
- create and label a histogram, bar chart, scatter plot, and line plot;
- distinguish a **feature** from a **target**;
- describe a pattern without claiming more than the data supports;
- use AI to diagnose an error, then verify the proposed fix.

> **Need Python help?** Use the Python Preparation guide for variables, methods, imports, and error messages. You do not need to memorise Python syntax to complete this lab.

### Files you need

- This worksheet as a Jupyter Notebook.
- [bike_rentals.csv](./bike_rentals.csv)

The dataset is synthetic and was created for teaching. It contains 14 days of hourly bicycle-rental observations. The patterns are realistic enough for EDA, but the data must not be presented as measurements from a real city.

---

## Setup: Open, run, and save the notebook (5 minutes)

Download `bike_rentals.csv` from the Lesson 1 page and place it beside this notebook. If you are using Google Colab, the setup cell will ask you to upload the CSV when it cannot find the file.

Run the following cell without changing it.

```python
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_FILE = "bike_rentals.csv"

if not Path(DATA_FILE).exists():
    try:
        from google.colab import files
        print("Choose bike_rentals.csv from your computer.")
        files.upload()
    except ImportError:
        raise FileNotFoundError(
            "bike_rentals.csv was not found. Place it in the same folder as this notebook."
        )

df = pd.read_csv(DATA_FILE)
print("Dataset loaded successfully.")
```

Now check that the object named `df` is a pandas DataFrame.

```python
type(df)
```

You should see `pandas.core.frame.DataFrame`.

### Notebook check

1. Change the message below.
2. Run the cell.
3. Change it again and re-run it.

```python
message = "My Lesson 1 notebook is working."
print(message)
```

Save a copy of the notebook before continuing.

- **Google Colab:** File -> Save a copy in Drive
- **Jupyter:** File -> Save Notebook

---

## Part 1: Meet the dataset (8 minutes)

### Background

A DataFrame is a table with labelled rows and columns. Before plotting anything, we should understand what one row represents and what each column means.

### Task 1.1: Preview the first rows

Complete the method name.

```python
df.____()
```

**Hint:** The method is `.head()`.

### Task 1.2: Inspect the size and columns

Fill in the blanks.

```python
print("Rows and columns:", df.____)
print("Column names:", df.____.tolist())
```

**Hints:**

- `df.shape` gives `(number of rows, number of columns)`.
- `df.columns` stores the column labels.

Record your result:

> Number of rows: __________  Number of columns: __________

### Task 1.3: Understand the variables

Run the following cell.

```python
df.info()
```

Use the table and the descriptions below.

| Column | Meaning |
|---|---|
| `date` | Calendar date |
| `day` | Day of the week |
| `hour` | Hour of the day, from 0 to 23 |
| `temperature_c` | Temperature in degrees Celsius |
| `humidity_pct` | Relative humidity as a percentage |
| `wind_speed_kmh` | Wind speed in kilometres per hour |
| `weather` | Clear, Cloudy, or Rain |
| `is_weekend` | 1 for Saturday/Sunday, otherwise 0 |
| `rentals` | Number of bicycle rentals in that hour |

Answer in your own words:

1. What does one row represent?

> ______________________________________________________________________

2. If our future goal is to predict bicycle rentals, which column is the **target**?

> Target: ______________________________

3. Give three columns that could be used as **features**.

> Features: ____________________, ____________________, ____________________

### Task 1.4: Summary statistics

Complete and run the code.

```python
df.____()
```

**Hint:** Use `.describe()`.

Look at `rentals`. Record its minimum, mean, and maximum.

> Minimum: __________  Mean: __________  Maximum: __________

**Think:** Does the mean describe a typical hour well, or might the distribution be uneven?

---

## Part 2: Check data quality (8 minutes)

### Background

Real datasets are rarely perfect. Missing values, repeated rows, and implausible values can change our conclusions. Code running without an error does not guarantee that the data is trustworthy.

### Task 2.1: Find missing values

Fill in the method name.

```python
missing_values = df.____().sum()
print(missing_values)
```

**Hint:** Use `.isna()`.

Which columns contain missing values?

> ______________________________________________________________________

How many missing cells are there in total?

```python
total_missing = missing_values.____()
print("Total missing cells:", total_missing)
```

### Task 2.2: Find duplicate rows

Complete the code.

```python
duplicate_rows = df.____().sum()
print("Duplicate rows:", duplicate_rows)
```

**Hint:** Use `.duplicated()`.

### Task 2.3: Create a clean working copy

For this first lab, we will remove the duplicate row and the small number of rows containing missing values. In a real project, this decision would require more thought.

```python
df_clean = (
    df.drop_duplicates()
      .dropna()
      .copy()
)

print("Original shape:", df.shape)
print("Clean shape:   ", df_clean.shape)
```

How many rows were removed?

```python
rows_removed = len(____) - len(____)
print("Rows removed:", rows_removed)
```

### Task 2.4: Debug with AI, then verify

The code below contains an error. Run it and read the final line of the error message.

```python
average_rentals = df_clean["rental"].mean()
print(average_rentals)
```

Ask an AI assistant:

> I am learning pandas. Explain this error, identify the likely cause, and suggest the smallest change. Do not rewrite the full exercise.

Then answer:

1. What caused the error?

> ______________________________________________________________________

2. What is the corrected line?

```python
average_rentals = _______________________________________________
```

3. How did you verify that the fix was correct?

> ______________________________________________________________________

---

## Part 3: Explore individual variables (10 minutes)

### Background

Univariate analysis studies one variable at a time. A histogram helps us see where numerical values are concentrated, while a bar chart helps us compare category counts.

### Task 3.1: Distribution of rentals

Complete the column name, number of bins, and labels.

```python
plt.figure(figsize=(8, 5))
plt.hist(df_clean["____"], bins=____, color="steelblue", edgecolor="white")
plt.xlabel("____________________")
plt.ylabel("____________________")
plt.title("____________________")
plt.tight_layout()
plt.show()
```

**Hints:**

- Plot the `rentals` column.
- Start with `bins=20`.
- Every plot should say what the horizontal and vertical axes represent.

Describe the shape of the distribution:

> ______________________________________________________________________

Are most observations near the low, middle, or high end?

> ______________________________________________________________________

### Task 3.2: Count weather categories

```python
weather_counts = df_clean["weather"].____()
print(weather_counts)
```

**Hint:** Use `.value_counts()`.

Now draw a bar chart.

```python
plt.figure(figsize=(7, 4))
weather_counts.plot(kind="____", color=["#5e7ee8", "#e1a63d", "#7fa58d"])
plt.xlabel("Weather")
plt.ylabel("Number of hourly observations")
plt.title("Weather conditions in the dataset")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
```

Which weather category is most common?

> ______________________________________________________________________

---

## Part 4: Explore relationships (10 minutes)

### Background

Bivariate analysis studies the relationship between two variables. We will begin with a question:

> Do warmer hours tend to have more bicycle rentals?

### Task 4.1: Temperature and rentals

Fill in the column names.

```python
plt.figure(figsize=(8, 5))
plt.scatter(
    df_clean["________________"],
    df_clean["________________"],
    alpha=0.55,
    color="#3769cf"
)
plt.xlabel("Temperature (C)")
plt.ylabel("Bicycle rentals per hour")
plt.title("Temperature and bicycle rentals")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
```

Answer carefully:

1. Is the overall relationship positive, negative, or unclear?

> ______________________________________________________________________

2. Are there points that do not follow the overall pattern?

> ______________________________________________________________________

3. Does this plot prove that higher temperature **causes** more rentals? Why or why not?

> ______________________________________________________________________
>
> ______________________________________________________________________

### Task 4.2: Hourly pattern

Calculate the mean rentals for each hour.

```python
hourly_rentals = df_clean.groupby("____")["____"].mean()

plt.figure(figsize=(9, 5))
plt.plot(hourly_rentals.index, hourly_rentals.values, marker="o", color="#cc513f")
plt.xlabel("Hour of day")
plt.ylabel("Mean bicycle rentals")
plt.title("Average rental pattern across the day")
plt.xticks(range(0, 24, 2))
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
```

At what times do rentals peak? Suggest one realistic explanation.

> ______________________________________________________________________
>
> ______________________________________________________________________

---

## Part 5: Compare groups and investigate (6 minutes)

### Task 5.1: Weekday versus weekend

Complete the group and target columns.

```python
weekend_comparison = df_clean.groupby("___________")["___________"].mean()
weekend_comparison.index = ["Weekday", "Weekend"]

plt.figure(figsize=(6, 4))
weekend_comparison.plot(kind="bar", color=["#5e7ee8", "#ef6a4d"])
plt.xlabel("")
plt.ylabel("Mean bicycle rentals")
plt.title("Average rentals: weekday vs weekend")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
```

Which group has the higher overall mean?

> ______________________________________________________________________

Why might an overall mean hide important hourly differences?

> ______________________________________________________________________

### Task 5.2: Your mini-investigation

Choose **one** question:

1. How does humidity relate to rentals?
2. How do rentals differ across Clear, Cloudy, and Rain conditions?
3. Does the hourly rental pattern differ between weekdays and weekends?

Create one appropriate plot. You may adapt code from earlier parts. Give the plot a meaningful title and label every axis.

```python
# Write or adapt your code here.


```

Write two evidence-based sentences:

> Finding 1: ____________________________________________________________
>
> Finding 2: ____________________________________________________________

Write one conclusion that your plot **cannot** support:

> Limitation: ___________________________________________________________

**AI Help:** You may ask AI to recommend a plot type, but include the names and meanings of your variables. Decide for yourself whether its suggestion matches your question.

---

## Reflection and readiness check (3 minutes)

Answer without running new code.

1. What is the difference between a feature and a target?

> ______________________________________________________________________

2. Why should we inspect missing values before creating a model?

> ______________________________________________________________________

3. A scatter plot shows that rentals are higher in warmer hours. Give one possible confounding variable.

> ______________________________________________________________________

4. When AI suggests a code fix, what should you do before accepting it?

> ______________________________________________________________________

### Readiness checklist

- [ ] I can load a CSV file into pandas.
- [ ] I can inspect rows, columns, and data types.
- [ ] I can check missing values and duplicates.
- [ ] I can create a labelled plot.
- [ ] I can describe a pattern without confusing association with causation.
- [ ] I can read an error message and verify a proposed fix.

If two or more items remain unclear, revisit the relevant section of Python Preparation before Lesson 2.

---

## Core summary

Today you completed the first stage of a machine learning workflow:

> **Question -> Data -> Check -> Visualise -> Interpret**

You learned how to:

- inspect a DataFrame;
- identify simple data-quality issues;
- explore numerical and categorical variables;
- compare variables and groups visually;
- distinguish features from a target;
- communicate findings with appropriate caution.

In Lesson 2, we will continue the workflow:

> **Features + Target -> Model -> Prediction -> Evaluation**

---

# MPS439 Extension: Question-led EDA (30 minutes)

The extension is intentionally open-ended. Do not produce extra plots without a purpose. Begin with a question, choose a visualisation that answers it, and explain what the result means.

## Extension 1: Build a reusable data audit (10 minutes)

Write a function called `audit_dataframe(data)` that returns or prints:

- number of rows and columns;
- data type of every column;
- missing-value count for every column;
- number of duplicate rows;
- numerical summary statistics.

```python
def audit_dataframe(data):
    """Produce a compact quality report for a pandas DataFrame."""
    # Your implementation here.
    pass


audit_dataframe(df)
```

**Challenge:** Make the output concise enough that you would genuinely use it at the start of a future project.

## Extension 2: Multivariable visual exploration (10 minutes)

Choose one of the following:

### Option A: Weather changes the temperature-rentals relationship

Create a scatter plot of temperature against rentals, using colour to distinguish `weather`.

### Option B: Weekday and weekend hourly patterns

Create two lines on the same plot: mean hourly rentals for weekdays and weekends.

### Option C: Correlation heatmap

Calculate correlations between numerical variables and create an annotated heatmap. Explain why correlation with `hour` or `is_weekend` must be interpreted carefully.

You may use seaborn:

```python
import seaborn as sns
```

Your visualisation must include:

- a clear analytical question;
- an appropriate plot type;
- readable labels and legend;
- a caption explaining the main pattern.

## Extension 3: EDA mini-report (10 minutes)

Write a short report containing:

1. **Question:** What did you investigate?
2. **Evidence:** Which visual pattern supports your conclusion?
3. **Interpretation:** What real-world mechanism might explain it?
4. **Limitation:** What alternative explanation or missing variable matters?
5. **Next step:** What would you investigate or model next?

Aim for 150-250 words. Analytical depth matters more than the number of plots.

## Extension reflection

1. Which visualisation was most informative, and why?
2. Did any plot change your initial assumption?
3. Which variable would be most dangerous to interpret causally?
4. What additional data would make the analysis more credible?

---

## Lab complete

Keep your completed notebook. The EDA habits from this lab - checking quality, choosing purposeful plots, and explaining limitations - will be expected throughout the course and in both assessments.

<!-- COURSE_FEEDBACK_QR:START -->
---

## 30-second feedback

Scan the code to share anonymous feedback or post a question for this lab. It opens the correct **Lesson 01 lab** record automatically.

![Feedback QR code for Lesson 01 lab](./feedback-qr.png){fig-align="center" width="180px"}
<!-- COURSE_FEEDBACK_QR:END -->
