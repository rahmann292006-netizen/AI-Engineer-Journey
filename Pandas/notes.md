# Pandas Notes

## What is Pandas?

Pandas is an open-source Python library used for data manipulation, analysis, and cleaning. It provides powerful and easy-to-use data structures for handling structured data.

---

# Why Pandas?

- Easy data manipulation
- Fast data analysis
- Handles missing values
- Reads data from multiple file formats
- Widely used in Data Science and Machine Learning

---

# Main Data Structures

## 1. Series

A Series is a one-dimensional labeled array.

Example:

```python
import pandas as pd

s = pd.Series([10,20,30,40])

print(s)
```

---

## 2. DataFrame

A DataFrame is a two-dimensional table consisting of rows and columns.

Example:

```python
import pandas as pd

data = {
    "Name":["Ali","Rahman","Sara"],
    "Age":[20,21,19]
}

df = pd.DataFrame(data)

print(df)
```

---

# Reading Data

Read a CSV file:

```python
df = pd.read_csv("students.csv")
```

Useful Functions:

```python
df.head()
df.tail()
df.shape
df.columns
df.info()
df.describe()
```

---

# Key Learnings

- Learned what Pandas is.
- Understood the difference between Series and DataFrame.
- Created DataFrames using dictionaries.
- Loaded datasets using `read_csv()`.
- Explored datasets using built-in functions.
- Learned basic inspection techniques before analysis.

---

# Day 7 Status

✅ Pandas Fundamentals Completed

Next:
- Data Selection
- Indexing
- Filtering

---

# Day 8 - Indexing, Selection & Filtering

## Selecting Columns

Select a single column:

```python
df["Name"]
```

Select multiple columns:

```python
df[["Name", "Age"]]
```

---

## Selecting Rows

Select rows using slicing:

```python
df[0:5]
```

---

## loc[]

`loc[]` selects data using labels.

Example:

```python
df.loc[0]
df.loc[0:3]
```

---

## iloc[]

`iloc[]` selects data using integer positions.

Example:

```python
df.iloc[0]
df.iloc[0:3]
```

---

## Filtering Data

Example:

```python
df[df["Age"] > 20]
```

Multiple conditions:

```python
df[(df["Age"] > 20) & (df["Marks"] > 80)]
```

---

## Sorting

Ascending:

```python
df.sort_values("Marks")
```

Descending:

```python
df.sort_values("Marks", ascending=False)
```

---

# Key Learnings

- Selected rows and columns efficiently.
- Used `loc[]` and `iloc[]` for indexing.
- Filtered data using logical conditions.
- Sorted datasets based on column values.
- Learned essential data selection techniques used in real-world analysis.

---

# Day 9 - Handling Missing Values & Data Cleaning

## Missing Values

Missing values occur when data is incomplete or unavailable.

Check missing values:

```python
df.isnull()
```

Count missing values:

```python
df.isnull().sum()
```

---

## Dropping Missing Values

Remove rows with missing values:

```python
df.dropna()
```

Remove columns with missing values:

```python
df.dropna(axis=1)
```

---

## Filling Missing Values

Fill with a constant value:

```python
df.fillna(0)
```

Fill using mean:

```python
df["Marks"] = df["Marks"].fillna(df["Marks"].mean())
```

Fill using median:

```python
df["Marks"] = df["Marks"].fillna(df["Marks"].median())
```

---

## Duplicate Values

Check duplicates:

```python
df.duplicated()
```

Remove duplicates:

```python
df.drop_duplicates()
```

---

## Rename Columns

```python
df.rename(columns={"Marks":"Score"})
```

---

## Key Learnings

- Identified missing values.
- Removed missing values using `dropna()`.
- Filled missing values using `fillna()`.
- Detected and removed duplicate rows.
- Renamed columns for better readability.
- Learned basic data cleaning techniques used before Machine Learning.

---

# Day 10 - Data Transformation using Pandas

## apply()

The `apply()` function applies a custom function to rows or columns.

```python
df["Salary"] = df["Salary"].apply(lambda x: x + 5000)
```

---

## map()

Used to map existing values to new values.

```python
df["Gender"] = df["Gender"].map({
    "M":"Male",
    "F":"Female"
})
```

---

## replace()

Replaces existing values.

```python
df["City"] = df["City"].replace({
    "BLR":"Bangalore",
    "HYD":"Hyderabad"
})
```

---

## String Operations

Common string methods:

```python
df["Name"].str.upper()
df["Name"].str.lower()
df["Name"].str.len()
df["Name"].str.title()
```

---

## Creating New Columns

```python
df["Bonus"] = df["Salary"] * 0.10
```

---

## Key Learnings

- Learned to transform data using `apply()`.
- Mapped categorical values with `map()`.
- Replaced values using `replace()`.
- Performed string operations.
- Created new columns from existing data.
- Understood real-world data preprocessing techniques.

---

---

# Day 11 - GroupBy Fundamentals

## What is GroupBy?

GroupBy follows the **Split → Apply → Combine** strategy.

It groups rows based on common values and performs calculations on each group.

---

## Creating Groups

```python
df.groupby("Department")
```

---

## Common Aggregations

### Count

```python
df.groupby("Department")["Salary"].count()
```

### Sum

```python
df.groupby("Department")["Salary"].sum()
```

### Mean

```python
df.groupby("Department")["Salary"].mean()
```

### Maximum

```python
df.groupby("Department")["Salary"].max()
```

### Minimum

```python
df.groupby("Department")["Salary"].min()
```

### Size

```python
df.groupby("Department").size()
```

---

## Applications

- Department-wise salary analysis
- Sales reports
- Student marks analysis
- Customer segmentation

---

## Key Learnings

- Understood Split → Apply → Combine.
- Learned grouping using `groupby()`.
- Applied aggregation functions.
- Performed grouped data analysis.
- Learned why GroupBy is one of the most important features in Pandas.

---

---

# Day 12 - Advanced GroupBy Operations

## agg()

Performs multiple aggregation functions simultaneously.

```python
df.groupby("Department").agg({
    "Salary":["mean","max","min","sum"],
    "Experience":["mean","max"]
})
```

---

## describe()

Provides statistical summary for each group.

```python
df.groupby("Department").describe()
```

---

## get_group()

Returns a specific group.

```python
df.groupby("Department").get_group("IT")
```

---

## Iterating Through Groups

```python
for dept, group in df.groupby("Department"):
    print(dept)
    print(group)
```

---

## Multiple Column Grouping

```python
df.groupby(["Department","Gender"]).mean()
```

---

## Advantages

- Flexible data analysis.
- Multiple statistics in one step.
- Better insights from grouped datasets.
- Useful for Exploratory Data Analysis (EDA).

---

## Key Learnings

- Learned advanced aggregation using `agg()`.
- Generated statistical summaries with `describe()`.
- Retrieved specific groups using `get_group()`.
- Iterated through grouped data.
- Performed grouping using multiple columns.
- Built a stronger foundation for real-world data analysis.

---
## Day 13

### Topics Covered

- Concatenation
- Merge
- Join
- Inner Join
- Left Join
- Right Join
- Outer Join

### Key Learnings

- Combined multiple datasets.
- Understood different join strategies.
- Learned how relational data is handled in Pandas.
