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
