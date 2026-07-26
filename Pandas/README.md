# 📊 Pandas Journey

This folder documents my hands-on learning journey with **Pandas**, the most widely used Python library for data manipulation and analysis. It contains practice notebooks, notes, and examples covering essential Pandas concepts required for Data Science and Machine Learning.

---

## 📚 Topics Covered

- Introduction to Pandas
- Series
- DataFrames
- Reading CSV Files
- Data Inspection
- Selecting Rows & Columns
- Indexing (`loc[]` & `iloc[]`)
- Filtering Data
- Handling Missing Values
- Sorting & Ranking
- GroupBy Operations
- Aggregation
- Merge & Join
- Concatenation
- Pivot Tables
- Apply Function
- String Operations
- Date & Time Operations

---

## 📂 Files

- **pandas_basics.ipynb** → Hands-on practice and examples
- **notes.md** → Concept notes and explanations

---

## 🎯 Learning Outcome

After completing this journey, I will be able to:

- Load and inspect datasets efficiently.
- Clean and preprocess data.
- Filter and transform data.
- Perform grouping and aggregation.
- Merge multiple datasets.
- Prepare datasets for Machine Learning.
- Analyze real-world datasets using Pandas.

---

## 🚀 Status

🟡 In Progress

---

## ⭐ Why This Repository?

This repository is part of my **AI Engineer Learning Journey**, where I document concepts, hands-on practice, and daily progress while building a strong foundation in Machine Learning and Generative AI.

---

## 🔗 Next Topics

- Matplotlib
- Seaborn
- Scikit-learn
- Machine Learning

# Day 17 - DataFrame Selection, Indexing & Filtering

## Selecting Columns

Select a single column:

```python
df["batter"]
```

Select multiple columns:

```python
df[["batter", "batsman_runs"]]
```

---

## Selecting Rows

First 5 rows

```python
df[0:5]
```

Rows 10 to 20

```python
df[10:21]
```

---

## loc[]

Used for label-based indexing.

Examples:

```python
df.loc[0]
df.loc[0:5]
```

Select rows and columns:

```python
df.loc[0:5, ["batter", "batsman_runs"]]
```

---

## iloc[]

Used for integer position indexing.

```python
df.iloc[0]
```

```python
df.iloc[0:5]
```

```python
df.iloc[0:5, 0:2]
```

---

## Filtering Data

Runs greater than 50

```python
df[df["batsman_runs"] > 50]
```

Runs greater than 20

```python
df[df["batsman_runs"] > 20]
```

---

## Multiple Conditions

```python
df[(df["batsman_runs"] > 20) & (df["ID"] > 100)]
```

---

# Key Learnings

- Selecting rows
- Selecting columns
- loc[]
- iloc[]
- Boolean Filtering
- Multiple Conditions

---

# Day 17 Status

✅ DataFrame Selection & Filtering Completed

Next:
- Sorting
- Renaming Columns
- Value Counts
- Unique Values
