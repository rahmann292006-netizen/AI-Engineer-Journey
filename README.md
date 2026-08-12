# 🚀 AI Engineer Journey — 84 Days

> **Learning things the hard way.**

This repository documents my 84-day journey toward becoming an AI Engineer.

Instead of only watching tutorials, I'm learning by **practicing, building, breaking things, and documenting what I learn every day.**

---

## 📊 Phase 1 — Python → Data Analysis → EDA

The first phase of my journey was focused on building a strong foundation in Python and learning how to actually work with data.

I started with NumPy and Pandas, then moved into visualization and Exploratory Data Analysis.

---

# 🔢 NumPy

### What I practiced

- NumPy arrays
- Array creation
- Indexing & slicing
- Boolean indexing
- Array operations
- Reshaping
- Stacking
- Splitting
- Basic numerical operations

### Goal

Build enough NumPy understanding to work comfortably with numerical datasets before moving deeper into Data Science.

---

# 🐼 Pandas

After NumPy, I moved into Pandas and started working with DataFrames.

### Topics practiced

- DataFrame basics
- `read_csv()`
- `head()` / `tail()`
- Selecting rows and columns
- `loc`
- `iloc`
- Missing values
- Sorting
- Column operations
- Merge
- Join
- Concat
- Basic data manipulation

This was the point where working with real datasets started becoming much easier.

---

# 📊 Exploratory Data Analysis

After learning the basics of Pandas and visualization, I started exploring datasets instead of just manipulating them.

For EDA practice, I mainly worked with the **Iris dataset**.

The goal wasn't just to create graphs.

It was to learn how to ask questions about data and find patterns.

---

## 📅 Day 26 — Univariate Analysis

Started exploring individual variables.

### Practiced

- Distribution analysis
- Histograms
- Feature statistics
- Understanding individual feature behaviour
- Basic visualization

### Key takeaway

Before comparing variables, understand each variable individually.

---

## 📅 Day 27 — Bivariate Analysis

Moved from one variable to relationships between two variables.

### Practiced

- Scatter plots
- Feature relationships
- Comparing numerical variables
- Species-wise comparisons
- Understanding trends between features

This helped me move from:

**"What does this feature look like?"**

to:

**"How does this feature relate to another one?"**

---

## 📅 Day 28 — Multivariate Analysis

Went beyond two variables and started looking at multiple features together.

### Practiced

- Correlation analysis
- Feature relationships
- Species-wise comparisons
- Multivariate visualization
- Understanding relationships across multiple variables

### Key takeaway

Real datasets rarely tell their story through one variable at a time.

---

## 📅 Day 29 — Outlier Detection

Next, I started looking for unusual values in the dataset.

### Practiced

- Quartiles
- IQR
- Lower and upper bounds
- Potential outlier detection
- Boxplots
- Feature-wise analysis

### IQR Method

```text
IQR = Q3 - Q1

Lower Bound = Q1 - 1.5 × IQR

Upper Bound = Q3 + 1.5 × IQR
