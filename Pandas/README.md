# Pandas Learning Journey 🐼

This folder documents my hands-on learning journey with **Pandas**, focusing on data manipulation, cleaning, transformation and analysis.

Pandas is one of the most important tools in my AI Engineer journey because real-world Machine Learning starts with understanding and preparing data.

---

## 📚 Topics Covered

### 1. Pandas Fundamentals

| Concept | Methods & Attributes |
|---------|---------------------|
| **Series** | `pd.Series()`, indexing, vectorized operations |
| **DataFrame** | `pd.DataFrame()`, construction from dict/list/arrays |
| **Reading Data** | `read_csv()`, `read_excel()`, `read_json()`, `read_sql()` |
| **Inspection** | `head()`, `tail()`, `info()`, `describe()`, `shape`, `columns`, `dtypes`, `sample()` |
| **Basic Stats** | `mean()`, `median()`, `std()`, `min()`, `max()`, `value_counts()` |

### 2. Data Selection & Indexing

| Technique | Use Case |
|-----------|----------|
| Column selection | `df['col']`, `df[['col1', 'col2']]` |
| Row selection by label | `loc[row_label]`, `loc[row_label, col_label]` |
| Row selection by position | `iloc[row_pos]`, `iloc[row_pos, col_pos]` |
| Boolean indexing | `df[df['col'] > value]` |
| Multiple conditions | `df[(df['a'] > 1) & (df['b'] == 'x')]` |
| Query method | `df.query('col > 5 and col2 == "A"')` |
| `isin()` | Filter by list of values |
| `between()` | Range filtering |
| `where()` / `mask()` | Conditional replacement |

### 3. Data Cleaning

| Task | Methods |
|------|---------|
| **Detect missing** | `isnull()`, `notnull()`, `isna()`, `sum()` |
| **Fill missing** | `fillna(value)`, `fillna(method='ffill'/'bfill')`, `interpolate()` |
| **Drop missing** | `dropna(axis=0/1, how='any'/'all', thresh=n)` |
| **Duplicates** | `duplicated()`, `drop_duplicates(subset=['col'])` |
| **Type conversion** | `astype()`, `to_numeric()`, `to_datetime()`, `convert_dtypes()` |
| **String cleaning** | `str.strip()`, `str.lower()`, `str.replace()`, `str.extract()` |

### 4. Data Transformation

| Operation | Methods |
|-----------|---------|
| **Create columns** | `df['new'] = expression`, `assign()` |
| **Update columns** | `df['col'] = new_values`, `map()`, `apply()` |
| **Sort** | `sort_values()`, `sort_index()` |
| **Rename** | `rename()`, `columns = [...]` |
| **Apply functions** | `apply()`, `applymap()`, `pipe()` |
| **Binning** | `cut()`, `qcut()` |
| **Pivot/Reshape** | `pivot()`, `pivot_table()`, `melt()`, `stack()`, `unstack()` |

### 5. Combining Data

| Method | Description |
|--------|-------------|
| `concat()` | Stack DataFrames vertically/horizontally |
| `merge()` | SQL-style joins (inner, left, right, outer) |
| `join()` | Join on index |
| `compare()` | Compare two DataFrames (v1.1+) |

**Join Types:** Inner, Left, Right, Outer, Cross

### 6. Grouping & Aggregation

| Operation | Syntax |
|-----------|--------|
| Single aggregation | `df.groupby('col')['target'].mean()` |
| Multiple aggregations | `df.groupby('col').agg({'a': 'sum', 'b': ['mean', 'std']})` |
| Named aggregations | `df.groupby('col').agg(total=('a', 'sum'), avg=('b', 'mean'))` |
| Filter groups | `filter()`, `transform()` |
| Iterate groups | `for name, group in df.groupby('col'):` |
| Pivot tables | `pd.pivot_table()` |

---

## 📂 Files

| File | Description | Link |
|------|-------------|------|
| `pandas_basics.ipynb` | Hands-on Pandas practice with exercises | [Open Notebook](pandas_basics.ipynb) |
| `notes.md` | Detailed Pandas concepts, syntax reference & examples | [View Notes](notes.md) |
| `Projects/` | Practice projects using real-world datasets | [Explore Projects](Projects/) |

---

## 🎯 Learning Outcomes

After completing this journey, I can:

### Core Skills
- [x] Create and manipulate **Series** and **DataFrames** from various sources
- [x] Select, filter, and index data efficiently using `loc`, `iloc`, and boolean masks
- [x] Handle missing data: detect, impute, and remove strategically
- [x] Identify and remove duplicate records with precision
- [x] Convert data types safely using `astype`, `to_numeric`, `to_datetime`

### Transformation & Analysis
- [x] Engineer new features through column operations and `apply()`
- [x] Sort, rank, and bin data for analysis
- [x] Reshape data with `pivot`, `melt`, and `pivot_table`
- [x] Combine datasets using `concat`, `merge`, and `join` with appropriate join types

### Aggregation & Summarization
- [x] Perform group-wise analysis with `groupby()`
- [x] Apply multiple aggregation functions simultaneously
- [x] Use named aggregations for readable output
- [x] Leverage `transform()` and `filter()` for group-aware operations

### ML-Ready Data Preparation
- [x] Prepare clean, structured datasets for **Exploratory Data Analysis**
- [x] Prepare feature matrices and target vectors for **Machine Learning**
- [x] Export processed data in multiple formats (CSV, Parquet, Excel, SQL)

---

## 💡 Best Practices & Tips

| Category | Recommendation |
|----------|----------------|
| **Performance** | Use vectorized operations over `apply()`/`iterrows()` |
| **Memory** | Specify `dtype` in `read_csv()`, use `category` for low-cardinality columns |
| **Chaining** | Use method chaining with `pipe()` for readable pipelines |
| **Copy vs View** | Be explicit with `.copy()` to avoid `SettingWithCopyWarning` |
| **Indexing** | Prefer `loc`/`iloc` over chained indexing `df[col][row]` |
| **Dates** | Parse dates during `read_csv()` with `parse_dates=` |

---

## 📖 Resources & References

- [Official Pandas Documentation](https://pandas.pydata.org/docs/)
- [Pandas API Reference](https://pandas.pydata.org/docs/reference/index.html)
- [10 Minutes to Pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
- [Pandas Cookbook](https://pandas.pydata.org/docs/user_guide/cookbook.html)
- [Effective Pandas (Tom Augspurger)](https://effective-pandas.com/)

---

## 🔥 Key Takeaway

Machine Learning doesn't start with the model.

It starts with:

```text
Raw Data
   ↓
Understand
   ↓
Clean
   ↓
Transform
   ↓
Analyze
   ↓
Machine Learning
```

---

## 📈 Next Steps

Continue the journey with:

1. **Exploratory Data Analysis (EDA)** → `../EDA/`
2. **Data Visualization** → `../Matplotlib/`
3. **Machine Learning Workflows** → `../Machine Learning/`
4. **Real-world Projects** → `Projects/`