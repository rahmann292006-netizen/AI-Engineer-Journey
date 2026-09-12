# 🚀 AI Engineer Journey — 84 Days

> **Learning things the hard way.**

This repository documents my 84-day journey toward becoming an AI Engineer.

Instead of only watching tutorials, I'm learning by **practicing, building, breaking things, and documenting what I learn every day.**

---

## 📊 Phase 1 — Python → Data Analysis → EDA

The first phase of my journey was focused on building a strong foundation in Python and learning how to actually work with data.

I started with NumPy and Pandas, then moved into visualization and Exploratory Data Analysis.

---

## 🔢 NumPy

### What I Practiced

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

## 🐼 Pandas

After NumPy, I moved into Pandas and started working with DataFrames.

### Topics Practiced

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

## 📊 Exploratory Data Analysis

After learning the basics of Pandas and visualization, I started exploring datasets instead of just manipulating them.

For EDA practice, I mainly worked with the **Iris dataset**.

The goal wasn't just to create graphs — it was to learn how to ask questions about data and find patterns.

---

### 📅 Day 26 — Univariate Analysis

Started exploring individual variables.

#### Practiced

- Distribution analysis
- Histograms
- Feature statistics
- Understanding individual feature behaviour
- Basic visualization

#### Key Takeaway

> Before comparing variables, understand each variable individually.

---

### 📅 Day 27 — Bivariate Analysis

Moved from one variable to relationships between two variables.

#### Practiced

- Scatter plots
- Feature relationships
- Comparing numerical variables
- Species-wise comparisons
- Understanding trends between features

This helped me move from:

> **"What does this feature look like?"**

to:

> **"How does this feature relate to another one?"**

---

### 📅 Day 28 — Multivariate Analysis

Went beyond two variables and started looking at multiple features together.

#### Practiced

- Correlation analysis
- Feature relationships
- Species-wise comparisons
- Multivariate visualization
- Understanding relationships across multiple variables

#### Key Takeaway

> Real datasets rarely tell their story through one variable at a time.

---

### 📅 Day 29 — Outlier Detection

Next, I started looking for unusual values in the dataset.

#### Practiced

- Quartiles
- IQR
- Lower and upper bounds
- Potential outlier detection
- Boxplots
- Feature-wise analysis

#### IQR Method

```text
IQR = Q3 - Q1

Lower Bound = Q1 - 1.5 × IQR

Upper Bound = Q3 + 1.5 × IQR
```

#### Key Takeaway

> An outlier isn't automatically bad data. It could be a measurement error, data-entry mistake, rare observation, or a genuine unusual case. The important part is understanding the data before deciding what to do with it.

---

### 📅 Day 30 — Data Cleaning & EDA Wrap-up

Day 30 was about putting everything together.

#### Practiced

- Dataset inspection
- Missing-value checks
- Duplicate detection
- Removing duplicates
- Data-type inspection
- Descriptive statistics
- Outlier detection
- Correlation analysis
- Species-wise comparisons
- Final visualization
- Building a complete EDA workflow

#### Final EDA Workflow

```text
Raw Dataset
     ↓
Inspect
     ↓
Clean
     ↓
Understand distributions
     ↓
Find relationships
     ↓
Detect outliers
     ↓
Analyze correlations
     ↓
Compare groups
     ↓
Extract insights
```

#### Biggest Lesson

> EDA isn't about making random charts until something looks interesting. It's about asking the right questions about the data.

---

## 🧠 What I Learned from the EDA Phase

The biggest shift for me wasn't learning another Pandas function — it was learning to look at a dataset and ask:

- What is actually inside this data?
- Are there missing values?
- Are there duplicates?
- What does each feature look like?
- How are features related?
- Are there unusual observations?
- Do different groups behave differently?
- What can I actually conclude from the data?

This phase helped me move from simply handling data to **actually thinking about data**.

---

## 🛠️ Tools & Technologies

### 🐍 Core Language
| Tool | Purpose | Version Used |
|------|---------|--------------|
| [![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/) | Primary programming language for all data analysis and ML tasks | 3.10+ |

### 📚 Data Science Libraries
| Tool | Purpose | Key Features Used |
|------|---------|-------------------|
| [![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/) | Numerical computing foundation | Arrays, broadcasting, vectorized operations, linear algebra |
| [![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/) | Data manipulation & analysis | DataFrames, I/O, grouping, merging, time series |
| [![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=matplotlib&logoColor=white)](https://matplotlib.org/) | Static visualizations | Plots, subplots, customization, annotations |
| [![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://seaborn.pydata.org/) | Statistical visualizations | Pair plots, heatmaps, categorical plots, distributions |
| [![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/) | Machine learning | Preprocessing, model selection, metrics, pipelines |

### 🛠️ Development Environment
| Tool | Purpose |
|------|---------|
| [![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/) | Interactive notebooks for exploration & documentation |
| [![VS Code](https://img.shields.io/badge/VS_Code-007ACC?style=for-the-badge&logo=visual-studio-code&logoColor=white)](https://code.visualstudio.com/) | Primary IDE with Python/Jupyter extensions |
| [![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)](https://git-scm.com/) | Version control & collaboration |

### 📦 Package Management
| Tool | Purpose |
|------|---------|
| [![pip](https://img.shields.io/badge/pip-3776AB?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/) | Python package installer |
| [![conda](https://img.shields.io/badge/conda-44A833?style=for-the-badge&logo=anaconda&logoColor=white)](https://docs.conda.io/) | Environment & package management |

---

## 📂 Repository Structure

```
AI-Engineer-Journey/
├── Numpy/                 # NumPy hands-on practice
├── Pandas/                # Pandas basics & data manipulation
├── Matplotlib/            # Visualization practice & projects
├── Machine Learning/      # ML workflows, FastAPI, GenAI
├── *.csv                  # Datasets used for practice
├── *.ipynb                # Jupyter notebooks for each phase
└── README.md              # This file
```

---

## 📈 Progress Tracking

| Phase | Topic | Days | Status |
|-------|-------|------|--------|
| 1 | Python → Data Analysis → EDA | 1-30 | ✅ Complete |
| 2 | Machine Learning Fundamentals | 31-50 | 🔄 In Progress |
| 3 | Deep Learning | 51-65 | ⏳ Planned |
| 4 | MLOps & Deployment | 66-75 | ⏳ Planned |
| 5 | GenAI & LLMs | 76-84 | ⏳ Planned |

---

## 🤝 Connect

- **GitHub**: [@rahmann292006-netizen](https://github.com/rahmann292006-netizen)
- **Repository**: [AI-Engineer-Journey](https://github.com/rahmann292006-netizen/AI-Engineer-Journey)

---

> *"The best way to learn is to build. The best way to build is to start."*