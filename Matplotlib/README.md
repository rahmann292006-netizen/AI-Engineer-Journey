# Matplotlib Learning Journey 📊

This folder documents my hands-on journey with **Matplotlib**, Python's foundational visualization library.

Visualization is an important part of Data Science and Machine Learning because it helps reveal patterns, trends, distributions, and relationships inside data.

---

## 📋 Table of Contents

- [Topics Covered](#-topics-covered)
- [Real Dataset Practice](#-real-dataset-practice)
- [Files](#-files)
- [Datasets Used](#-datasets-used)
- [Visual Outputs](#-visual-outputs)
- [Learning Outcomes](#-learning-outcomes)
- [How to Run](#-how-to-run)
- [Key Takeaway](#-key-takeaway)

---

## 📚 Topics Covered

### Basic Visualization
| Plot Type | Description | Key Functions |
|-----------|-------------|---------------|
| Line Plot | Basic line charts for trends | `plt.plot()` |
| Bar Chart | Categorical comparisons | `plt.bar()`, `plt.barh()` |
| Histogram | Distribution of numerical data | `plt.hist()` |
| Scatter Plot | Relationship between two variables | `plt.scatter()` |
| Pie Chart | Proportional representation | `plt.pie()` |
| Box Plot | Statistical distribution summary | `plt.boxplot()` |

### Plot Customization
| Customization | Description |
|---------------|-------------|
| Figure Size | `plt.figure(figsize=(w,h))` |
| Colors | Named colors, hex codes, RGB tuples |
| Line Styles | Solid, dashed, dotted, dash-dot |
| Markers | Circle, square, triangle, star, etc. |
| Titles & Labels | `plt.title()`, `plt.xlabel()`, `plt.ylabel()` |
| Legends | `plt.legend()` for multi-series plots |
| Grid | `plt.grid(True)` for readability |
| Axis Limits | `plt.xlim()`, `plt.ylim()` |
| Annotations | `plt.annotate()` for highlighting points |
| Styles | `plt.style.use()` for themes |

### Advanced Visualization
| Topic | Description |
|-------|-------------|
| Subplots | Multiple plots in one figure via `plt.subplot()` or `plt.subplots()` |
| Multiple Plots | Overlaying multiple series on same axes |
| Figure & Axes | Object-oriented interface (`fig, ax = plt.subplots()`) |
| Saving Figures | `plt.savefig()` with formats (PNG, PDF, SVG) |
| Real-world Datasets | IPL cricket data, Car Price dataset |

---

## 🧪 Real Dataset Practice

Applied Matplotlib to **two real-world datasets**:

### 1. IPL Dataset (Cricket Deliveries)
- **Source**: `deliveries.csv` (IPL ball-by-ball data)
- **Practiced**:
  - Bar charts: Top 10 batters by runs
  - Histograms: Distribution of runs per ball
  - Scatter plots: Runs vs ball index
  - Pie charts: Top 5 batter contribution
  - Subplots: Combined bar + histogram view
  - Figure customization: Rotation, grid, tight layout
  - Saving visualizations: `top10_batters.png`

### 2. Car Price Dataset
- **Source**: `car_price.csv` (automobile specifications & prices)
- **Practiced**:
  - Histogram: Car price distribution
  - Bar chart: Fuel type distribution
  - Pie chart: Transmission/drive wheel distribution
  - Scatter plot: Year vs Price
  - Bar chart: Top 10 most expensive cars

This practice bridged the gap between learning plotting syntax and actually using visualization for data analysis.

---

## 📂 Files

| File | Description |
|------|-------------|
| `matplotlib.ipynb` | Complete hands-on Matplotlib practice notebook (Days 21-24) |
| `notes.md` | Daily learning notes with concepts, code snippets, and reflections |
| `car_price.csv` | Car price dataset for visualization practice |
| `deliveries.csv` | IPL deliveries dataset (ball-by-ball data) |
| `custom_plot.png` | Saved example of customized line plot |
| `top10_batters.png` | Saved visualization of top 10 IPL batters |

---

## 📊 Datasets Used

| Dataset | Rows | Columns | Description |
|---------|------|---------|-------------|
| IPL Deliveries | ~180k | 20+ | Ball-by-ball data from Indian Premier League matches |
| Car Price | 205 | 26 | Automobile specifications with price as target variable |

---

## 🖼️ Visual Outputs

| Output | Description |
|--------|-------------|
| `custom_plot.png` | Customized line plot demonstrating figure size, colors, line styles, markers, annotations, grid, legend, axis limits, and saving |
| `top10_batters.png` | Bar chart of top 10 IPL batters with rotated labels, grid, and tight layout |

---

## 🎯 Learning Outcomes

After completing this journey, I can:

- ✅ Create different types of visualizations (line, bar, histogram, scatter, pie, box)
- ✅ Customize plots for better readability and professional appearance
- ✅ Visualize distributions and relationships in data
- ✅ Compare categories using bar charts and pie charts
- ✅ Create multiple plots using subplots (both `plt.subplot()` and `plt.subplots()`)
- ✅ Save visualizations in various formats (PNG, PDF, SVG)
- ✅ Use visualization as part of an Exploratory Data Analysis (EDA) workflow
- ✅ Combine Pandas data manipulation with Matplotlib visualization
- ✅ Handle real-world datasets with column name variations
- ✅ Apply object-oriented Matplotlib interface for complex layouts

---

## 🚀 How to Run

### Prerequisites
```bash
pip install matplotlib pandas numpy
```

### Run the Notebook
```bash
jupyter notebook matplotlib.ipynb
```

### Or run individual cells in VS Code
Open `matplotlib.ipynb` and execute cells sequentially.

### View Saved Figures
- `custom_plot.png` - Customized line plot example
- `top10_batters.png` - Top 10 IPL batters bar chart

---

## 🔥 Key Takeaway

> **Visualization is not just about making graphs look good.**  
> **It is about answering questions from data.**

### The Data-to-Decision Flow

```text
Dataset
   ↓
Visualization
   ↓
Pattern
   ↓
Insight
   ↓
Decision
```

---

## 📈 Progress Tracking

| Day | Topic | Status |
|-----|-------|--------|
| 21 | Basic Plots (Line, Bar, Histogram, Scatter, Pie, Box) | ✅ |
| 22 | Plot Customization (Colors, Styles, Markers, Annotations, Saving) | ✅ |
| 23 | Real Dataset Visualization (IPL + Car Price) | ✅ |
| 24 | Matplotlib Milestone & Summary | ✅ |

---

## 🔗 Next Steps

- [ ] **Seaborn** - Statistical visualization built on Matplotlib
- [ ] **Exploratory Data Analysis (EDA)** - Comprehensive data investigation
- [ ] **Plotly** - Interactive visualizations
- [ ] **Dashboarding** - Streamlit/Dash for data apps

---

*Part of the [AI Engineer Journey](../README.md) — Day 21 to 24 of 84*