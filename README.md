# AI Engineer Journey

This repository contains learning notes, notebooks, exercises, and small projects from my path through Python, data analysis, visualization, exploratory data analysis (EDA), and machine learning. The journey is organized as an 84-day learning program. Its completion and several later milestones are reported by me; the repository itself contains day-specific records through Day 45, with gaps noted in the tracker below.

## Table of contents

- [84-day roadmap overview](#84-day-roadmap-overview)
- [Progress summary](#progress-summary)
- [Learning phases](#learning-phases)
- [Day 1–Day 84 tracker](#day-1day-84-tracker)
- [Exercises and implementations](#exercises-and-implementations)
- [Projects](#projects)
- [Technologies and tools](#technologies-and-tools)
- [Repository structure](#repository-structure)
- [Setup and running notebooks](#setup-and-running-notebooks)
- [Milestones and lessons](#milestones-and-lessons)
- [Current status and next steps](#current-status-and-next-steps)
- [Future updates](#future-updates)
- [Connect](#connect)

## 84-day roadmap overview

The repository records a progression from NumPy and Pandas through visualization, EDA, and machine-learning workflows. Day-level notes and Git history support the topics shown for Days 2–45. The available repository records do not describe Day 1 or provide individual learning records for Days 46–84.

| Learning phase | Days represented in the repository | Evidence-backed focus |
| --- | --- | --- |
| NumPy practice | 2–6 | Arrays, indexing, broadcasting, vectorization, manipulation, and linear algebra |
| Pandas practice | 7–20 | DataFrames, cleaning, transformation, grouping, combining, dates, and summaries |
| Visualization | 21–24 | Matplotlib fundamentals, styling, and dataset visualizations |
| Exploratory data analysis | 25–30 | Car-price and Iris data exploration, analysis, outliers, and cleaning |
| Machine learning | 31–45 | Classification, regression, evaluation, validation, regularization, and model selection |
| Day-level records unavailable | 1, 46–84 | No individual topic or exercise verified in the current repository |

The historical roadmap assigned later days to deep learning, deployment/MLOps, and generative AI, but its day ranges conflict with the Day 45 note, which points to deep learning on Day 46. Those entries are retained only as historical planning context, not as verified daily progress.

## Progress summary

The 84-day journey is reported complete by the repository owner. The auditable day-by-day record is narrower:

| Status | Days | Count | Basis |
| --- | --- | ---: | --- |
| Completed | 2–45 | 44 | Day-specific commits and current notebooks or notes |
| In Progress | — | 0 | No current day is recorded as in progress |
| Planned | — | 0 | Older roadmap plans are historical and conflict with later journey-level completion |
| Undocumented | 1, 46–84 | 40 | No individual day record was found in the current files or Git history |

An earlier root README summary described Days 1–45 as complete, but no specific Day 1 topic or exercise was found. Day 1 is therefore shown as undocumented in the day-level tracker. The owner-reported completion of the full journey is kept separate from repository-verifiable daily records.

## Learning phases

### NumPy and Pandas foundations — Days 2–20

Hands-on NumPy work covered array creation and properties, indexing, Boolean masking, broadcasting, vectorization, stacking, splitting, sorting, searching, and linear algebra. Pandas work moved into Series and DataFrames, selection and filtering, cleaning, transformations, grouping and aggregation, combining data, pivot tables, dates, and descriptive statistics.

References: [NumPy notebook](Numpy/numpy_hands_on.ipynb), [NumPy notes](Numpy/notes.md), [Pandas notebook](Pandas/pandas_basics.ipynb), and [Pandas notes](Pandas/notes.md).

### Visualization and EDA — Days 21–30

The repository documents Matplotlib fundamentals and customization, IPL data visualizations, and EDA exercises. EDA topics include univariate, bivariate, and multivariate analysis; outlier detection; and a cleaning workflow using dataset practice.

References: [Matplotlib notebook](Matplotlib/matplotlib.ipynb), [Matplotlib notes](Matplotlib/notes.md), [EDA guide](EDA/README.md), and [EDA notebook](EDA/eda_final_workout.ipynb).

### Machine-learning fundamentals — Days 31–45

The machine-learning record covers a workflow and classification examples, feature scaling, K-nearest neighbors, pipeline and parameter search, linear regression, residual analysis, model complexity, cross-validation, Ridge and Lasso, and model selection.

References: [Machine Learning progress notes](Machine%20Learning/Readme.md) and [workflow notebook](Machine%20Learning/Ml_workflow.ipynb).

## Day 1–Day 84 tracker

“Completed” means a day-specific repository record was found. For undocumented days, no individual exercise or outcome is inferred from a phase title or the overall journey completion statement.

| Day | Topic | Practical work or record | Status | Evidence |
| --- | --- | --- | --- | --- |
| 1 | No day-specific topic recorded | No day-specific exercise found; an earlier README only reported Days 1–45 complete | Undocumented | — |
| 2 | NumPy setup and array practice | Repository organization and initial NumPy practice | Completed | [NumPy notebook](Numpy/numpy_hands_on.ipynb) |
| 3 | NumPy indexing and Boolean masks | Practice indexing, slicing, and Boolean selection | Completed | [NumPy notebook](Numpy/numpy_hands_on.ipynb) |
| 4 | Broadcasting and vectorization | Apply NumPy broadcasting and vectorized operations | Completed | [NumPy notebook](Numpy/numpy_hands_on.ipynb) |
| 5 | Array manipulation and search | Practice stacking, splitting, sorting, and searching | Completed | [NumPy notebook](Numpy/numpy_hands_on.ipynb) |
| 6 | NumPy linear algebra | Practice linear algebra operations | Completed | [NumPy notebook](Numpy/numpy_hands_on.ipynb) |
| 7 | Pandas and DataFrame basics | Create and inspect Series and DataFrames | Completed | [Pandas notebook](Pandas/pandas_basics.ipynb) |
| 8 | Indexing, selection, and filtering | Select rows and columns and filter data | Completed | [Pandas notebook](Pandas/pandas_basics.ipynb) |
| 9 | Missing values and data cleaning | Inspect and handle missing and duplicate values | Completed | [Pandas notes](Pandas/notes.md) |
| 10 | Data transformation | Practice `apply`, `map`, `replace`, string operations, and new columns | Completed | [Pandas notebook](Pandas/pandas_basics.ipynb) |
| 11 | GroupBy fundamentals | Group data and calculate aggregations | Completed | [Pandas notebook](Pandas/pandas_basics.ipynb) |
| 12 | Advanced GroupBy | Practice custom aggregations and grouped operations | Completed | [Pandas notebook](Pandas/pandas_basics.ipynb) |
| 13 | Combining DataFrames | Practice merge, join, and concatenation | Completed | [Pandas notebook](Pandas/pandas_basics.ipynb) |
| 14 | Pivot tables, crosstab, and MultiIndex | Summarize and reshape grouped data | Completed | [Pandas notebook](Pandas/pandas_basics.ipynb) |
| 15 | GroupBy, transform, and filter | Compare aggregation, transformation, and group filtering | Completed | [Pandas notes](Pandas/notes.md) |
| 16 | DateTime handling | Parse dates and derive or compare date components | Completed | [Pandas notebook](Pandas/pandas_basics.ipynb) |
| 17 | DataFrame selection and filtering | Practice label/position selection and multiple conditions | Completed | [Pandas notebook](Pandas/pandas_basics.ipynb) |
| 18 | Sorting, unique values, and descriptive statistics | Inspect distinct values, counts, and summary measures | Completed | [Pandas notebook](Pandas/pandas_basics.ipynb) |
| 19 | Sorting and descriptive statistics | Continue Pandas sorting and statistical summaries | Completed | [Pandas notes](Pandas/notes.md) |
| 20 | Pandas revision | Review Pandas fundamentals and data-analysis practice | Completed | [Pandas notebook](Pandas/pandas_basics.ipynb) |
| 21 | Matplotlib fundamentals | Begin plotting and data visualization practice | Completed | [Matplotlib notebook](Matplotlib/matplotlib.ipynb) |
| 22 | Plot customization | Practice Matplotlib styling and customization | Completed | [Matplotlib notebook](Matplotlib/matplotlib.ipynb) |
| 23 | IPL data visualization | Visualize IPL delivery data; a top-batters plot is included | Completed | [Matplotlib notebook](Matplotlib/matplotlib.ipynb) |
| 24 | Matplotlib with real-world data | Practice visualization with real-world data | Completed | [Matplotlib notebook](Matplotlib/matplotlib.ipynb) |
| 25 | Introduction to EDA | Begin EDA using a car-price dataset | Completed | [Car-price data](car_price.csv), [EDA guide](EDA/README.md) |
| 26 | Univariate analysis | Explore individual variables in the car-price dataset | Completed | [Matplotlib notes](Matplotlib/notes.md), [car-price data](Matplotlib/car_price.csv) |
| 27 | Bivariate analysis | Compare pairs of Iris features and their relationships | Completed | [EDA notebook](EDA/eda_final_workout.ipynb) |
| 28 | Multivariate analysis | Explore correlations and multiple Iris features | Completed | [EDA notebook](EDA/eda_final_workout.ipynb) |
| 29 | Outlier detection | Practice quartiles, IQR bounds, and boxplot analysis | Completed | [EDA guide](EDA/README.md) |
| 30 | Data cleaning and EDA workflow | Inspect, clean, analyze, and visualize the Iris dataset | Completed | [EDA notebook](EDA/eda_final_workout.ipynb) |
| 31 | Machine-learning workflow | Set up a basic workflow, split data, and fit a model | Completed | [ML notebook](Machine%20Learning/Ml_workflow.ipynb) |
| 32 | Logistic regression | Train a first classification model | Completed | [ML notebook](Machine%20Learning/Ml_workflow.ipynb) |
| 33 | Classification evaluation | Practice model evaluation metrics | Completed | [ML notebook](Machine%20Learning/Ml_workflow.ipynb) |
| 34 | Feature scaling and preprocessing | Apply scaling in a preprocessing workflow | Completed | [ML notebook](Machine%20Learning/Ml_workflow.ipynb) |
| 35 | K-nearest neighbors | Train a KNN classifier | Completed | [ML notebook](Machine%20Learning/Ml_workflow.ipynb) |
| 36 | Parameter tuning and pipelines | Practice GridSearchCV with a pipeline | Completed | [ML notebook](Machine%20Learning/Ml_workflow.ipynb) |
| 37 | Single-feature linear regression | Fit and evaluate a linear regression model | Completed | [ML notebook](Machine%20Learning/Ml_workflow.ipynb) |
| 38 | Multiple linear regression | Fit a regression model with multiple features | Completed | [ML notebook](Machine%20Learning/Ml_workflow.ipynb) |
| 39 | Regression evaluation and residual analysis | Review regression metrics and residuals | Completed | [ML notebook](Machine%20Learning/Ml_workflow.ipynb) |
| 40 | Simple linear regression from scratch | Implement gradient descent and compare with scikit-learn | Completed | [ML notes](Machine%20Learning/Readme.md) |
| 41 | Residual diagnostics | Explore residual patterns and diagnostic methods | Completed | [ML notes](Machine%20Learning/Readme.md) |
| 42 | Overfitting and underfitting | Compare model complexity and generalization | Completed | [ML notebook](Machine%20Learning/Ml_workflow.ipynb) |
| 43 | Cross-validation | Practice K-fold validation and compare model scores | Completed | [ML notebook](Machine%20Learning/Ml_workflow.ipynb) |
| 44 | Ridge and Lasso regularization | Compare L1 and L2 regularization | Completed | [ML notebook](Machine%20Learning/Ml_workflow.ipynb) |
| 45 | Model selection | Compare candidate models and validation approaches | Completed | [ML notebook](Machine%20Learning/Ml_workflow.ipynb), [ML notes](Machine%20Learning/Readme.md) |
| 46 | No day-specific topic or exercise verified | Earlier Day 45 note proposed deep learning as the next topic; no completion record found | Undocumented | [Historical roadmap note](Machine%20Learning/Readme.md) |
| 47 | No day-specific record | No verified topic, exercise, or implementation found | Undocumented | — |
| 48 | No day-specific record | No verified topic, exercise, or implementation found | Undocumented | — |
| 49 | No day-specific record | No verified topic, exercise, or implementation found | Undocumented | — |
| 50 | No day-specific record | No verified topic, exercise, or implementation found | Undocumented | — |
| 51 | No day-specific record | No verified topic, exercise, or implementation found | Undocumented | — |
| 52 | No day-specific record | No verified topic, exercise, or implementation found | Undocumented | — |
| 53 | No day-specific record | No verified topic, exercise, or implementation found | Undocumented | — |
| 54 | No day-specific record | No verified topic, exercise, or implementation found | Undocumented | — |
| 55 | No day-specific record | No verified topic, exercise, or implementation found | Undocumented | — |
| 56 | No day-specific record | No verified topic, exercise, or implementation found | Undocumented | — |
| 57 | No day-specific record | No verified topic, exercise, or implementation found | Undocumented | — |
| 58 | No day-specific record | No verified topic, exercise, or implementation found | Undocumented | — |
| 59 | No day-specific record | No verified topic, exercise, or implementation found | Undocumented | — |
| 60 | No day-specific record | No verified topic, exercise, or implementation found | Undocumented | — |
| 61 | No day-specific record | No verified topic, exercise, or implementation found | Undocumented | — |
| 62 | No day-specific record | No verified topic, exercise, or implementation found | Undocumented | — |
| 63 | No day-specific record | No verified topic, exercise, or implementation found | Undocumented | — |
| 64 | No day-specific record | No verified topic, exercise, or implementation found | Undocumented | — |
| 65 | No day-specific record | No verified topic, exercise, or implementation found | Undocumented | — |
| 66 | No day-specific record | No verified topic, exercise, or implementation found | Undocumented | — |
| 67 | No day-specific record | No verified topic, exercise, or implementation found | Undocumented | — |
| 68 | No day-specific record | No verified topic, exercise, or implementation found | Undocumented | — |
| 69 | No day-specific record | No verified topic, exercise, or implementation found | Undocumented | — |
| 70 | No day-specific record | No verified topic, exercise, or implementation found | Undocumented | — |
| 71 | No day-specific record | No verified topic, exercise, or implementation found | Undocumented | — |
| 72 | No day-specific record | No verified topic, exercise, or implementation found | Undocumented | — |
| 73 | No day-specific record | No verified topic, exercise, or implementation found | Undocumented | — |
| 74 | No day-specific record | No verified topic, exercise, or implementation found | Undocumented | — |
| 75 | No day-specific record | No verified topic, exercise, or implementation found | Undocumented | — |
| 76 | No day-specific record | No verified topic, exercise, or implementation found | Undocumented | — |
| 77 | No day-specific record | No verified topic, exercise, or implementation found | Undocumented | — |
| 78 | No day-specific record | No verified topic, exercise, or implementation found | Undocumented | — |
| 79 | No day-specific record | No verified topic, exercise, or implementation found | Undocumented | — |
| 80 | No day-specific record | No verified topic, exercise, or implementation found | Undocumented | — |
| 81 | No day-specific record | No verified topic, exercise, or implementation found | Undocumented | — |
| 82 | No day-specific record | No verified topic, exercise, or implementation found | Undocumented | — |
| 83 | No day-specific record | No verified topic, exercise, or implementation found | Undocumented | — |
| 84 | No day-specific record | No verified topic, exercise, or implementation found | Undocumented | — |

## Exercises and implementations

- NumPy exercises are recorded in [the NumPy notebook](Numpy/numpy_hands_on.ipynb), including array operations and linear algebra.
- Pandas exercises and output are in [the Pandas notebook](Pandas/pandas_basics.ipynb) and [saved output](Pandas/output.csv).
- Matplotlib practice is in [the visualization notebook](Matplotlib/matplotlib.ipynb), with saved figures in `Matplotlib/`.
- EDA practice is available in [the EDA notebook](EDA/eda_final_workout.ipynb); a broader workflow guide is in [EDA/README.md](EDA/README.md).
- Machine-learning exercises are in [the workflow notebook](Machine%20Learning/Ml_workflow.ipynb), with additional day notes in [Machine Learning/Readme.md](Machine%20Learning/Readme.md).
- Additional repository notebooks include [Netflix titles EDA](netflix-shows-and-movies-exploratory-analysis.ipynb) and [red-wine classification](red-wine-quality.ipynb).

## Projects

### In-repository project

- [Image Matrix Operations](Numpy/Projects/Image-Matrix-Operations/README.md) — a NumPy and Pillow mini-project. Its project documentation describes image flipping, cropping, rotation, brightness adjustment, grayscale conversion, and inversion. The notebook and sample image are in the same folder.

### Owner-reported milestones not stored in this repository

- **Storytelling-style portfolio:** an interactive experience presenting the learning journey. No source files or demo link are present here.
- **WeatherGPT:** a weather assistant project. Project details and a source link are not present here.
- **Smart India Hackathon (SIH):** attendance was reported; no result, award, or project outcome is claimed here.

## Technologies and tools

### Practiced in repository materials

| Area | Technologies and tools |
| --- | --- |
| Programming and notebooks | Python, Jupyter notebooks |
| Data and numerical work | NumPy, pandas |
| Visualization | Matplotlib; Seaborn appears in the red-wine notebook |
| Machine learning | scikit-learn |
| Image project | Pillow |
| Additional notebook | Plotly |
| Version control | Git |

This describes tools present in code or documentation; it is not a claim of mastery.

### Currently strengthening

- Python programming fundamentals
- Data science and machine-learning fundamentals

### Next learning areas

- Generative AI and Agentic AI
- Practical project development, testing, and deployment
- Data structures and algorithms (DSA)

## Repository structure

```text
AI-Engineer-Journey/
├── EDA/
│   ├── README.md
│   └── eda_final_workout.ipynb
├── Machine Learning/
│   ├── Ml_workflow.ipynb
│   ├── Readme.md
│   ├── actual_vs_predicted.png
│   └── residual_plot.png
├── Matplotlib/
│   ├── matplotlib.ipynb
│   ├── notes.md
│   ├── README.md
│   └── visualization data and figures
├── Numpy/
│   ├── numpy_hands_on.ipynb
│   ├── notes.md
│   ├── README.md
│   └── Projects/Image-Matrix-Operations/
├── Pandas/
│   ├── pandas_basics.ipynb
│   ├── notes.md
│   ├── README.md
│   └── output.csv
├── *.csv                         # Practice datasets
├── *.ipynb                       # Additional analysis notebooks
└── README.md
```

## Setup and running notebooks

The repository does not include a dependency lockfile or environment specification. The notebooks use common Python data-science packages; create an environment and install the packages needed by the notebook you plan to open.

```bash
python -m venv .venv
```

Activate the environment, then install Jupyter and the notebook dependencies. For the main learning notebooks:

```bash
python -m pip install jupyter numpy pandas matplotlib scikit-learn
```

Some notebooks also use `Pillow`, `seaborn`, `plotly`, or `termcolor`; install those if needed for the selected notebook:

```bash
python -m pip install pillow seaborn plotly termcolor
```

Start Jupyter from the repository root so relative dataset paths resolve:

```bash
python -m jupyter notebook
```

Then open a notebook such as `Numpy/numpy_hands_on.ipynb`, `EDA/eda_final_workout.ipynb`, or `Machine Learning/Ml_workflow.ipynb`. No standalone FastAPI or GenAI application entry point or setup instructions are present in the tracked repository.

## Milestones and lessons

The following milestones were reported by me; the portfolio and WeatherGPT source code are not stored in this repository:

- Completed the 84-day / 12-week learning journey.
- Built a storytelling-style interactive portfolio for the journey.
- Built WeatherGPT, a weather assistant project.
- Attended Smart India Hackathon (SIH).
- Continued strengthening Python, data science, and machine-learning fundamentals.
- My last social media post was on September 27. Posting consistency dropped while I focused on project building and implementation. The year was not specified.

The repository's learning notes emphasize inspecting and cleaning data before analysis, understanding relationships and outliers, and validating models with appropriate evaluation methods. These themes are reflected in the EDA and machine-learning notebooks.

## Current status and next steps

The 84-day journey is complete by owner report. The repository provides a detailed daily trail for Days 2–45; it does not contain enough day-specific evidence to reconstruct the rest of the journey or assign those days individual topics and exercises.

The next phase is to:

- Strengthen Python, data science, and machine-learning fundamentals.
- Learn Generative AI and Agentic AI in greater depth.
- Build practical projects and improve testing and deployment.
- Practice DSA.
- Prepare for internships and entry-level roles.

This is a learning plan, not a claim of mastery or job readiness.

## Future updates

Future updates can add dated daily records, project source links, and reproducible environment instructions as those materials become available. Day-level progress should be updated from those records rather than inferred from the roadmap.

## Connect

- **GitHub:** [@rahmann292006-netizen](https://github.com/rahmann292006-netizen)
- **Repository:** [AI-Engineer-Journey](https://github.com/rahmann292006-netizen/AI-Engineer-Journey)
