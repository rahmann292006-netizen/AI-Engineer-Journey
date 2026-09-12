# AI Engineer Journey

> A hands-on learning repository documenting my path from Python and data analysis to machine learning, deployment, and generative AI.

This is not a collection of tutorial notes alone. Each section contains experiments, notebooks, visualizations, and small projects built while learning the underlying concepts.

## Current progress

The journey is organised as an 84-day roadmap. The foundations in Python, data analysis, visualization, and exploratory data analysis are complete. **Machine learning fundamentals (Days 31–45) are complete** — covering classification, regression, model evaluation, cross-validation, regularization, and model selection.

| Phase | Focus | Status |
| --- | --- | --- |
| 1 | Python, data analysis, and EDA | Complete |
| 2 | Machine learning fundamentals | **Complete (Days 31–45)** |
| 3 | Deep learning | Planned |
| 4 | MLOps and deployment | Planned |
| 5 | Generative AI and LLMs | Planned |

## Machine Learning Progress: Days 31–45

### 🎯 Classification Track (Days 31–36)
- **Day 31:** Machine Learning Workflow Fundamentals — train/test split, stratification, ML pipeline
- **Day 32:** First ML Model — Logistic Regression — binary/multiclass classification, decision boundaries
- **Day 33:** Model Evaluation & Metrics — accuracy, precision, recall, F1, confusion matrix, ROC-AUC
- **Day 34:** Feature Scaling & Preprocessing Pipeline — StandardScaler, MinMaxScaler, Pipeline
- **Day 35:** K-Nearest Neighbors (KNN) Classification — distance metrics, neighbor search, scaling impact
- **Day 36:** Hyperparameter Tuning with GridSearchCV & Pipelines — parameter grids, cross-validated search

### 📈 Regression Track (Days 37–39)
- **Day 37:** Linear Regression — Single Feature — slope, intercept, R², residual basics
- **Day 38:** Multiple Linear Regression — multiple features, coefficient interpretation, multicollinearity
- **Day 39:** Regression Model Evaluation & Residual Analysis — MAE/MSE/RMSE/R², residual plots, diagnostics

### 🔬 Hands-on Practice (Days 40–45)
- **Day 40:** Simple Linear Regression — From Scratch — gradient descent, cost function, manual vs sklearn
- **Day 41:** Residual Analysis Deep Dive — heteroscedasticity, autocorrelation, leverage, Cook's distance
- **Day 42:** Overfitting vs Underfitting — polynomial complexity, bias-variance tradeoff, train-test gaps
- **Day 43:** Cross-Validation — K-Fold, stratified CV, cross_val_score, model stability assessment
- **Day 44:** Regularization — Ridge & Lasso — L1/L2 penalties, alpha tuning, coefficient shrinkage, feature selection
- **Day 45:** Model Selection — comparing models, CV scores, generalization, practical validation

All ML work is implemented in `Machine Learning/Ml_workflow.ipynb`.

## Explore the work

| Area | What it covers | Start here |
| --- | --- | --- |
| NumPy | Arrays, indexing, reshaping, vectorised operations, and image manipulation | [Module notes](Numpy/README.md) |
| Pandas | DataFrames, cleaning, selection, joins, and CSV workflows | [Module notes](Pandas/README.md) |
| Matplotlib | Core plotting and visualisation practice | [Module notes](Matplotlib/README.md) |
| EDA | Dataset inspection, cleaning, distributions, relationships, and outliers | [EDA guide](EDA/README.md) |
| Machine learning | Classification, regression, cross-validation, regularization, model selection | [ML progress notes](Machine%20Learning/Readme.md) |

### Featured notebooks and project

- [NumPy hands-on practice](Numpy/numpy_hands_on.ipynb)
- [Pandas basics](Pandas/pandas_basics.ipynb)
- [Matplotlib practice](Matplotlib/matplotlib.ipynb)
- [EDA workout](EDA/eda_final_workout.ipynb)
- [Machine-learning workflow](Machine%20Learning/Ml_workflow.ipynb) — **Days 31–45 complete**
- [Image Matrix Operations](Numpy/Projects/Image-Matrix-Operations/README.md) — a small NumPy and Pillow project for flipping, cropping, rotating, brightening, grayscaling, and inverting images.

## Learning approach

I use a simple cycle throughout the repository:

```text
Learn a concept -> practise it in code -> apply it to data -> inspect the result -> document the takeaway
```

For example, the EDA work moves from checking data quality to understanding distributions, relationships, correlations, and unusual observations before modelling. The machine-learning work then builds on that foundation with reproducible training and evaluation workflows.

## Tools

- Python
- Jupyter Notebook
- NumPy and pandas
- Matplotlib and Seaborn
- scikit-learn
- Pillow (PIL)

## Run a notebook locally

1. Clone the repository and enter it.
2. Create and activate a virtual environment.
3. Install the libraries needed by the notebook you want to run.
4. Start Jupyter:

```bash
jupyter notebook
```

The notebooks use a mixture of included CSV files, built-in scikit-learn datasets, and small generated datasets. Read the notebook before running it to see its specific dependencies and data source.

## Repository layout

```text
AI-Engineer-Journey/
├── Numpy/                 # NumPy practice and an image-matrix project
├── Pandas/                # Data manipulation practice
├── Matplotlib/            # Plotting exercises and exported charts
├── EDA/                   # Exploratory data-analysis notebook and guide
├── Machine Learning/      # ML notebooks, diagnostics, FastAPI, and GenAI work
├── *.csv                  # Practice datasets
└── README.md
```

## Technologies & Topics Learned So Far (Days 1–45)

| Category | Topics |
| --- | --- |
| **Python & Data** | NumPy, pandas, Matplotlib, Seaborn, Jupyter |
| **EDA** | Data cleaning, distributions, correlations, outliers, feature engineering basics |
| **Classification** | Logistic Regression, KNN, GridSearchCV, Pipelines, evaluation metrics (precision, recall, F1, ROC-AUC) |
| **Regression** | Simple & Multiple Linear Regression, residual analysis, heteroscedasticity, multicollinearity |
| **Model Validation** | Train/test split, K-Fold CV, cross_val_score, bias-variance tradeoff |
| **Regularization** | Ridge (L2), Lasso (L1), alpha tuning, coefficient shrinkage, feature selection |
| **Model Selection** | Comparing models via CV, generalization assessment, practical validation |

## Current status

✅ **Days 1–45 complete** (Python foundations → EDA → Machine Learning fundamentals)
🔜 **Next:** Day 46 — Deep Learning fundamentals (neural networks, backpropagation, PyTorch/TensorFlow basics)

---

## Connect

- GitHub: [@rahmann292006-netizen](https://github.com/rahmann292006-netizen)
- Repository: [AI-Engineer-Journey](https://github.com/rahmann292006-netizen/AI-Engineer-Journey)

> The best way to learn is to build, inspect what happened, and keep going.
