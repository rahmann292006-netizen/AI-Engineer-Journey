# Day 31/84 — Machine Learning Workflow

Started the transition from EDA to Machine Learning today.

### What I learned

- Difference between features (X) and target (y)
- Train/Test split
- `train_test_split()`
- `test_size`
- `random_state`
- `stratify`
- Basic Machine Learning workflow
- Why keeping test data separate matters

### Practice

Used the Iris dataset to:

- Load features and target
- Split data into training and testing sets
- Compare dataset shapes
- Inspect target distributions
- Experiment with different test sizes
- Understand the role of random state

### Basic Workflow

Data
→ Cleaning
→ EDA
→ Feature/Target Selection
→ Train/Test Split
→ Model Training
→ Evaluation
→ Improvement

### Key Takeaway

EDA helps understand the data.

Machine Learning starts when we use that data to build a system that can learn patterns and make predictions.

Next → First Machine Learning Model.

# Machine Learning Journey 🤖

## Day 32–33/84 — First ML Model + Model Evaluation

After understanding the basic Machine Learning workflow, I finally moved from preparing data to actually training and evaluating a model.

---

## 🚀 Day 32 — My First ML Model

Today I trained my first classification model using **Logistic Regression**.

### What I learned

- Loading a dataset
- Separating features (`X`) and target (`y`)
- Train/Test splitting
- Creating a Logistic Regression model
- Training with `.fit()`
- Making predictions with `.predict()`
- Measuring model accuracy
- Using a trained model to predict new data

### Dataset

I used the **Iris dataset**.

Features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

Target classes:

- Setosa
- Versicolor
- Virginica

### Workflow

```text
Iris Dataset
     ↓
Features (X) + Target (y)
     ↓
Train / Test Split
     ↓
Logistic Regression
     ↓
Model Training
     ↓
Predictions
     ↓
Accuracy
