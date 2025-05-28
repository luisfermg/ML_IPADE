# Passenger Satisfaction Classification Report

This report summarizes the steps followed to build a model for predicting airline passenger satisfaction.

## Overview

The problem is formulated as a **supervised classification** task. Each row in the dataset contains demographic information and service ratings along with a label indicating whether the passenger was **satisfied** or **neutral/dissatisfied**. Our goal is to learn a model that predicts the satisfaction label for new passengers.

## Steps

1. **Data loading and cleaning**
   - The dataset provided on Kaggle was loaded from `train.csv` and `test.csv`.
   - Missing values were inspected and imputed inside a preprocessing pipeline using `SimpleImputer`.
   - Numeric features were standardized with `StandardScaler` and categorical features were encoded using `OneHotEncoder`.

2. **Exploratory analysis**
   - Descriptive statistics such as mean age and class distributions were computed.
   - Counts of the `satisfaction` label showed a slight class imbalance toward the `neutral or dissatisfied` class.

3. **Model selection**
   - Two algorithms were compared using five‑fold cross validation: **Logistic Regression** and **Random Forest**.
   - A grid search over the regularization strength of the logistic regression was performed. The best parameters were selected based on cross‑validation accuracy.

4. **Training and evaluation**
   - The best estimator from the grid search was trained on the training split and evaluated on a validation split and on the provided `test.csv` file.
   - Accuracy and a full classification report were printed in the notebook.
   - The trained model was saved to `best_model.joblib` for reuse.

5. **Prediction on new data**
   - A convenience function `predict_new` was implemented in the notebook to load a CSV file and generate predictions with the saved model.

## Questions

**a. What type of model is your algorithm, supervised or unsupervised learning?**

The algorithms used (logistic regression and random forest) are supervised learning models because they train on labeled examples.

**b. Is it a regression, classification, or clustering problem?**

It is a classification problem since the goal is to assign each passenger to one of two classes: `satisfied` or `neutral or dissatisfied`.

**c. Why did you use that algorithm?**

Logistic regression is a simple, interpretable baseline for binary classification. Random forests were tested because they capture nonlinear relationships and typically provide strong performance with minimal tuning. Grid search identified the best regularization strength for logistic regression.

**d. What is the main question or problem you want to solve?**

We aim to predict passenger satisfaction based on the features describing their flight experience. This can help an airline identify factors that influence customer satisfaction and improve service quality.

## Performance Metrics

The notebook reports the following metrics:

- Cross‑validation accuracy for each algorithm.
- Validation and test accuracy of the best estimator.
- A classification report with precision, recall and F1‑score.

These results demonstrate how well the model generalizes to unseen data.

