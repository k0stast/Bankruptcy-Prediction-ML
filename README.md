# Bankruptcy Prediction using Supervised Machine Learning

This project focuses on predicting corporate bankruptcy using real-world financial data and supervised classification techniques. It was developed as part of a postgraduate course and aims to evaluate various machine learning models under some business constraints.

## 🔍 Project Overview

The goal is to classify companies as **healthy (1)** or **bankrupt (2)** based on a set of financial ratios and binary activity indicators. The dataset contains multiple yearly records for different companies.

## 📁 Dataset Description

Each row in the dataset represents one company in a specific year and includes:

- 8 financial performance indicators (e.g., liquidity, profitability, inventory days)
- 3 binary activity indicators (e.g., exports, imports, representation status)
- `target`: 1 = healthy, 2 = bankrupt
- `year`: the fiscal year

## 🧪 Methodology

### 1. Data Preprocessing

- Renamed columns for clarity
- Checked and confirmed absence of missing values
- Normalized numeric features to [0, 1] using MinMaxScaler

### 2. Visualization

- Figure 1: Number of healthy vs bankrupt companies per year
- Figure 2: Min, Max, Mean values of indicators by class (log-scale)

### 3. Cross-validation and Balancing

- Used **StratifiedKFold (4 splits)** to maintain class distribution across folds
- Applied undersampling to achieve a 3:1 ratio (healthy:bankrupt) in the training set
- Ensured test set remained untouched for fair evaluation

### 4. Model Training

Trained and evaluated the following classifiers:

- Linear Discriminant Analysis (LDA)
- Logistic Regression
- Decision Tree
- Random Forest
- k-Nearest Neighbors (k-NN)
- Naive Bayes
- Support Vector Machine (SVM)
- Multi-Layer Perceptron (MLP)

### 5. Metrics Used

For each model and fold (on both train and test sets):

- Accuracy
- Precision
- Recall (for bankrupt class)
- Specificity (for healthy class)
- F1 Score
- ROC-AUC

### 6. Performance Recording

Each run logged the confusion matrix, evaluation metrics, and metadata (fold, model, balance info) into a DataFrame, then exported to `balancedDataOutcomes.csv`.

## ✅ Key Findings

- **Random Forest** was identified as the best-performing model overall, achieving:
  - Recall ≥ 60% for bankrupt companies
  - Specificity ≥ 70% for healthy companies
- Balancing the training set improved minority class recognition without harming overall accuracy.

## 🛠️ Technologies Used

- Python 3.x
- pandas
- scikit-learn
- matplotlib
- seaborn
- Excel
- Google Colab

## 📚 References

- [scikit-learn documentation](https://scikit-learn.org/stable/)
- [pandas documentation](https://pandas.pydata.org/)
- [Wikipedia – Confusion Matrix](https://en.wikipedia.org/wiki/Confusion_matrix)
