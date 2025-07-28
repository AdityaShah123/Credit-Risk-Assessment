# Credit Risk Assessment
This project explores machine learning techniques to assess credit default risk. It aims to help financial institutions make better lending decisions by identifying high-risk applicants based on historical financial, demographic, and behavioral data.

## Overview

Credit risk modeling plays a crucial role in loan underwriting. Traditional rule-based systems often fail to detect complex patterns, leading to missed risks or unnecessary rejections. This project applies a combination of supervised and unsupervised learning techniques to a synthetic dataset of loan applications.

## Dataset

- **Source**: Simulated credit risk dataset
- **Observations**: 32,581 records
- **Target Variable**: `loan_status` (1 = Default, 0 = Non-default)
- **Features**: Annual income, home ownership, employment length, loan amount, interest rate, credit history, loan purpose, and more
- **Class Imbalance**: Approximately 22% of applicants defaulted

## Project Workflow

### 1. Data Cleaning and Preprocessing
- Missing value imputation (median/mode)
- Log transformation of skewed numerical features
- Label encoding and one-hot encoding for categorical variables
- Creation of derived features such as debt-to-income ratio and income-interest interaction

### 2. Exploratory Data Analysis (EDA)
- Investigated default trends by home ownership, loan intent, income, and interest rate
- Correlation analysis to identify strong predictors
- Visualizations using scatter plots, heatmaps, and histograms

### 3. Feature Engineering
- Debt-to-Income Ratio
- Income × Interest Rate (interaction feature)
- Credit score related ratios and employment stability

### 4. Modeling
Applied and evaluated the following classification algorithms:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- Support Vector Machines (SVM)

### 5. Dimensionality Reduction and Clustering
- Principal Component Analysis (PCA) for dimensionality reduction and visualization
- K-Means Clustering to segment borrowers into low, medium, and high-risk profiles

## Evaluation Metrics

All models were evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

| Model              | ROC-AUC |
|-------------------|---------|
| Logistic Regression | 0.87   |
| Decision Tree       | 0.84   |
| Random Forest       | 0.93   |
| Gradient Boosting   | 0.92   |
| SVM (with PCA)      | 0.90   |
| AdaBoost            | 0.90   |

Random Forest delivered the highest ROC-AUC and served as the most reliable classifier. PCA-based visualization and K-Means clustering helped enhance interpretability and group borrowers by risk.

## Technologies Used

- Python
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- XGBoost, AdaBoost
- SVM, PCA, K-Means

## Future Work

- Incorporate real-world behavioral or transactional data
- Apply SMOTE or other resampling methods to improve class balance
- Use SHAP or LIME for model explainability
- Explore deep learning models and ensemble stacking
- Deploy via a simple dashboard for decision-makers

## Contributors

- Aditya Shah  
- Manan Parikh  
- Kris Patel  
MSDS Program, Rutgers University
