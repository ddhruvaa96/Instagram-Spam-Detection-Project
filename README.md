# Instagram Spam Detection using Machine Learning

## Project Overview

This project detects whether an Instagram account is **Fake (Spam)** or **Genuine** using Machine Learning techniques.

The project performs data preprocessing, exploratory data analysis (EDA), visualization, model training, and prediction using two classification algorithms:

- K-Nearest Neighbors (KNN)
- Decision Tree

The performance of both models is evaluated and compared using accuracy, confusion matrix, and classification report.

---

## Features

- Load dataset from Excel
- Data cleaning
- Missing value analysis
- Duplicate removal
- Exploratory Data Analysis (EDA)
- Data visualization
- Train-Test Split
- KNN Classification
- Decision Tree Classification
- Confusion Matrix
- Classification Report
- Predict a new Instagram account

---

## Dataset

The dataset contains Instagram profile information such as:

- Followers
- Following
- Username Length
- Username Contains Numbers
- Full Name Length
- Private Account
- Business Account
- External URL
- Recently Joined
- Channel Availability
- Guides Availability

### Target Variable

| Value | Meaning |
|-------|---------|
| 0 | Genuine Account |
| 1 | Fake (Spam) Account |

---

## Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

## Machine Learning Workflow

Dataset
↓
Data Cleaning
↓
Exploratory Data Analysis
↓
Input & Target Separation
↓
Train-Test Split
↓
KNN Model
↓
Decision Tree Model
↓
Prediction
↓
Model Evaluation
