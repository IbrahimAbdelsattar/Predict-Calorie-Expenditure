# 🔥 Calories Burnt Prediction using Machine Learning

## 📌 Project Overview

This project aims to develop a machine learning model that can accurately predict the number of **Calories burnt** by an individual during physical activity based on various physiological and activity-related features. The primary motivation behind this project is to support health and fitness applications by providing reliable calorie estimation, which can help users track and manage their energy expenditure.

---

## 📂 Dataset Description

The training and testing datasets were synthetically generated from a deep learning model trained on the **Calories Burnt Prediction** dataset. While the feature distributions are close to the original dataset, there are slight variations, which makes it a valuable exercise in generalization and model robustness.

- **`train.csv`**: Training data containing both features and the target variable (`Calories`).
- **`test.csv`**: Testing data without the target column; used for model inference.
- **`sample_submission.csv`**: Sample file showing the expected submission format.

---

## 📈 Features

Each row in the dataset represents an individual's physical characteristics and recorded activity data. Below are the key input features:

- `Gender`: Biological sex of the individual (Male/Female)
- `Age`: Age in years
- `Height`: Height in centimeters
- `Weight`: Weight in kilograms
- `Duration`: Duration of the activity (minutes)
- `Heart_Rate`: Heart rate during the activity (bpm)
- `Body_Temp`: Recorded body temperature (°C)
- `Calories`: *(Target Variable)* Total calories burnt

---

## ⚙️ Methodology

The modeling pipeline includes the following key steps:

1. **Data Preprocessing**
   - Handled categorical data using label encoding.
   - Removed unnecessary columns such as `id`.
   - Normalized and cleaned data for consistent model input.

2. **Exploratory Data Analysis (EDA)**
   - Univariate and bivariate visualizations to understand feature distributions.
   - Correlation heatmaps to identify important relationships.

3. **Model Training**
   - Trained and evaluated six different regression models:
     - Linear Regression
     - Ridge Regression
     - K-Nearest Neighbors (KNN)
     - Random Forest Regressor
     - Gradient Boosting Regressor
     - XGBoost Regressor
   - Used RMSE and R² Score for evaluation.

4. **Model Selection**
   - XGBoost Regressor was selected as the final model based on superior performance metrics.

5. **Prediction**
   - Predictions were made on the test set using the best-performing model.
   - Final results saved in the required submission format.

---

## 🏆 Model Performance (Validation Set)

| Model                  | RMSE       | R² Score   |
|------------------------|------------|------------|
| **XGBoost Regressor**  | **3.80**   | **0.9963** |
| Random Forest Regressor| 3.82       | 0.9962     |
| K-Nearest Neighbors    | 4.50       | 0.9948     |
| Gradient Boosting      | 4.75       | 0.9942     |
| Ridge Regression       | 11.06      | 0.9684     |
| Linear Regression      | 11.06      | 0.9684     |

---

## 🧠 Technologies Used

- Python 🐍
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Seaborn & Matplotlib (for EDA and visualization)

---
