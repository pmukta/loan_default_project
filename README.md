# 💰 Loan Default Prediction App

A Machine Learning web app built with **Streamlit** that predicts whether a loan is likely to be repaid or defaulted.  
The model uses a **Random Forest Classifier** trained on preprocessed financial data and enhanced with **SMOTE** to handle class imbalance.

---

## 🚀 Features

- Predicts loan repayment likelihood based on applicant details  
- Handles class imbalance using **SMOTE (Synthetic Minority Over-sampling Technique)**  
- Interactive Streamlit UI for easy input and visualization  
- Displays prediction confidence levels  
- Model evaluation metrics included (Confusion Matrix, Classification Report)

---

## 🧠 Model Overview

The model is a **RandomForestClassifier** trained with the following parameters:

- `n_estimators=200`
- `max_depth=8`
- `class_weight='balanced'`
- Random seed: `42`

SMOTE is applied to balance the training data between defaulted and repaid loans.

---

## 📂 Project Structure

