# 💰 Loan Default Prediction App

![App Preview](app_preview.png)
*A Machine Learning web app that predicts whether a loan will be repaid or defaulted.*

---

## 🚀 Overview

This project uses a **Random Forest Classifier** enhanced with **SMOTE (Synthetic Minority Over-sampling Technique)**  
to predict whether a borrower is likely to default on a loan.

The app is built with **Streamlit**, providing a simple and interactive user interface  
for entering loan applicant details and viewing prediction results in real time.

---

## 🌟 Features

- 📊 Predicts loan repayment or default
- ⚖️ Uses **SMOTE** to handle class imbalance
- 🧠 **Random Forest Classifier** for robust predictions
- 💬 Confidence percentage displayed with each result
- 🖥️ User-friendly Streamlit interface

---

## 🧠 Model Details

**Algorithm:** Random Forest Classifier  
**Hyperparameters:**
- `n_estimators=200`
- `max_depth=8`
- `class_weight='balanced'`
- `random_state=42`

---

## 📁 Project Structure

loan_default_project/
│
├── app.py # Streamlit web application
├── save_model.py # Model training & saving (with SMOTE)
├── loan_default_model.pkl # Trained Random Forest model
├── X_train.csv
├── X_test.csv
├── y_train.csv
├── y_test.csv
├── requirements.txt # Python dependencies
└── README.md # Documentation


---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/loan_default_project.git
cd loan_default_project

2️⃣ Create and Activate a Virtual Environment
python -m venv venv
venv\Scripts\activate     # On Windows
# OR
source venv/bin/activate  # On Mac/Linux

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Train the Model (optional)
python save_model.py

5️⃣ Run the App
streamlit run app.py

Open the local URL (e.g. http://localhost:8501) in your browser to use the app.

🧾 Requirements
See requirements.txt

📊 Sample Output

Example 1:
✅ Loan likely to be repaid
Confidence: 91.24%

Example 2:
⚠️ Loan likely to default
Confidence: 78.56%

📈 Model Evaluation
Metric	Score
Accuracy	~0.73
Precision (Default)	0.27
Recall (Default)	0.39
F1-Score (Default)	0.32
🧩 Tech Stack

Python 3.9+

Pandas, NumPy

Scikit-learn

Imbalanced-learn

Streamlit

👨‍💻 Author

Mukta Seerapu
💼 17 years in Banking | 🎯 Transitioning to AI & Data Science
📧 [email: pmukta@gmail.com ]
📧 [LinkedIn Profile](www.linkedin.com/in/mukta-puvvula-94861819b)


🪪 License

Licensed under the MIT License – you’re free to use and modify this project with credit.


