# 💰 Loan Default Prediction using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An **AI-powered FinTech project** that predicts whether a loan will likely be **repaid or defaulted**, built using **Random Forest**, **Streamlit**, and **SMOTE** for class balancing.  
Designed to demonstrate how traditional banking expertise can merge with data-driven decision-making in the age of AI.

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

yaml
Copy code

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/pmukta/loan_default_project.git
cd loan_default_project
2️⃣ Create & Activate a Virtual Environment
bash
Copy code
python -m venv venv
venv\Scripts\activate     # On Windows
# OR
source venv/bin/activate  # On Mac/Linux
3️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
4️⃣ Train the Model (Optional)
bash
Copy code
python save_model.py
5️⃣ Run the Application
bash
Copy code
streamlit run app.py
Then open the local URL (usually http://localhost:8501) in your browser to use the app.

🧾 Requirements
All dependencies are listed in requirements.txt, including:

pandas

numpy

scikit-learn

imbalanced-learn

streamlit

Install everything with:

bash
Copy code
pip install -r requirements.txt
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
🐍 Python 3.9+

📊 Pandas, NumPy

🧠 Scikit-learn

⚖️ Imbalanced-learn (SMOTE)

🌐 Streamlit

🚀 Features
Predicts loan repayment or default using trained Random Forest model

Includes SMOTE oversampling for class balancing

Interactive and user-friendly Streamlit interface

Confidence scores for each prediction

Ready to deploy or enhance with new data

👩‍💻 Author
Mukta Seerapu
💼 17 years of experience in Banking | 🎯 Transitioning to AI & Data Science

📧 Email: pmukta@gmail.com
🔗 LinkedIn Profile

🪪 License
Licensed under the MIT License — you’re free to use, modify, and share this project with attribution.

⭐ Acknowledgements
Special thanks to the open-source community behind:

Streamlit

Scikit-learn

Imbalanced-learn

If you found this project useful, consider ⭐ starring the repository on GitHub!