# 🏦 Loan Default Prediction Model

A **machine learning project** that predicts whether a loan will default or be repaid, using **SMOTE for class balancing** and a **Random Forest classifier**.  
This project also includes a **Streamlit web app** for easy user interaction.

---

## 📁 Project Structure

loan_default_project/
│
├── app.py # Streamlit web application
├── save_model.py # Model training & saving (with SMOTE)
├── model_train.py # Model training script
├── preprocess.py # Data preprocessing logic
├── visualize_model.py # Model visualization (feature importance, decision tree)
├── test_model.py # Testing the saved model
│
├── loan_default_model.pkl # Trained Random Forest model
├── X_train.csv
├── X_test.csv
├── y_train.csv
├── y_test.csv
│
├── feature_importance.png # Feature importance chart
├── decision_tree.png # Decision tree visualization
│
├── requirements.txt # Python dependencies
├── README.md # Documentation
└── Loan_Default_Model_Report.pdf # Detailed project report

yaml
Copy code

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/pmukta/loan_default_project.git
cd loan_default_project
2️⃣ Create and Activate a Virtual Environment
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
4️⃣ Train the Model (optional)
bash
Copy code
python save_model.py
5️⃣ Run the App
bash
Copy code
streamlit run app.py

Then open the local URL (e.g. http://localhost:8501) in your browser to use the app.

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
📧 pmukta@gmail.com

🔗 LinkedIn Profile www.linkedin.com/in/mukta-puvvula-94861819b

🪪 License

Licensed under the MIT License — you’re free to use and modify this project with credit.

⭐ Acknowledgements
Special thanks to the open-source community behind:

Streamlit

Scikit-learn

Imbalanced-learn

If you found this project useful, consider ⭐ starring the repository on GitHub!
