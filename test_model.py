import pickle
import pandas as pd

# Load trained model
with open("loan_default_model.pkl", "rb") as f:
    model = pickle.load(f)

# 🔹 Define one sample input (you can change values)
sample = {
    'credit.policy': [1],
    'purpose': ['credit_card'],  # try 'debt_consolidation' too
    'int.rate': [0.12],
    'installment': [200],
    'log.annual.inc': [10.5],
    'dti': [15.0],
    'fico': [700],
    'days.with.cr.line': [4000],
    'revol.bal': [5000],
    'revol.util': [45.0],
    'inq.last.6mths': [2],
    'delinq.2yrs': [0],
    'pub.rec': [0]
}

# Convert to DataFrame
sample_df = pd.DataFrame(sample)

# 🔹 Perform one-hot encoding like training
sample_encoded = pd.get_dummies(sample_df, columns=['purpose'])

# 🔹 Define all expected columns (from training)
expected_cols = [
    'credit.policy', 'int.rate', 'installment', 'log.annual.inc', 'dti', 'fico',
    'days.with.cr.line', 'revol.bal', 'revol.util', 'inq.last.6mths', 'delinq.2yrs', 'pub.rec',
    'purpose_credit_card', 'purpose_debt_consolidation', 'purpose_educational',
    'purpose_home_improvement', 'purpose_major_purchase', 'purpose_small_business'
]

# 🔹 Add missing columns and reorder
for col in expected_cols:
    if col not in sample_encoded.columns:
        sample_encoded[col] = 0
sample_encoded = sample_encoded[expected_cols]

# 🔹 Make prediction
prediction = model.predict(sample_encoded)[0]

# 🔹 Output
print("✅ Prediction Complete!")
if prediction == 0:
    print("💰 Loan likely to be repaid.")
else:
    print("⚠️ Loan likely to default.")
