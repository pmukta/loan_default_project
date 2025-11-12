import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# 1️⃣ Load dataset
loans = pd.read_csv("loan_data.csv")

# 2️⃣ Encode categorical column (purpose)
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_purpose = encoder.fit_transform(loans[['purpose']])

# Convert encoded array to DataFrame
encoded_df = pd.DataFrame(encoded_purpose, columns=encoder.get_feature_names_out(['purpose']))

# Combine numeric + encoded columns
loans_encoded = pd.concat([loans.drop('purpose', axis=1), encoded_df], axis=1)

# 3️⃣ Split data into features (X) and target (y)
X = loans_encoded.drop('not.fully.paid', axis=1)
y = loans_encoded['not.fully.paid']

# 4️⃣ Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5️⃣ Save for later use
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("✅ Data preprocessing completed and files saved successfully!")
