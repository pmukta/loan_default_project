import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1️⃣ Load preprocessed data
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

# 2️⃣ Train Decision Tree model
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)

# 3️⃣ Make predictions
predictions = dtree.predict(X_test)

# 4️⃣ Evaluate model
print("✅ Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\n✅ Classification Report:")
print(classification_report(y_test, predictions))

print("🎯 Model training and evaluation completed successfully!")

# --- Save the trained model ---
import pickle

with open("loan_default_model.pkl", "wb") as f:
    pickle.dump(dtree, f)

print("✅ Model saved successfully as loan_default_model.pkl")
