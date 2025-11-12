import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# 1️⃣ Load preprocessed data
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

# 2️⃣ Balance data using SMOTE
print("⚙️ Applying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("✅ Class distribution after SMOTE:")
print(pd.Series(y_train_res).value_counts())

# 3️⃣ Train Random Forest model
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)
rf.fit(X_train_res, y_train_res)

# 4️⃣ Evaluate model
predictions = rf.predict(X_test)
print("\n✅ Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\n✅ Classification Report:")
print(classification_report(y_test, predictions))

# 5️⃣ Show sample default predictions
X_test_preview = X_test.copy()
X_test_preview["Actual"] = y_test
X_test_preview["Predicted"] = predictions

defaults = X_test_preview[X_test_preview["Predicted"] == 1]
print("\n⚠️ Sample of predicted DEFAULT cases:")
if len(defaults) == 0:
    print("👉 No default cases predicted.")
else:
    print(defaults.head(10))

# 6️⃣ Save trained model
with open("loan_default_model.pkl", "wb") as f:
    pickle.dump(rf, f)

print("\n💾 Model saved successfully as loan_default_model.pkl")
