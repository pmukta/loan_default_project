import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import joblib

# 1️⃣ Load training data
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()

# 2️⃣ Retrain (or you can load your existing model later)
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)

# 3️⃣ Save model for reuse
joblib.dump(dtree, "loan_tree_model.pkl")

# 4️⃣ Plot feature importance
importances = pd.Series(dtree.feature_importances_, index=X_train.columns)
top_features = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 5))
top_features.plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Top 10 Features Affecting Loan Default")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# 5️⃣ Save decision tree Figures (optional, smaller version)
plt.savefig("feature_importance.png")
plt.close()

plt.figure(figsize=(15, 8))
plot_tree(dtree, filled=True, max_depth=3, feature_names=X_train.columns, fontsize=6)
plt.title("Decision Tree (first 3 levels)")
plt.tight_layout()
plt.savefig("decision_tree.png")
plt.close()

print("✅ Visualization complete! Charts saved as PNG files.")
