import sys

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

sys.path.append('.')
from utils.io.dataset import load_tabular_dataset


X_train, y_train = load_tabular_dataset('diabetes130', train=True)
X_test, y_test   = load_tabular_dataset('diabetes130', train=False)


# Train XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", conf_matrix)
