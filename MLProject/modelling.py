import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Telco_Customer_Churn")

DATA_DIR = "Telco-Customer-Churn_preprocessing"

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()

mlflow.sklearn.autolog()

mlflow.sklearn.autolog()

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("=== Hasil Evaluasi Model ===")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-score  : {f1:.4f}")

print("Training selesai. Jalankan 'mlflow ui --backend-store-uri ./mlruns' "
      "untuk melihat hasil tracking di browser.")