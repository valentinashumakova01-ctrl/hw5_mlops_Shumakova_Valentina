import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import yaml
import mlflow
import os

os.makedirs("models", exist_ok=True)

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["train"]

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

model = RandomForestClassifier(
    n_estimators=params["n_estimators"],
    max_depth=params["max_depth"],
    random_state=params["random_state"]
)
model.fit(X_train, y_train)

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

mlflow.set_experiment("iris_experiment")
with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_artifact("models/model.pkl")
    
    report = classification_report(y_test, y_pred, output_dict=True)
    mlflow.log_metric("precision", report["weighted avg"]["precision"])
    mlflow.log_metric("recall", report["weighted avg"]["recall"])

print(f"Model trained! Accuracy: {accuracy:.4f}")
print(f"MLflow run saved. Run 'mlflow ui' to see results")
