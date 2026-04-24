import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import yaml
import os

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["prepare"]

os.makedirs("data/processed", exist_ok=True)

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

df.to_csv("data/raw.csv", index=False)

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, 
    test_size=params["test_size"], 
    random_state=params["random_state"]
)

pd.DataFrame(X_train, columns=iris.feature_names).to_csv("data/processed/X_train.csv", index=False)
pd.DataFrame(X_test, columns=iris.feature_names).to_csv("data/processed/X_test.csv", index=False)
pd.DataFrame(y_train, columns=["target"]).to_csv("data/processed/y_train.csv", index=False)
pd.DataFrame(y_test, columns=["target"]).to_csv("data/processed/y_test.csv", index=False)

print(f"Data prepared: train={len(X_train)}, test={len(X_test)}")
