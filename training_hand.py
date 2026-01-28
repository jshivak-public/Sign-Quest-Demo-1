import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

CSV_PATH = "asl_abc_data.csv"
MODEL_OUT = "model.pk1"

UNKNOWN_THRESHOLD = 0.80  # runtime: if max prob < this -> "unknown"

df = pd.read_csv(CSV_PATH).dropna()

features = df.drop("Class", axis=1)
labels = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    features, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
])

pipeline.fit(X_train, y_train)

yhat = pipeline.predict(X_test)
print(classification_report(y_test, yhat))

bundle = {
    "pipeline": pipeline,
    "unknown_threshold": UNKNOWN_THRESHOLD,
    "classes": list(pipeline.classes_)
}

with open(MODEL_OUT, "wb") as f:
    pickle.dump(bundle, f)

print("Saved:", MODEL_OUT)
print("Classes:", bundle["classes"])
print("Unknown threshold:", UNKNOWN_THRESHOLD)
