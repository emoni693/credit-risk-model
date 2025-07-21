# src/train.py
import pandas as pd
import numpy as np
import mlflow
from mlflow.models import ModelSignature
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from proxy_target_engineering import calculate_rfm, cluster_customers, assign_high_risk

# === Step 1: Load Data ===
df = pd.read_csv("data/data.csv")
df.columns = df.columns.str.lower()  # ensure all lowercase

# Convert to datetime and drop timezone if present
df['transactionstarttime'] = pd.to_datetime(df['transactionstarttime'], errors='coerce', utc=True)
df['transactionstarttime'] = df['transactionstarttime'].dt.tz_localize(None)

snapshot_date = pd.to_datetime("2018-12-01")

# Now it's safe to check .tzinfo
if snapshot_date.tzinfo is not None:
    snapshot_date = snapshot_date.tz_convert(None)



# === Step 2: Generate Target Variable ===
# snapshot_date = snapshot_date.tz_localize(None)
rfm = calculate_rfm(df, snapshot_date)
rfm_clustered = cluster_customers(rfm)
rfm_labeled = assign_high_risk(rfm_clustered)
df = df.merge(rfm_labeled, on="customerid", how="left")

# === Step 3: Feature Engineering ===
df['transactionhour'] = df['transactionstarttime'].dt.hour
df['transactionmonth'] = df['transactionstarttime'].dt.month

features = ['amount', 'value', 'transactionhour', 'transactionmonth']
target = 'high_risk'  # since you used this name in assign_high_risk()


X = df[features].fillna(0)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Step 4: Define Models ===
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier()
}

param_grid = {
    "LogisticRegression": {"clf__C": [0.1, 1, 10]},
    "RandomForest": {"clf__n_estimators": [50, 100], "clf__max_depth": [5, 10]}
}

best_model = None
best_score = 0

# === Step 5: MLflow Tracking ===
mlflow.set_tracking_uri("file:///D:/10/credit-risk-model/mlruns")
# for Docker compatibility
mlflow.set_experiment("credit-risk")

for name, model in models.items():
    print(f"Training {name}...")
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', model)
    ])

    grid = GridSearchCV(pipe, param_grid[name], cv=3, scoring='f1')
    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)
    signature = infer_signature(X_test, y_pred)
    y_prob = grid.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    with mlflow.start_run(run_name=name):
       mlflow.log_params(grid.best_params_)
       mlflow.log_metrics({
         "accuracy": acc,
         "precision": prec,
         "recall": rec,
         "f1_score": f1,
         "roc_auc": roc
    })

    # Log the model
    mlflow.sklearn.log_model(
    sk_model=grid.best_estimator_,
    artifact_path="model",
    registered_model_name="credit-risk-model",
    input_example=X_test.iloc[:2],
    signature=signature
)

    # Register the best model
    if f1 > best_score:
        best_score = f1
        best_model = grid.best_estimator_

        # Register model
        mlflow.sklearn.log_model(
            sk_model=grid.best_estimator_,
            artifact_path="model",
            registered_model_name="BestCreditRiskModel"
        )
print("Training complete. Best model:", best_model)
