import mlflow

mlflow.set_experiment("mlflow-test")

with mlflow.start_run(run_name="demo"):
    mlflow.log_param("param1", 5)
    mlflow.log_metric("metric1", 0.85)
