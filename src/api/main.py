from fastapi import FastAPI
from src.api.pydantic_models import CustomerFeatures, PredictionResponse
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

# Load model from MLflow registry
model = mlflow.pyfunc.load_model("models:/credit-risk-model@production")

@app.get("/")
def read_root():
    return {"message": "Credit Risk Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
def predict_risk(data: CustomerFeatures):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)
    return {"high_risk_probability": float(prediction[0])}
