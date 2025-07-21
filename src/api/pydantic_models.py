from pydantic import BaseModel

class CustomerFeatures(BaseModel):
    Amount: float
    Value: float
    TransactionHour: int
    TransactionMonth: int

class PredictionResponse(BaseModel):
    high_risk_probability: float
