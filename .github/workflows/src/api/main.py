"""
FastAPI Deployment for Credit Risk Model
Task 6 Component 1: FastAPI application with /predict endpoint
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import joblib
import os

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Prediction API",
    description="Predicts customer credit risk probability for Bati Bank's buy-now-pay-later service",
    version="1.0.0"
)


# Request model
class CreditRiskRequest(BaseModel):
    customer_id: str = Field(..., description="Unique customer identifier")
    total_amount: float = Field(..., description="Total transaction amount", gt=0)
    avg_amount: float = Field(..., description="Average transaction amount", gt=0)
    transaction_count: int = Field(..., description="Number of transactions", gt=0)
    std_amount: float = Field(0.0, description="Standard deviation of amounts")
    recency_days: int = Field(..., description="Days since last transaction", ge=0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "CUST12345",
                "total_amount": 2500.00,
                "avg_amount": 125.00,
                "transaction_count": 20,
                "std_amount": 45.50,
                "recency_days": 5
            }
        }


# Response model
class CreditRiskResponse(BaseModel):
    customer_id: str
    risk_probability: float = Field(..., ge=0, le=1, description="Probability of being high-risk (0-1)")
    risk_category: str = Field(..., description="Risk category: Low, Medium, or High")
    recommended_credit_limit: float = Field(..., description="Recommended credit limit in ETB")
    recommended_loan_duration_months: int = Field(..., description="Recommended loan duration in months")
    
    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "CUST12345",
                "risk_probability": 0.32,
                "risk_category": "Low Risk",
                "recommended_credit_limit": 5000.00,
                "recommended_loan_duration_months": 6
            }
        }


# Load model (would be loaded from MLflow in production)
# For demo, create a simple model
def load_model():
    """Load the trained model from MLflow or local file"""
    # In production, load from MLflow:
    # model = mlflow.pyfunc.load_model("models:/CreditRiskModel/Production")
    
    # For demo, return a simple function
    def predict(features):
        # Simple risk score based on recency and transaction count
        recency = features.get('recency_days', 30)
        count = features.get('transaction_count', 5)
        
        # Higher risk if high recency (inactive) or low transaction count
        risk_score = (min(recency / 60, 1.0) * 0.6) + (max(1 - count / 20, 0) * 0.4)
        return risk_score
    
    return predict


def categorize_risk(probability):
    """Categorize risk based on probability threshold"""
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.6:
        return "Medium Risk"
    else:
        return "High Risk"


def calculate_credit_limit(probability):
    """Calculate recommended credit limit based on risk probability"""
    if probability < 0.3:
        return 5000.00
    elif probability < 0.6:
        return 2000.00
    else:
        return 500.00


def calculate_loan_duration(probability):
    """Calculate recommended loan duration based on risk probability"""
    if probability < 0.3:
        return 12
    elif probability < 0.6:
        return 6
    else:
        return 3


# Load model on startup
model = load_model()


@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Credit Risk Prediction API",
        "version": "1.0.0"
    }


@app.post("/predict", response_model=CreditRiskResponse)
def predict_risk(request: CreditRiskRequest):
    """
    Predict credit risk probability for a customer
    
    Takes customer transaction features and returns risk probability,
    risk category, recommended credit limit, and loan duration.
    """
    try:
        # Prepare features for prediction
        features = {
            'recency_days': request.recency_days,
            'transaction_count': request.transaction_count,
            'total_amount': request.total_amount,
            'avg_amount': request.avg_amount,
            'std_amount': request.std_amount
        }
        
        # Get risk probability
        risk_probability = float(model(features))
        
        # Categorize risk
        risk_category = categorize_risk(risk_probability)
        
        # Calculate recommendations
        credit_limit = calculate_credit_limit(risk_probability)
        loan_duration = calculate_loan_duration(risk_probability)
        
        return CreditRiskResponse(
            customer_id=request.customer_id,
            risk_probability=risk_probability,
            risk_category=risk_category,
            recommended_credit_limit=credit_limit,
            recommended_loan_duration_months=loan_duration
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)