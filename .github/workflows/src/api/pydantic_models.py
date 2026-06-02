"""
Pydantic Models for API Request/Response Validation
Task 6 Component 1: Pydantic request/response models
"""

from pydantic import BaseModel, Field, validator
from typing import Optional


class PredictRequest(BaseModel):
    """
    Request model for credit risk prediction
    """
    customer_id: str = Field(..., description="Unique customer identifier", min_length=1, max_length=50)
    total_amount: float = Field(..., description="Total transaction amount", gt=0, le=1_000_000)
    avg_amount: float = Field(..., description="Average transaction amount", gt=0, le=1_000_000)
    transaction_count: int = Field(..., description="Number of transactions", gt=0, le=10_000)
    std_amount: float = Field(0.0, description="Standard deviation of amounts", ge=0, le=1_000_000)
    recency_days: int = Field(..., description="Days since last transaction", ge=0, le=3650)
    
    @validator('avg_amount')
    def validate_avg_amount(cls, v, values):
        if 'total_amount' in values and 'transaction_count' in values:
            expected_avg = values['total_amount'] / values['transaction_count']
            if abs(v - expected_avg) > 0.01:
                raise ValueError(f'avg_amount ({v}) does not match total_amount/transaction_count ({expected_avg:.2f})')
        return v
    
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


class PredictResponse(BaseModel):
    """
    Response model for credit risk prediction
    """
    customer_id: str = Field(..., description="Customer identifier")
    risk_probability: float = Field(..., description="Probability of being high-risk", ge=0, le=1)
    risk_category: str = Field(..., description="Risk category")
    recommended_credit_limit: float = Field(..., description="Recommended credit limit in ETB", ge=0)
    recommended_loan_duration_months: int = Field(..., description="Recommended loan duration", ge=1, le=36)
    timestamp: str = Field(..., description="Prediction timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "CUST12345",
                "risk_probability": 0.32,
                "risk_category": "Low Risk",
                "recommended_credit_limit": 5000.00,
                "recommended_loan_duration_months": 6,
                "timestamp": "2026-06-03T10:30:00Z"
            }
        }


class HealthResponse(BaseModel):
    """
    Response model for health check
    """
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Current timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2026-06-03T10:30:00Z"
            }
        }