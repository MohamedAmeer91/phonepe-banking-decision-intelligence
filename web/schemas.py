"""Pydantic request models."""
from typing import List, Literal

from pydantic import BaseModel, Field, field_validator


class ScoreRequest(BaseModel):
    X1: float = Field(..., ge=0, le=600, description="account_age_months")
    X2: int = Field(..., ge=0, le=100, description="failed_txn_count_90d")
    X3: float = Field(..., ge=0, le=1, description="credit_utilisation_ratio")
    X4: float = Field(..., ge=0, le=1000, description="avg_monthly_inflow_inr_thousands")
    X5: float = Field(..., ge=0, le=100, description="repayment_score")
    X6: Literal[0, 1] = Field(..., description="irregular_salary_flag")
    X7: Literal[0, 1] = Field(..., description="autopay_flag")
    tau: float = Field(0.15, ge=0, le=1)


class SegmentRequest(BaseModel):
    X1: float = Field(..., ge=0, le=600)
    X2: float = Field(..., ge=0, le=100)
    X3: float = Field(..., ge=0, le=1)
    X4: float = Field(..., ge=0, le=1000)
    X5: float = Field(..., ge=0, le=100)


class EnsembleRequest(BaseModel):
    models: List[Literal["logistic", "knn", "stump"]] = Field(..., min_length=1, max_length=3)
    mode: Literal["hard", "soft"] = "hard"
    tau: float = Field(0.3, ge=0, le=1)

    @field_validator("models")
    @classmethod
    def unique(cls, v):
        if len(set(v)) != len(v):
            raise ValueError("models must be unique")
        return v
