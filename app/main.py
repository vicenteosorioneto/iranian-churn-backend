"""FastAPI application entry point."""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import analytics, health, models, prediction

app = FastAPI(
    title="Iranian Churn API",
    description="API for churn model metrics and predictions.",
    version="0.1.0",
)

# Configure origins explicitly; browsers cannot call the API without CORS headers.
# Example: CHURN_CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
origins = [
    value.strip()
    for value in os.getenv(
        "CHURN_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if value.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(health.router, prefix="/api")
app.include_router(analytics.router, prefix="/api")
app.include_router(models.router, prefix="/api")
app.include_router(prediction.router, prefix="/api")
