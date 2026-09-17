"""FastAPI application entry point."""

from fastapi import FastAPI

from app.api.routes import health, models, prediction

app = FastAPI(
    title="Iranian Churn API",
    description="API for churn model metrics and predictions.",
    version="0.1.0",
)

app.include_router(health.router, prefix="/api")
app.include_router(models.router, prefix="/api")
app.include_router(prediction.router, prefix="/api")
