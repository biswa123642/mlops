from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import (
    ALLOWED_ORIGINS,
    HOST,
    PORT,
)
from app.routes import router


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Backend API for customer churn prediction",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=[
        "GET",
        "POST",
    ],
    allow_headers=[
        "Content-Type",
    ],
)


app.include_router(router)


@app.get("/")
def root():
    return {
        "application": "Customer Churn Prediction API",
        "status": "Running",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=True,
    )