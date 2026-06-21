from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import engine, Base
import models
from auth import router as auth_router
from routers.users import router as users_router
from routers.behaviors import router as behaviors_router
from routers.predictions import router as predictions_router
from routers.reports import router as reports_router
from sqlalchemy import text

Base.metadata.create_all(bind=engine)

def _migrate_add_feature_details_column():
    with engine.connect() as conn:
        result = conn.execute(text("PRAGMA table_info(prediction_results)"))
        columns = [row[1] for row in result.fetchall()]
        if "feature_details" not in columns:
            conn.execute(text("ALTER TABLE prediction_results ADD COLUMN feature_details JSON"))
            conn.commit()

_migrate_add_feature_details_column()

app = FastAPI(title="Conversion Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3007", "http://127.0.0.1:3007"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(users_router)
app.include_router(behaviors_router)
app.include_router(predictions_router)
app.include_router(reports_router)
