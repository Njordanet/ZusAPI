from fastapi import APIRouter
from app.api.v1.endpoints import health, weather, lightning, dataset, radar

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health"])
api_router.include_router(weather.router, tags=["Weather"])
api_router.include_router(radar.router, tags=["Doppler Radar & Nowcast"])
api_router.include_router(lightning.router, tags=["Lightning Telemetry"])
api_router.include_router(dataset.router, tags=["Datasets"])
