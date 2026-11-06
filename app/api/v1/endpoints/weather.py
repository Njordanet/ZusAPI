from fastapi import APIRouter, Query, HTTPException
from app.models.weather import WeatherResponse
from app.services.weather_provider import weather_service

router = APIRouter()

@router.get(
    "/weather",
    response_model=WeatherResponse,
    summary="Get aggregated current weather & forecasts"
)
async def get_weather(
    lat: float = Query(..., ge=-90.0, le=90.0, description="Latitude"),
    lon: float = Query(..., ge=-180.0, le=180.0, description="Longitude"),
    city: str = Query("Unknown Location", description="City or Region Name")
):
    try:
        data = await weather_service.get_forecast(latitude=lat, longitude=lon, city_name=city)
        return data
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch meteorological data: {str(e)}")
