from typing import Optional
from fastapi import APIRouter, Query, HTTPException
from app.models.weather import WeatherResponse
from app.services.weather_provider import weather_service

router = APIRouter()


def _c_to_f(val: Optional[float]) -> Optional[float]:
    if val is None:
        return None
    return round(val * 9.0 / 5.0 + 32.0, 1)


def _kmh_to_unit(val: Optional[float], unit_mode: str) -> Optional[float]:
    if val is None:
        return None
    if unit_mode in ("imperial", "us"):
        return round(val * 0.621371, 1)  # mph
    if unit_mode == "si":
        return round(val / 3.6, 2)       # m/s
    return val


def _apply_weather_transforms(
    resp: WeatherResponse,
    units: str,
    exclude: Optional[str],
    hours: Optional[int],
) -> WeatherResponse:
    unit_mode = units.lower().strip()
    excluded = {s.strip().lower() for s in (exclude or "").split(",") if s.strip()}

    if unit_mode in ("metric", "ca", "") and not excluded and hours is None:
        return resp

    data = resp.model_copy(deep=True)

    if hours is not None and hours > 0:
        data.hourly = data.hourly[:hours]

    if "nearbystations" in excluded or "stations" in excluded:
        data.nearbyStations = []
    if "modelweights" in excluded or "weights" in excluded:
        data.modelWeights = []
    if "alerts" in excluded:
        data.alerts = []
    if "hourly" in excluded:
        data.hourly = []
    if "daily" in excluded:
        data.daily = []
    if "doppler" in excluded or "dopplertelemetry" in excluded:
        data.dopplerTelemetry = None

    if unit_mode in ("imperial", "us", "si"):
        is_us = unit_mode in ("imperial", "us")
        c = data.current
        if is_us:
            c.temperature = _c_to_f(c.temperature) or c.temperature
            c.feelsLike = _c_to_f(c.feelsLike) or c.feelsLike
            c.tempMin = _c_to_f(c.tempMin) or c.tempMin
            c.tempMax = _c_to_f(c.tempMax) or c.tempMax
            c.dewPoint = _c_to_f(c.dewPoint)
            c.visibility = round(c.visibility * 0.621371, 1)
        c.windSpeed = _kmh_to_unit(c.windSpeed, unit_mode) or 0.0
        c.windGust = _kmh_to_unit(c.windGust, unit_mode)

        for h in data.hourly:
            if is_us:
                h.temperature = _c_to_f(h.temperature) or h.temperature
                h.tempMin = _c_to_f(h.tempMin)
                h.tempMax = _c_to_f(h.tempMax)
                h.feelsLike = _c_to_f(h.feelsLike)
                h.feelsMin = _c_to_f(h.feelsMin)
                h.feelsMax = _c_to_f(h.feelsMax)
                h.dewPoint = _c_to_f(h.dewPoint)
                h.dewMin = _c_to_f(h.dewMin)
                h.dewMax = _c_to_f(h.dewMax)
                if h.visibilityKm is not None:
                    h.visibilityKm = round(h.visibilityKm * 0.621371, 2)
            h.windSpeed = _kmh_to_unit(h.windSpeed, unit_mode) or 0.0
            h.windMin = _kmh_to_unit(h.windMin, unit_mode)
            h.windMax = _kmh_to_unit(h.windMax, unit_mode)
            h.windGusts = _kmh_to_unit(h.windGusts, unit_mode)
            h.gustMin = _kmh_to_unit(h.gustMin, unit_mode)
            h.gustMax = _kmh_to_unit(h.gustMax, unit_mode)

        for d in data.daily:
            if is_us:
                d.minTemperature = _c_to_f(d.minTemperature) or d.minTemperature
                d.maxTemperature = _c_to_f(d.maxTemperature) or d.maxTemperature
            d.windSpeedMax = _kmh_to_unit(d.windSpeedMax, unit_mode) or 0.0

    return data


@router.get(
    "/weather",
    response_model=WeatherResponse,
    summary="Get aggregated current weather & forecasts (supports metric/imperial/si units & block filtering)",
)
async def get_weather(
    lat: float = Query(..., ge=-90.0, le=90.0, description="Latitude"),
    lon: float = Query(..., ge=-180.0, le=180.0, description="Longitude"),
    city: str = Query("Unknown Location", description="City or Region Name"),
    units: str = Query("metric", description="Units: metric (default °C, km/h), imperial/us (°F, mph), or si (°C, m/s)"),
    exclude: Optional[str] = Query(None, description="Comma-separated blocks to omit (e.g. nearbyStations,modelWeights,hourly,daily,alerts)"),
    hours: Optional[int] = Query(None, ge=1, le=240, description="Optional limit on number of hourly forecast points returned"),
):
    try:
        data = await weather_service.get_forecast(latitude=lat, longitude=lon, city_name=city)
        return _apply_weather_transforms(data, units=units, exclude=exclude, hours=hours)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch meteorological data: {str(e)}")
