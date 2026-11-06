import httpx
from fastapi import APIRouter, Query
from app.config import settings
from app.models.weather import LightningTelemetryResponse

router = APIRouter()


@router.get(
    "/lightning",
    response_model=LightningTelemetryResponse,
    summary="Get localized convective instability (CAPE) & lightning threat level",
)
async def get_lightning(
    lat: float = Query(..., ge=-90.0, le=90.0),
    lon: float = Query(..., ge=-180.0, le=180.0),
    radius_km: float = Query(50.0, ge=1.0, le=500.0),
):
    """
    Evaluates atmospheric convective potential (CAPE J/kg and Lifted Index)
    and active thunderstorm codes around the queried coordinates.
    """
    cape = 0.0
    lifted_idx = 2.0
    wmo_code = 0

    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.get(
                settings.open_meteo_forecast_url,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "weather_code",
                    "hourly": "cape,lifted_index",
                    "forecast_days": 1,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                wmo_code = int(data.get("current", {}).get("weather_code") or 0)
                cape_arr = data.get("hourly", {}).get("cape") or [0.0]
                li_arr = data.get("hourly", {}).get("lifted_index") or [2.0]
                if cape_arr and cape_arr[0] is not None:
                    cape = float(cape_arr[0])
                if li_arr and li_arr[0] is not None:
                    lifted_idx = float(li_arr[0])
    except Exception:
        pass

    if wmo_code in (96, 99) or cape >= 2500:
        threat = "Divine Wrath (Severe)"
    elif wmo_code == 95 or cape >= 1000:
        threat = "High"
    elif cape >= 300 or lifted_idx < 0:
        threat = "Moderate"
    elif cape >= 100:
        threat = "Low"
    else:
        threat = "Calm"

    return LightningTelemetryResponse(
        strikesCount24h=0,
        nearestStrike=None,
        stormThreatLevel=threat,
        capeJkg=round(cape, 1),
        liftedIndex=round(lifted_idx, 2),
        strikes=[],
    )
