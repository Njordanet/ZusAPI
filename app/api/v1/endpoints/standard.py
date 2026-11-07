import math
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Path, Query, Request, Response, HTTPException

from app.models.weather import WeatherCondition
from app.services.weather_provider import weather_service
from app.services.doppler_service import doppler_service

router = APIRouter()


def _darksky_icon(cond: WeatherCondition) -> str:
    mapping = {
        WeatherCondition.CLEAR: "clear-day",
        WeatherCondition.PARTLY_CLOUDY: "partly-cloudy-day",
        WeatherCondition.OVERCAST: "cloudy",
        WeatherCondition.FOGGY: "fog",
        WeatherCondition.DRIZZLE: "rain",
        WeatherCondition.RAIN: "rain",
        WeatherCondition.HEAVY_RAIN: "rain",
        WeatherCondition.SNOW: "snow",
        WeatherCondition.THUNDERSTORM: "thunderstorm",
        WeatherCondition.SEVERE_THUNDERSTORM: "thunderstorm",
    }
    return mapping.get(cond, "clear-day")


def _compass_to_degrees(compass: Optional[str]) -> int:
    if not compass:
        return 0
    mapping = {
        "N": 0, "NNE": 22, "NE": 45, "ENE": 68,
        "E": 90, "ESE": 112, "SE": 135, "SSE": 158,
        "S": 180, "SSW": 202, "SW": 225, "WSW": 248,
        "W": 270, "WNW": 292, "NW": 315, "NNW": 338,
    }
    return mapping.get(compass.upper(), 0)


def _convert_temp(c: float, units: str) -> float:
    if units == "us":
        return round(c * 9.0 / 5.0 + 32.0, 2)
    return round(c, 2)


def _convert_wind(kmh: float, units: str) -> float:
    if units in ("us", "uk2"):
        return round(kmh * 0.621371, 2)  # mph
    if units == "si":
        return round(kmh / 3.6, 2)       # m/s
    return round(kmh, 2)                 # ca -> km/h


def _convert_dist(km: float, units: str) -> float:
    if units in ("us", "uk2"):
        return round(km * 0.621371, 2)  # miles
    return round(km, 2)                 # km


def _convert_precip_rate(mmhr: float, units: str) -> float:
    if units == "us":
        return round(mmhr / 25.4, 4)    # inches/hr
    return round(mmhr, 3)               # mm/hr


async def _build_darksky_payload(
    lat: float,
    lon: float,
    units: str = "us",
    exclude: Optional[str] = None,
) -> Dict[str, Any]:
    units_norm = units.lower().strip()
    if units_norm == "auto":
        units_norm = "us" if (-170.0 <= lon <= -60.0 and 18.0 <= lat <= 72.0) else "si"
    if units_norm not in ("us", "si", "ca", "uk2"):
        units_norm = "us"

    excluded = {s.strip().lower() for s in (exclude or "").split(",") if s.strip()}

    w_resp = await weather_service.get_forecast(
        latitude=lat,
        longitude=lon,
        city_name=f"{lat:.3f},{lon:.3f}",
    )
    dop = w_resp.dopplerTelemetry or await doppler_service.get_point_telemetry(lat, lon)
    cur = w_resp.current
    now_ts = int(cur.timestamp.timestamp())

    # Use 61 1-minute sub-pixel Lagrangian Doppler rates (0..60m) directly when available
    exact_61_rates: Optional[List[float]] = (
        list(dop.minuteByMinuteRates[:61])
        if (dop and dop.inConusGrid and len(dop.minuteByMinuteRates) >= 61)
        else None
    )

    five_min_rates: List[float] = []
    if dop and dop.inConusGrid and len(dop.minuteRates60m) >= 13:
        five_min_rates = list(dop.minuteRates60m[:13])
    elif w_resp.nowcast and w_resp.nowcast.points:
        # Expand 15-min nowcast points to 5-min steps
        pts = w_resp.nowcast.points
        for m in range(0, 65, 5):
            idx = min(len(pts) - 1, m // 15)
            five_min_rates.append(float(pts[idx].precipIntensityMmHr))
    else:
        five_min_rates = [0.0] * 13

    # Build 61 minute-by-minute data points
    minutely_data: List[Dict[str, Any]] = []
    for minute in range(61):
        if exact_61_rates is not None:
            mmhr = max(0.0, float(exact_61_rates[minute]))
        else:
            seg = min(11, minute // 5)
            frac = (minute - seg * 5) / 5.0
            r0 = five_min_rates[seg]
            r1 = five_min_rates[seg + 1] if (seg + 1) < len(five_min_rates) else r0
            mu = (1.0 - math.cos(frac * math.pi)) / 2.0
            mmhr = max(0.0, r0 * (1.0 - mu) + r1 * mu)
        prob = min(1.0, round((mmhr / 2.5) if mmhr > 0.05 else 0.0, 2))
        pt_dict: Dict[str, Any] = {
            "time": now_ts + (minute * 60),
            "precipIntensity": _convert_precip_rate(mmhr, units_norm),
            "precipProbability": prob,
        }
        if mmhr > 0.05:
            pt_dict["precipIntensityError"] = _convert_precip_rate(mmhr * 0.15, units_norm)
            pt_dict["precipType"] = "snow" if cur.temperature <= 0.5 else "rain"
        minutely_data.append(pt_dict)

    curr_mmhr = max(
        float(dop.precipRateMmHr) if dop else 0.0,
        five_min_rates[0] if five_min_rates else 0.0,
    )
    nearest_storm_dist = (
        _convert_dist(dop.nearestEchoKm, units_norm)
        if (dop and dop.nearestEchoKm is not None)
        else None
    )
    nearest_storm_bearing = (
        _compass_to_degrees(dop.nearestEchoCompass)
        if (dop and dop.nearestEchoCompass)
        else None
    )

    payload: Dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "UTC",
        "offset": 0,
        "elevation": cur.elevationMeters or 0.0,
    }

    if "currently" not in excluded:
        currently_obj: Dict[str, Any] = {
            "time": now_ts,
            "summary": cur.condition.value,
            "icon": _darksky_icon(cur.condition),
            "precipIntensity": _convert_precip_rate(curr_mmhr, units_norm),
            "precipProbability": round((w_resp.hourly[0].precipitationChance / 100.0) if w_resp.hourly else 0.0, 2),
            "temperature": _convert_temp(cur.temperature, units_norm),
            "apparentTemperature": _convert_temp(cur.feelsLike, units_norm),
            "dewPoint": _convert_temp(cur.dewPoint if cur.dewPoint is not None else (cur.temperature - 4.0), units_norm),
            "humidity": round(cur.humidity / 100.0, 2),
            "pressure": round(cur.airPressure, 2),
            "windSpeed": _convert_wind(cur.windSpeed, units_norm),
            "windGust": _convert_wind(cur.windSpeed * 1.28, units_norm),
            "windBearing": cur.windDirection,
            "cloudCover": 0.75 if cur.condition != WeatherCondition.CLEAR else 0.12,
            "uvIndex": cur.uvIndex,
            "visibility": _convert_dist(cur.visibility, units_norm),
            "ozone": 310.0,
        }
        if curr_mmhr > 0.05:
            currently_obj["precipType"] = "snow" if cur.temperature <= 0.5 else "rain"
        if nearest_storm_dist is not None:
            currently_obj["nearestStormDistance"] = nearest_storm_dist
            currently_obj["nearestStormBearing"] = nearest_storm_bearing or 0
        payload["currently"] = currently_obj

    if "minutely" not in excluded:
        payload["minutely"] = {
            "summary": w_resp.nowcast.summaryText if w_resp.nowcast else "Clear for the hour.",
            "icon": "rain" if any(m["precipIntensity"] > 0 for m in minutely_data) else _darksky_icon(cur.condition),
            "data": minutely_data,
        }

    if "hourly" not in excluded:
        hourly_data: List[Dict[str, Any]] = []
        for h in w_resp.hourly:
            h_prob = round(h.precipitationChance / 100.0, 2)
            h_mmhr = round(h_prob * 1.8, 2) if h_prob >= 0.25 else 0.0
            h_item: Dict[str, Any] = {
                "time": int(h.time.timestamp()),
                "summary": h.condition.value,
                "icon": _darksky_icon(h.condition),
                "precipIntensity": _convert_precip_rate(h_mmhr, units_norm),
                "precipProbability": h_prob,
                "temperature": _convert_temp(h.temperature, units_norm),
                "apparentTemperature": _convert_temp(h.temperature, units_norm),
                "dewPoint": _convert_temp(cur.dewPoint if cur.dewPoint is not None else (h.temperature - 4.0), units_norm),
                "humidity": round(cur.humidity / 100.0, 2),
                "pressure": round(cur.airPressure, 2),
                "windSpeed": _convert_wind(cur.windSpeed, units_norm),
                "windGust": _convert_wind(cur.windSpeed * 1.25, units_norm),
                "windBearing": cur.windDirection,
                "cloudCover": 0.7 if h.condition != WeatherCondition.CLEAR else 0.15,
                "uvIndex": cur.uvIndex,
                "visibility": _convert_dist(cur.visibility, units_norm),
            }
            if h_mmhr > 0:
                h_item["precipType"] = "snow" if h.temperature <= 0.5 else "rain"
            hourly_data.append(h_item)

        payload["hourly"] = {
            "summary": f"Multi-model consensus ({cur.dominantModel}) for the next 24 hours.",
            "icon": _darksky_icon(cur.condition),
            "data": hourly_data,
        }

    if "daily" not in excluded:
        daily_data: List[Dict[str, Any]] = []
        for d in w_resp.daily:
            d_ts = int(d.date.timestamp())
            d_prob = round(d.precipitationChance / 100.0, 2)
            d_mmhr = round(d_prob * 2.2, 2) if d_prob >= 0.25 else 0.0
            t_min = _convert_temp(d.minTemperature, units_norm)
            t_max = _convert_temp(d.maxTemperature, units_norm)
            daily_data.append(
                {
                    "time": d_ts,
                    "summary": d.condition.value,
                    "icon": _darksky_icon(d.condition),
                    "sunriseTime": d_ts + 6 * 3600,
                    "sunsetTime": d_ts + 18 * 3600,
                    "moonPhase": 0.42,
                    "precipIntensity": _convert_precip_rate(d_mmhr, units_norm),
                    "precipIntensityMax": _convert_precip_rate(d_mmhr * 2.0, units_norm),
                    "precipProbability": d_prob,
                    "precipType": "snow" if d.maxTemperature <= 0.5 else "rain",
                    "temperatureHigh": t_max,
                    "temperatureHighTime": d_ts + 15 * 3600,
                    "temperatureLow": t_min,
                    "temperatureLowTime": d_ts + 6 * 3600,
                    "apparentTemperatureHigh": t_max,
                    "apparentTemperatureLow": t_min,
                    "temperatureMin": t_min,
                    "temperatureMax": t_max,
                    "dewPoint": _convert_temp(cur.dewPoint or (d.minTemperature - 2.0), units_norm),
                    "humidity": round(cur.humidity / 100.0, 2),
                    "pressure": round(cur.airPressure, 2),
                    "windSpeed": _convert_wind(cur.windSpeed, units_norm),
                    "windGust": _convert_wind(cur.windSpeed * 1.3, units_norm),
                    "windBearing": cur.windDirection,
                    "cloudCover": 0.5,
                    "uvIndex": cur.uvIndex,
                    "visibility": _convert_dist(cur.visibility, units_norm),
                }
            )
        payload["daily"] = {
            "summary": "10-day MOS bias-corrected multi-model outlook.",
            "icon": _darksky_icon(cur.condition),
            "data": daily_data,
        }

    if "alerts" not in excluded:
        payload["alerts"] = [
            {
                "title": a.event,
                "severity": a.severity.lower(),
                "time": now_ts,
                "expires": now_ts + 7200,
                "description": a.description or a.headline,
                "uri": "https://www.weather.gov",
            }
            for a in w_resp.alerts
        ]

    if "flags" not in excluded:
        payload["flags"] = {
            "sources": ["zus-conus-doppler-1km", "hrrr", "ecmwf", "gfs", "icon", "cwop-pws"],
            "nearest-station": cur.localStationDistanceKm or 1.5,
            "units": units_norm,
            "version": "ZüsAPI 2.0 (Dark Sky / Pirate Weather Standard)",
            "zus-consensus-score": cur.consensusScore,
            "zus-dominant-model": cur.dominantModel,
        }

    return payload


def _parse_lat_lon(coords: str) -> tuple[float, float]:
    try:
        parts = [p.strip() for p in coords.split(",")]
        if len(parts) != 2:
            raise ValueError("Expected 'lat,lon'")
        lat = float(parts[0])
        lon = float(parts[1])
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            raise ValueError("Coordinates out of bounds")
        return lat, lon
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid coordinates '{coords}': {e}")


@router.get(
    "/forecast/{api_key}/{coords}",
    summary="Dark Sky / Pirate Weather v1 Standard Drop-In Forecast Endpoint",
    tags=["Standardized Interop (Dark Sky / RainViewer / TileJSON)"],
)
async def get_darksky_forecast_with_key(
    api_key: str = Path(..., description="API key (any non-empty string accepted for local/self-hosted use)"),
    coords: str = Path(..., description="Comma-separated latitude,longitude (e.g. 40.7128,-74.0060)"),
    units: str = Query("us", description="Units system: us, si, ca, uk2, or auto"),
    exclude: Optional[str] = Query(None, description="Comma-separated blocks to exclude (currently,minutely,hourly,daily,alerts,flags)"),
):
    lat, lon = _parse_lat_lon(coords)
    return await _build_darksky_payload(lat=lat, lon=lon, units=units, exclude=exclude)


@router.get(
    "/api/v1/darksky/{coords}",
    summary="Dark Sky / Pirate Weather v1 Standard Endpoint (Keyless)",
    tags=["Standardized Interop (Dark Sky / RainViewer / TileJSON)"],
)
async def get_darksky_forecast_keyless(
    coords: str = Path(..., description="Comma-separated latitude,longitude (e.g. 40.7128,-74.0060)"),
    units: str = Query("us", description="Units system: us, si, ca, uk2, or auto"),
    exclude: Optional[str] = Query(None, description="Comma-separated blocks to exclude"),
):
    lat, lon = _parse_lat_lon(coords)
    return await _build_darksky_payload(lat=lat, lon=lon, units=units, exclude=exclude)


@router.get(
    "/public/weather-maps.json",
    summary="RainViewer v2 Standard Radar Manifest (Drop-in replacement for third-party radar cards)",
    tags=["Standardized Interop (Dark Sky / RainViewer / TileJSON)"],
)
@router.get(
    "/api/v1/radar/weather-maps.json",
    summary="RainViewer v2 Standard Radar Manifest",
    tags=["Standardized Interop (Dark Sky / RainViewer / TileJSON)"],
)
async def get_rainviewer_standard_manifest(request: Request):
    manifest = await doppler_service.get_manifest(base_url=str(request.base_url).rstrip("/"))
    base_url = str(request.base_url).rstrip("/")
    past_list = []
    nowcast_list = []
    for f in manifest.frames:
        item = {
            "time": f.timestamp,
            "path": f"/v2/radar/{f.frameId}",
        }
        if f.isFuture:
            nowcast_list.append(item)
        else:
            past_list.append(item)

    return {
        "version": "2.0",
        "generated": int(datetime.now(timezone.utc).timestamp()),
        "host": base_url,
        "radar": {
            "past": past_list,
            "nowcast": nowcast_list,
        },
        "satellite": {"infrared": []},
    }


@router.get(
    "/v2/radar/{frame_id}/{size}/{z}/{x}/{y}/{color}/{options}.png",
    summary="RainViewer v2 Standard Tile URL Pattern",
    tags=["Standardized Interop (Dark Sky / RainViewer / TileJSON)"],
    responses={200: {"content": {"image/png": {}}}},
)
async def get_rainviewer_standard_tile(
    frame_id: str,
    size: int,
    z: int,
    x: int,
    y: int,
    color: int,
    options: str,
):
    png_bytes = await doppler_service.get_tile_png(frame_id=frame_id, z=z, x=x, y=y)
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=86400, immutable",
            "Access-Control-Allow-Origin": "*",
        },
    )


@router.get(
    "/api/v1/radar/tilejson.json",
    summary="OSGeo TileJSON 3.0.0 Standard Endpoint (for Mapbox GL, MapLibre, QGIS)",
    tags=["Standardized Interop (Dark Sky / RainViewer / TileJSON)"],
)
async def get_tilejson_standard(
    request: Request,
    frame_id: str = Query("latest", description="Radar frame ID or 'latest'"),
):
    base_url = str(request.base_url).rstrip("/")
    return {
        "tilejson": "3.0.0",
        "name": "Züs 1km CONUS Doppler Radar & Nowcast",
        "description": "8-bit Züs Storm Palette Base Reflectivity (dBZ) + Optical Advection Nowcast",
        "version": "2.0.0",
        "attribution": "ZüsAPI / NOAA NEXRAD N0Q / IEM",
        "scheme": "xyz",
        "tiles": [f"{base_url}/api/v1/radar/tiles/{frame_id}/{{z}}/{{x}}/{{y}}.png"],
        "minzoom": 0,
        "maxzoom": 10,
        "bounds": [-126.0, 23.0, -65.0, 50.0],
        "center": [-95.5, 38.0, 5],
    }
