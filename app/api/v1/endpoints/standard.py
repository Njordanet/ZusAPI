import math
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
from fastapi import APIRouter, Path, Query, Request, Response, HTTPException

from app.models.weather import WeatherCondition, WeatherResponse
from app.services.weather_provider import weather_service
from app.services.doppler_service import doppler_service

router = APIRouter()


def _calc_moon_phase(dt: datetime) -> float:
    """
    Calculates astronomical synodic lunar phase in [0.0, 1.0):
      0.00 = New Moon
      0.25 = First Quarter
      0.50 = Full Moon
      0.75 = Last Quarter
    Reference New Moon: 2000-01-06 18:14 UTC (JD 2451550.26)
    Synodic month: 29.530588853 days
    """
    ref_new_moon_ts = 947182440.0  # 2000-01-06T18:14:00Z
    synodic_sec = 29.530588853 * 86400.0
    phase = ((dt.timestamp() - ref_new_moon_ts) % synodic_sec) / synodic_sec
    return round(phase, 2)


def _parse_iso_to_ts(iso_str: Optional[str], default_ts: int) -> int:
    if not iso_str:
        return default_ts
    try:
        s = iso_str.strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return default_ts


def _is_daytime_ts(ts: int, sunrise_ts: int, sunset_ts: int) -> bool:
    if sunset_ts > sunrise_ts:
        # Compare time-of-day modulo 86400 so hourly timestamps across days check against solar window
        tod = ts % 86400
        sr_tod = sunrise_ts % 86400
        ss_tod = sunset_ts % 86400
        if sr_tod < ss_tod:
            return sr_tod <= tod <= ss_tod
        return tod >= sr_tod or tod <= ss_tod
    return 6 <= ((ts // 3600) % 24) <= 18


def _darksky_icon(cond: WeatherCondition, is_day: bool = True, wind_kmh: float = 0.0) -> str:
    if cond in (WeatherCondition.CLEAR, WeatherCondition.PARTLY_CLOUDY) and wind_kmh >= 45.0:
        return "wind"
    if cond == WeatherCondition.CLEAR:
        return "clear-day" if is_day else "clear-night"
    if cond == WeatherCondition.PARTLY_CLOUDY:
        return "partly-cloudy-day" if is_day else "partly-cloudy-night"
    mapping = {
        WeatherCondition.OVERCAST: "cloudy",
        WeatherCondition.FOGGY: "fog",
        WeatherCondition.DRIZZLE: "rain",
        WeatherCondition.RAIN: "rain",
        WeatherCondition.HEAVY_RAIN: "rain",
        WeatherCondition.SNOW: "snow",
        WeatherCondition.THUNDERSTORM: "thunderstorm",
        WeatherCondition.SEVERE_THUNDERSTORM: "thunderstorm",
    }
    return mapping.get(cond, "clear-day" if is_day else "clear-night")


def _wmo_to_owm(wmo_code: int, is_day: bool = True) -> Dict[str, Any]:
    """Maps WMO weather code to standard OpenWeatherMap (OWM) condition id, main, description, and icon."""
    d_char = "d" if is_day else "n"
    table: Dict[int, Tuple[int, str, str, str]] = {
        0: (800, "Clear", "clear sky", f"01{d_char}"),
        1: (801, "Clouds", "few clouds", f"02{d_char}"),
        2: (802, "Clouds", "scattered clouds", f"03{d_char}"),
        3: (804, "Clouds", "overcast clouds", f"04{d_char}"),
        45: (741, "Fog", "fog", f"50{d_char}"),
        48: (741, "Fog", "depositing rime fog", f"50{d_char}"),
        51: (300, "Drizzle", "light intensity drizzle", f"09{d_char}"),
        53: (301, "Drizzle", "drizzle", f"09{d_char}"),
        55: (302, "Drizzle", "heavy intensity drizzle", f"09{d_char}"),
        61: (500, "Rain", "light rain", f"10{d_char}"),
        63: (501, "Rain", "moderate rain", f"10{d_char}"),
        65: (502, "Rain", "heavy intensity rain", f"10{d_char}"),
        71: (600, "Snow", "light snow", f"13{d_char}"),
        73: (601, "Snow", "snow", f"13{d_char}"),
        75: (602, "Snow", "heavy snow", f"13{d_char}"),
        77: (611, "Snow", "snow grains", f"13{d_char}"),
        80: (520, "Rain", "light intensity shower rain", f"09{d_char}"),
        81: (521, "Rain", "shower rain", f"09{d_char}"),
        82: (522, "Rain", "heavy intensity shower rain", f"09{d_char}"),
        85: (620, "Snow", "light shower snow", f"13{d_char}"),
        86: (621, "Snow", "shower snow", f"13{d_char}"),
        95: (200, "Thunderstorm", "thunderstorm", f"11{d_char}"),
        96: (201, "Thunderstorm", "thunderstorm with hail", f"11{d_char}"),
        99: (202, "Thunderstorm", "severe thunderstorm with heavy hail", f"11{d_char}"),
    }
    owm_id, main_str, desc, icon = table.get(wmo_code, (800, "Clear", "clear sky", f"01{d_char}"))
    return {"id": owm_id, "main": main_str, "description": desc, "icon": icon}


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
    if units in ("us", "imperial", "fahrenheit"):
        return round(c * 9.0 / 5.0 + 32.0, 2)
    if units == "standard":
        return round(c + 273.15, 2)
    return round(c, 2)


def _convert_wind(kmh: float, units: str) -> float:
    if units in ("us", "uk2", "imperial", "mph"):
        return round(kmh * 0.621371, 2)  # mph
    if units in ("si", "metric_owm", "ms", "standard"):
        return round(kmh / 3.6, 2)       # m/s
    if units == "kn":
        return round(kmh * 0.539957, 2)  # knots
    return round(kmh, 2)                 # ca / metric -> km/h


def _convert_dist(km: float, units: str) -> float:
    if units in ("us", "uk2", "imperial"):
        return round(km * 0.621371, 2)  # miles
    return round(km, 2)                 # km


def _convert_precip_rate(mmhr: float, units: str) -> float:
    if units in ("us", "imperial", "inch"):
        return round(mmhr / 25.4, 4)    # inches/hr
    return round(mmhr, 3)               # mm/hr


def _extract_61_minutely_rates(w_resp: WeatherResponse) -> List[float]:
    dop = w_resp.dopplerTelemetry
    if dop and dop.inConusGrid and len(dop.minuteByMinuteRates) >= 61:
        return [max(0.0, float(v)) for v in dop.minuteByMinuteRates[:61]]

    five_min_rates: List[float] = []
    if dop and dop.inConusGrid and len(dop.minuteRates60m) >= 13:
        five_min_rates = [float(v) for v in dop.minuteRates60m[:13]]
    elif w_resp.nowcast and w_resp.nowcast.points:
        pts = w_resp.nowcast.points
        for m in range(0, 65, 5):
            idx = min(len(pts) - 1, m // 15)
            five_min_rates.append(float(pts[idx].precipIntensityMmHr))
    else:
        five_min_rates = [0.0] * 13

    rates_61: List[float] = []
    for minute in range(61):
        seg = min(11, minute // 5)
        frac = (minute - seg * 5) / 5.0
        r0 = five_min_rates[seg]
        r1 = five_min_rates[seg + 1] if (seg + 1) < len(five_min_rates) else r0
        mu = (1.0 - math.cos(frac * math.pi)) / 2.0
        rates_61.append(round(max(0.0, r0 * (1.0 - mu) + r1 * mu), 3))
    return rates_61


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
    dop = w_resp.dopplerTelemetry
    cur = w_resp.current
    now_ts = int(cur.timestamp.timestamp())

    # Today's sunrise/sunset timestamps for accurate day/night icon selection
    sr_today_ts = (
        _parse_iso_to_ts(w_resp.daily[0].sunrise, now_ts - 6 * 3600)
        if w_resp.daily
        else now_ts - 6 * 3600
    )
    ss_today_ts = (
        _parse_iso_to_ts(w_resp.daily[0].sunset, now_ts + 6 * 3600)
        if w_resp.daily
        else now_ts + 6 * 3600
    )
    is_day_now = _is_daytime_ts(now_ts, sr_today_ts, ss_today_ts)

    rates_61 = _extract_61_minutely_rates(w_resp)
    minutely_data: List[Dict[str, Any]] = []
    for minute, mmhr in enumerate(rates_61):
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
        float(cur.precipIntensityMmHr or 0.0),
        float(dop.precipRateMmHr) if dop else 0.0,
        rates_61[0] if rates_61 else 0.0,
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
        "timezone": w_resp.timezone or "UTC",
        "offset": round(w_resp.utcOffsetSeconds / 3600.0, 2),
        "elevation": cur.elevationMeters or 0.0,
    }

    if "currently" not in excluded:
        currently_obj: Dict[str, Any] = {
            "time": now_ts,
            "summary": cur.condition.value,
            "icon": _darksky_icon(cur.condition, is_day=is_day_now, wind_kmh=cur.windSpeed),
            "precipIntensity": _convert_precip_rate(curr_mmhr, units_norm),
            "precipProbability": round((w_resp.hourly[0].precipitationChance / 100.0) if w_resp.hourly else 0.0, 2),
            "temperature": _convert_temp(cur.temperature, units_norm),
            "apparentTemperature": _convert_temp(cur.feelsLike, units_norm),
            "dewPoint": _convert_temp(cur.dewPoint if cur.dewPoint is not None else (cur.temperature - 4.0), units_norm),
            "humidity": round(cur.humidity / 100.0, 2),
            "pressure": round(cur.airPressure, 2),
            "windSpeed": _convert_wind(cur.windSpeed, units_norm),
            "windGust": _convert_wind(cur.windGust or (cur.windSpeed * 1.28), units_norm),
            "windBearing": cur.windDirection,
            "cloudCover": round((cur.cloudCover if cur.cloudCover is not None else 50) / 100.0, 2),
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
            "icon": "rain" if any(m["precipIntensity"] > 0 for m in minutely_data) else _darksky_icon(cur.condition, is_day=is_day_now),
            "data": minutely_data,
        }

    if "hourly" not in excluded:
        hourly_data: List[Dict[str, Any]] = []
        for h in w_resp.hourly:
            h_ts = int(h.time.timestamp())
            h_is_day = _is_daytime_ts(h_ts, sr_today_ts, ss_today_ts)
            h_prob = round(h.precipitationChance / 100.0, 2)
            h_mmhr = float(h.precipMm) if h.precipMm > 0 else (round(h_prob * 1.4, 2) if h_prob >= 0.35 else 0.0)
            h_feels = h.feelsLike if h.feelsLike is not None else h.temperature
            h_dew = h.dewPoint if h.dewPoint is not None else (h.temperature - 4.0)
            h_hum = h.humidity if h.humidity is not None else cur.humidity
            h_press = h.pressure if h.pressure is not None else cur.airPressure
            h_gust = h.windGusts if h.windGusts is not None else (h.windSpeed * 1.25)
            h_wdir = h.windDirection if h.windDirection is not None else cur.windDirection
            h_cloud = h.cloudCover if h.cloudCover is not None else (75 if h.condition != WeatherCondition.CLEAR else 15)
            h_uv = h.uvIndex if h.uvIndex is not None else (cur.uvIndex if h_is_day else 0.0)
            h_vis = h.visibilityKm if h.visibilityKm is not None else cur.visibility

            h_item: Dict[str, Any] = {
                "time": h_ts,
                "summary": h.condition.value,
                "icon": _darksky_icon(h.condition, is_day=h_is_day, wind_kmh=h.windSpeed),
                "precipIntensity": _convert_precip_rate(h_mmhr, units_norm),
                "precipProbability": h_prob,
                "temperature": _convert_temp(h.temperature, units_norm),
                "apparentTemperature": _convert_temp(h_feels, units_norm),
                "dewPoint": _convert_temp(h_dew, units_norm),
                "humidity": round(h_hum / 100.0, 2),
                "pressure": round(h_press, 2),
                "windSpeed": _convert_wind(h.windSpeed, units_norm),
                "windGust": _convert_wind(h_gust, units_norm),
                "windBearing": h_wdir,
                "cloudCover": round(h_cloud / 100.0, 2),
                "uvIndex": round(h_uv, 1),
                "visibility": _convert_dist(h_vis, units_norm),
            }
            if h_mmhr > 0:
                h_item["precipType"] = "snow" if h.temperature <= 0.5 else "rain"
            hourly_data.append(h_item)

        payload["hourly"] = {
            "summary": f"Multi-model consensus ({cur.dominantModel}) for the next {len(hourly_data)} hours.",
            "icon": _darksky_icon(cur.condition, is_day=is_day_now, wind_kmh=cur.windSpeed),
            "data": hourly_data,
        }

    if "daily" not in excluded:
        daily_data: List[Dict[str, Any]] = []
        for d_idx, d in enumerate(w_resp.daily):
            d_ts = int(d.date.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
            d_prob = round(d.precipitationChance / 100.0, 2)
            # Average hourly rate from daily sum (or probability fallback)
            d_mmhr = round(d.precipSumMm / 8.0, 2) if d.precipSumMm > 0 else (round(d_prob * 1.8, 2) if d_prob >= 0.30 else 0.0)
            t_min = _convert_temp(d.minTemperature, units_norm)
            t_max = _convert_temp(d.maxTemperature, units_norm)
            sr_ts = _parse_iso_to_ts(d.sunrise, d_ts + 6 * 3600)
            ss_ts = _parse_iso_to_ts(d.sunset, d_ts + 18 * 3600)

            # Slice corresponding 24h window from w_resp.hourly when available for true daily aggregates
            day_hours = w_resp.hourly[d_idx * 24 : (d_idx + 1) * 24]
            if day_hours:
                avg_hum = round(sum((h.humidity or cur.humidity) for h in day_hours) / len(day_hours) / 100.0, 2)
                avg_press = round(sum((h.pressure or cur.airPressure) for h in day_hours) / len(day_hours), 2)
                avg_cloud = round(sum((h.cloudCover or 50) for h in day_hours) / len(day_hours) / 100.0, 2)
                avg_dew_c = sum((h.dewPoint if h.dewPoint is not None else d.minTemperature) for h in day_hours) / len(day_hours)
                max_gust_kmh = max((h.windGusts or h.windSpeed * 1.25) for h in day_hours)
            else:
                avg_hum = round(cur.humidity / 100.0, 2)
                avg_press = round(cur.airPressure, 2)
                avg_cloud = 0.45
                avg_dew_c = d.minTemperature - 2.0
                max_gust_kmh = d.windSpeedMax * 1.3

            daily_data.append(
                {
                    "time": d_ts,
                    "summary": d.condition.value,
                    "icon": _darksky_icon(d.condition, is_day=True, wind_kmh=d.windSpeedMax),
                    "sunriseTime": sr_ts,
                    "sunsetTime": ss_ts,
                    "moonPhase": _calc_moon_phase(d.date),
                    "precipIntensity": _convert_precip_rate(d_mmhr, units_norm),
                    "precipIntensityMax": _convert_precip_rate(max(d_mmhr * 2.2, d.precipSumMm / 3.0), units_norm),
                    "precipAccumulation": round(d.precipSumMm / 25.4, 2) if units_norm == "us" else round(d.precipSumMm / 10.0, 2),
                    "precipProbability": d_prob,
                    "precipType": "snow" if d.maxTemperature <= 0.5 else "rain",
                    "temperatureHigh": t_max,
                    "temperatureHighTime": sr_ts + int((ss_ts - sr_ts) * 0.72),
                    "temperatureLow": t_min,
                    "temperatureLowTime": sr_ts - 1800,
                    "apparentTemperatureHigh": t_max,
                    "apparentTemperatureLow": t_min,
                    "temperatureMin": t_min,
                    "temperatureMax": t_max,
                    "dewPoint": _convert_temp(avg_dew_c, units_norm),
                    "humidity": avg_hum,
                    "pressure": avg_press,
                    "windSpeed": _convert_wind(d.windSpeedMax, units_norm),
                    "windGust": _convert_wind(max_gust_kmh, units_norm),
                    "windBearing": d.windDirectionDominant,
                    "cloudCover": avg_cloud,
                    "uvIndex": d.uvIndexMax,
                    "visibility": _convert_dist(cur.visibility, units_norm),
                }
            )
        payload["daily"] = {
            "summary": "10-day MOS bias-corrected multi-model outlook.",
            "icon": _darksky_icon(cur.condition, is_day=True),
            "data": daily_data,
        }

    if "alerts" not in excluded:
        payload["alerts"] = [
            {
                "title": a.event,
                "severity": a.severity.lower(),
                "time": now_ts,
                "expires": _parse_iso_to_ts(a.expires, now_ts + 7200),
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
            "version": "ZüsAPI 2.1 (Universal Dark Sky / Pirate Weather Standard)",
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
    tags=["Standardized Interop (Dark Sky / Open-Meteo / OWM / RainViewer / TileJSON)"],
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
    tags=["Standardized Interop (Dark Sky / Open-Meteo / OWM / RainViewer / TileJSON)"],
)
async def get_darksky_forecast_keyless(
    coords: str = Path(..., description="Comma-separated latitude,longitude (e.g. 40.7128,-74.0060)"),
    units: str = Query("us", description="Units system: us, si, ca, uk2, or auto"),
    exclude: Optional[str] = Query(None, description="Comma-separated blocks to exclude"),
):
    lat, lon = _parse_lat_lon(coords)
    return await _build_darksky_payload(lat=lat, lon=lon, units=units, exclude=exclude)


@router.get(
    "/v1/forecast",
    summary="Open-Meteo v1 Standard Drop-In Endpoint (serves Züs RLS-weighted & Doppler-fused forecast)",
    tags=["Standardized Interop (Dark Sky / Open-Meteo / OWM / RainViewer / TileJSON)"],
)
@router.get(
    "/api/v1/open-meteo/forecast",
    summary="Open-Meteo v1 Standard Drop-In Endpoint",
    tags=["Standardized Interop (Dark Sky / Open-Meteo / OWM / RainViewer / TileJSON)"],
)
async def get_open_meteo_standard_forecast(
    latitude: float = Query(..., ge=-90.0, le=90.0),
    longitude: float = Query(..., ge=-180.0, le=180.0),
    temperature_unit: str = Query("celsius", description="celsius or fahrenheit"),
    wind_speed_unit: str = Query("kmh", description="kmh, ms, mph, or kn"),
    precipitation_unit: str = Query("mm", description="mm or inch"),
    timeformat: str = Query("iso8601", description="iso8601 or unixtime"),
    timezone_param: str = Query("UTC", alias="timezone", description="UTC, auto, or IANA timezone"),
    forecast_days: int = Query(10, ge=1, le=16),
):
    """
    Universal drop-in replacement for Open-Meteo `/v1/forecast`.
    Third-party apps (Breezy Weather, Home Assistant, Geometric Weather) can point their
    Open-Meteo base URL to ZüsAPI to receive station-assimilated, RLS-weighted consensus data.
    """
    t_unit = temperature_unit.lower().strip()
    w_unit = wind_speed_unit.lower().strip()
    p_unit = precipitation_unit.lower().strip()
    use_unix = timeformat.lower().strip() == "unixtime"

    w_resp = await weather_service.get_forecast(
        latitude=latitude,
        longitude=longitude,
        city_name=f"{latitude:.3f},{longitude:.3f}",
    )
    cur = w_resp.current
    rates_61 = _extract_61_minutely_rates(w_resp)

    use_local_tz = timezone_param.lower().strip() not in ("utc", "gmt", "")
    tz_offset_sec = w_resp.utcOffsetSeconds if use_local_tz else 0
    tz_name = w_resp.timezone if use_local_tz else "GMT"

    def fmt_time(dt: datetime, date_only: bool = False) -> Any:
        if use_unix:
            return int(dt.timestamp())
        shifted = dt + timedelta(seconds=tz_offset_sec)
        if date_only:
            return shifted.strftime("%Y-%m-%d")
        return shifted.strftime("%Y-%m-%dT%H:%M")

    def conv_p(mm: float) -> float:
        return round(mm / 25.4, 3) if p_unit == "inch" else round(mm, 2)

    t_sym = "°F" if t_unit == "fahrenheit" else "°C"
    w_sym = {"mph": "mp/h", "ms": "m/s", "kn": "kn"}.get(w_unit, "km/h")
    p_sym = "inch" if p_unit == "inch" else "mm"

    # Build 15-minute array from fused 61-minute Doppler/NWP curve
    m15_times = [
        fmt_time(cur.timestamp.replace(second=0, microsecond=0) + timedelta(minutes=m))
        for m in (0, 15, 30, 45, 60)
    ]
    m15_precip = [conv_p(rates_61[m] / 4.0) for m in (0, 15, 30, 45, 60)]
    m15_codes = [cur.observedWeatherCode or 0] * 5

    hours_slice = w_resp.hourly[: min(len(w_resp.hourly), forecast_days * 24)]
    days_slice = w_resp.daily[: min(len(w_resp.daily), forecast_days)]

    return {
        "latitude": latitude,
        "longitude": longitude,
        "generationtime_ms": 0.42,
        "utc_offset_seconds": tz_offset_sec,
        "timezone": tz_name,
        "timezone_abbreviation": tz_name,
        "elevation": cur.elevationMeters or 0.0,
        "current_units": {
            "time": "unixtime" if use_unix else "iso8601",
            "interval": "seconds",
            "temperature_2m": t_sym,
            "relative_humidity_2m": "%",
            "apparent_temperature": t_sym,
            "precipitation": p_sym,
            "weather_code": "wmo code",
            "cloud_cover": "%",
            "pressure_msl": "hPa",
            "surface_pressure": "hPa",
            "wind_speed_10m": w_sym,
            "wind_direction_10m": "°",
            "wind_gusts_10m": w_sym,
        },
        "current": {
            "time": fmt_time(cur.timestamp),
            "interval": 300,
            "temperature_2m": _convert_temp(cur.temperature, t_unit),
            "relative_humidity_2m": cur.humidity,
            "apparent_temperature": _convert_temp(cur.feelsLike, t_unit),
            "dew_point_2m": _convert_temp(cur.dewPoint if cur.dewPoint is not None else cur.temperature - 4.0, t_unit),
            "precipitation": conv_p(cur.precipIntensityMmHr),
            "weather_code": cur.observedWeatherCode or 0,
            "cloud_cover": cur.cloudCover if cur.cloudCover is not None else 25,
            "pressure_msl": cur.airPressure,
            "surface_pressure": cur.airPressure,
            "wind_speed_10m": _convert_wind(cur.windSpeed, w_unit),
            "wind_direction_10m": cur.windDirection,
            "wind_gusts_10m": _convert_wind(cur.windGust or (cur.windSpeed * 1.25), w_unit),
            "uv_index": cur.uvIndex,
            "visibility": round(cur.visibility * 1000.0, 0),
            "cape": cur.capeJkg or 0.0,
        },
        "minutely_15": {
            "time": m15_times,
            "precipitation": m15_precip,
            "weather_code": m15_codes,
        },
        "hourly_units": {
            "time": "unixtime" if use_unix else "iso8601",
            "temperature_2m": t_sym,
            "relative_humidity_2m": "%",
            "apparent_temperature": t_sym,
            "precipitation_probability": "%",
            "precipitation": p_sym,
            "weather_code": "wmo code",
            "pressure_msl": "hPa",
            "cloud_cover": "%",
            "visibility": "m",
            "wind_speed_10m": w_sym,
            "wind_direction_10m": "°",
            "wind_gusts_10m": w_sym,
            "uv_index": "",
        },
        "hourly": {
            "time": [fmt_time(h.time) for h in hours_slice],
            "temperature_2m": [_convert_temp(h.temperature, t_unit) for h in hours_slice],
            "relative_humidity_2m": [h.humidity if h.humidity is not None else cur.humidity for h in hours_slice],
            "dew_point_2m": [_convert_temp(h.dewPoint if h.dewPoint is not None else h.temperature - 4.0, t_unit) for h in hours_slice],
            "apparent_temperature": [_convert_temp(h.feelsLike if h.feelsLike is not None else h.temperature, t_unit) for h in hours_slice],
            "precipitation_probability": [h.precipitationChance for h in hours_slice],
            "precipitation": [conv_p(h.precipMm) for h in hours_slice],
            "weather_code": [h.wmoCode for h in hours_slice],
            "pressure_msl": [h.pressure if h.pressure is not None else cur.airPressure for h in hours_slice],
            "cloud_cover": [h.cloudCover if h.cloudCover is not None else 25 for h in hours_slice],
            "visibility": [round((h.visibilityKm if h.visibilityKm is not None else cur.visibility) * 1000.0, 0) for h in hours_slice],
            "wind_speed_10m": [_convert_wind(h.windSpeed, w_unit) for h in hours_slice],
            "wind_direction_10m": [h.windDirection if h.windDirection is not None else cur.windDirection for h in hours_slice],
            "wind_gusts_10m": [_convert_wind(h.windGusts or (h.windSpeed * 1.25), w_unit) for h in hours_slice],
            "uv_index": [h.uvIndex if h.uvIndex is not None else 0.0 for h in hours_slice],
            "cape": [h.capeJkg if h.capeJkg is not None else 0.0 for h in hours_slice],
        },
        "daily_units": {
            "time": "unixtime" if use_unix else "iso8601",
            "weather_code": "wmo code",
            "temperature_2m_max": t_sym,
            "temperature_2m_min": t_sym,
            "precipitation_sum": p_sym,
            "precipitation_probability_max": "%",
            "wind_speed_10m_max": w_sym,
            "wind_direction_10m_dominant": "°",
            "uv_index_max": "",
        },
        "daily": {
            "time": [fmt_time(d.date, date_only=True) for d in days_slice],
            "weather_code": [d.wmoCode for d in days_slice],
            "temperature_2m_max": [_convert_temp(d.maxTemperature, t_unit) for d in days_slice],
            "temperature_2m_min": [_convert_temp(d.minTemperature, t_unit) for d in days_slice],
            "precipitation_sum": [conv_p(d.precipSumMm) for d in days_slice],
            "precipitation_probability_max": [d.precipitationChance for d in days_slice],
            "uv_index_max": [d.uvIndexMax for d in days_slice],
            "wind_speed_10m_max": [_convert_wind(d.windSpeedMax, w_unit) for d in days_slice],
            "wind_direction_10m_dominant": [d.windDirectionDominant for d in days_slice],
            "sunrise": [d.sunrise for d in days_slice],
            "sunset": [d.sunset for d in days_slice],
        },
    }


@router.get(
    "/data/3.0/onecall",
    summary="OpenWeatherMap (OWM) OneCall 3.0 Standard Drop-In Endpoint",
    tags=["Standardized Interop (Dark Sky / Open-Meteo / OWM / RainViewer / TileJSON)"],
)
@router.get(
    "/api/v1/owm/onecall",
    summary="OpenWeatherMap (OWM) OneCall 3.0 Standard Drop-In Endpoint",
    tags=["Standardized Interop (Dark Sky / Open-Meteo / OWM / RainViewer / TileJSON)"],
)
async def get_owm_onecall_standard(
    lat: float = Query(..., ge=-90.0, le=90.0),
    lon: float = Query(..., ge=-180.0, le=180.0),
    units: str = Query("metric", description="standard (Kelvin, m/s), metric (°C, m/s), or imperial (°F, mph)"),
    exclude: Optional[str] = Query(None, description="Comma-separated blocks to exclude (current,minutely,hourly,daily,alerts)"),
    appid: Optional[str] = Query(None, description="OWM API Key (ignored for local/self-hosted use)"),
):
    u_mode = units.lower().strip()
    if u_mode not in ("standard", "metric", "imperial"):
        u_mode = "metric"
    wind_mode = "imperial" if u_mode == "imperial" else "metric_owm"
    excluded = {s.strip().lower() for s in (exclude or "").split(",") if s.strip()}

    w_resp = await weather_service.get_forecast(
        latitude=lat, longitude=lon, city_name=f"{lat:.3f},{lon:.3f}"
    )
    cur = w_resp.current
    now_ts = int(cur.timestamp.timestamp())
    sr_today = _parse_iso_to_ts(w_resp.daily[0].sunrise, now_ts - 6 * 3600) if w_resp.daily else now_ts - 6 * 3600
    ss_today = _parse_iso_to_ts(w_resp.daily[0].sunset, now_ts + 6 * 3600) if w_resp.daily else now_ts + 6 * 3600
    is_day_now = _is_daytime_ts(now_ts, sr_today, ss_today)

    payload: Dict[str, Any] = {
        "lat": lat,
        "lon": lon,
        "timezone": w_resp.timezone or "UTC",
        "timezone_offset": w_resp.utcOffsetSeconds,
    }

    if "current" not in excluded:
        payload["current"] = {
            "dt": now_ts,
            "sunrise": sr_today,
            "sunset": ss_today,
            "temp": _convert_temp(cur.temperature, u_mode),
            "feels_like": _convert_temp(cur.feelsLike, u_mode),
            "pressure": round(cur.airPressure),
            "humidity": cur.humidity,
            "dew_point": _convert_temp(cur.dewPoint if cur.dewPoint is not None else cur.temperature - 4.0, u_mode),
            "uvi": cur.uvIndex,
            "clouds": cur.cloudCover if cur.cloudCover is not None else 25,
            "visibility": int(round(cur.visibility * 1000.0)),
            "wind_speed": _convert_wind(cur.windSpeed, wind_mode),
            "wind_deg": cur.windDirection,
            "wind_gust": _convert_wind(cur.windGust or (cur.windSpeed * 1.25), wind_mode),
            "weather": [_wmo_to_owm(cur.observedWeatherCode or 0, is_day=is_day_now)],
        }

    if "minutely" not in excluded:
        rates_61 = _extract_61_minutely_rates(w_resp)
        payload["minutely"] = [
            {"dt": now_ts + m * 60, "precipitation": round(mmhr, 2)}
            for m, mmhr in enumerate(rates_61)
        ]

    if "hourly" not in excluded:
        payload["hourly"] = [
            {
                "dt": int(h.time.timestamp()),
                "temp": _convert_temp(h.temperature, u_mode),
                "feels_like": _convert_temp(h.feelsLike if h.feelsLike is not None else h.temperature, u_mode),
                "pressure": round(h.pressure if h.pressure is not None else cur.airPressure),
                "humidity": h.humidity if h.humidity is not None else cur.humidity,
                "dew_point": _convert_temp(h.dewPoint if h.dewPoint is not None else h.temperature - 4.0, u_mode),
                "uvi": h.uvIndex if h.uvIndex is not None else 0.0,
                "clouds": h.cloudCover if h.cloudCover is not None else 25,
                "visibility": int(round((h.visibilityKm if h.visibilityKm is not None else cur.visibility) * 1000.0)),
                "wind_speed": _convert_wind(h.windSpeed, wind_mode),
                "wind_deg": h.windDirection if h.windDirection is not None else cur.windDirection,
                "wind_gust": _convert_wind(h.windGusts or (h.windSpeed * 1.25), wind_mode),
                "weather": [
                    _wmo_to_owm(
                        h.wmoCode,
                        is_day=_is_daytime_ts(int(h.time.timestamp()), sr_today, ss_today),
                    )
                ],
                "pop": round(h.precipitationChance / 100.0, 2),
                "rain": {"1h": round(h.precipMm, 2)} if h.precipMm > 0 and h.temperature > 0.5 else {},
                "snow": {"1h": round(h.precipMm, 2)} if h.precipMm > 0 and h.temperature <= 0.5 else {},
            }
            for h in w_resp.hourly[:48]
        ]

    if "daily" not in excluded:
        payload["daily"] = [
            {
                "dt": int(d.date.timestamp()),
                "sunrise": _parse_iso_to_ts(d.sunrise, int(d.date.timestamp()) + 6 * 3600),
                "sunset": _parse_iso_to_ts(d.sunset, int(d.date.timestamp()) + 18 * 3600),
                "moon_phase": _calc_moon_phase(d.date),
                "summary": d.condition.value,
                "temp": {
                    "day": _convert_temp(d.maxTemperature, u_mode),
                    "min": _convert_temp(d.minTemperature, u_mode),
                    "max": _convert_temp(d.maxTemperature, u_mode),
                    "night": _convert_temp(d.minTemperature, u_mode),
                    "eve": _convert_temp((d.maxTemperature + d.minTemperature) / 2.0, u_mode),
                    "morn": _convert_temp(d.minTemperature, u_mode),
                },
                "feels_like": {
                    "day": _convert_temp(d.maxTemperature, u_mode),
                    "night": _convert_temp(d.minTemperature, u_mode),
                    "eve": _convert_temp((d.maxTemperature + d.minTemperature) / 2.0, u_mode),
                    "morn": _convert_temp(d.minTemperature, u_mode),
                },
                "pressure": round(cur.airPressure),
                "humidity": cur.humidity,
                "wind_speed": _convert_wind(d.windSpeedMax, wind_mode),
                "wind_deg": d.windDirectionDominant,
                "weather": [_wmo_to_owm(d.wmoCode, is_day=True)],
                "clouds": 40 if d.wmoCode >= 2 else 10,
                "pop": round(d.precipitationChance / 100.0, 2),
                "rain": round(d.precipSumMm, 2) if d.precipSumMm > 0 else 0.0,
                "uvi": d.uvIndexMax,
            }
            for d in w_resp.daily
        ]

    if "alerts" not in excluded:
        payload["alerts"] = [
            {
                "sender_name": "NWS / ZüsAPI",
                "event": a.event,
                "start": now_ts,
                "end": _parse_iso_to_ts(a.expires, now_ts + 7200),
                "description": a.description or a.headline,
                "tags": [a.severity],
            }
            for a in w_resp.alerts
        ]

    return payload


@router.get(
    "/data/2.5/weather",
    summary="OpenWeatherMap (OWM) v2.5 Current Weather Drop-In Endpoint (for IoT displays, ESP32, Conky, widgets)",
    tags=["Standardized Interop (Dark Sky / Open-Meteo / OWM / RainViewer / TileJSON)"],
)
@router.get(
    "/api/v1/owm/weather",
    summary="OpenWeatherMap (OWM) v2.5 Current Weather Drop-In Endpoint",
    tags=["Standardized Interop (Dark Sky / Open-Meteo / OWM / RainViewer / TileJSON)"],
)
async def get_owm_v25_current_weather(
    lat: float = Query(..., ge=-90.0, le=90.0),
    lon: float = Query(..., ge=-180.0, le=180.0),
    units: str = Query("metric", description="standard, metric, or imperial"),
    appid: Optional[str] = Query(None, description="OWM API Key (ignored)"),
):
    u_mode = units.lower().strip()
    if u_mode not in ("standard", "metric", "imperial"):
        u_mode = "metric"
    wind_mode = "imperial" if u_mode == "imperial" else "metric_owm"

    w_resp = await weather_service.get_forecast(
        latitude=lat, longitude=lon, city_name=f"{lat:.3f},{lon:.3f}"
    )
    cur = w_resp.current
    now_ts = int(cur.timestamp.timestamp())
    sr_today = _parse_iso_to_ts(w_resp.daily[0].sunrise, now_ts - 6 * 3600) if w_resp.daily else now_ts - 6 * 3600
    ss_today = _parse_iso_to_ts(w_resp.daily[0].sunset, now_ts + 6 * 3600) if w_resp.daily else now_ts + 6 * 3600
    is_day_now = _is_daytime_ts(now_ts, sr_today, ss_today)

    return {
        "coord": {"lon": lon, "lat": lat},
        "weather": [_wmo_to_owm(cur.observedWeatherCode or 0, is_day=is_day_now)],
        "base": "stations",
        "main": {
            "temp": _convert_temp(cur.temperature, u_mode),
            "feels_like": _convert_temp(cur.feelsLike, u_mode),
            "temp_min": _convert_temp(cur.tempMin, u_mode),
            "temp_max": _convert_temp(cur.tempMax, u_mode),
            "pressure": round(cur.airPressure),
            "humidity": cur.humidity,
        },
        "visibility": int(round(cur.visibility * 1000.0)),
        "wind": {
            "speed": _convert_wind(cur.windSpeed, wind_mode),
            "deg": cur.windDirection,
            "gust": _convert_wind(cur.windGust or (cur.windSpeed * 1.25), wind_mode),
        },
        "clouds": {"all": cur.cloudCover if cur.cloudCover is not None else 25},
        "dt": now_ts,
        "sys": {
            "sunrise": sr_today,
            "sunset": ss_today,
        },
        "timezone": w_resp.utcOffsetSeconds,
        "name": cur.cityName,
        "cod": 200,
    }


@router.get(
    "/public/weather-maps.json",
    summary="RainViewer v2 Standard Radar Manifest (Drop-in replacement for third-party radar cards)",
    tags=["Standardized Interop (Dark Sky / Open-Meteo / OWM / RainViewer / TileJSON)"],
)
@router.get(
    "/api/v1/radar/weather-maps.json",
    summary="RainViewer v2 Standard Radar Manifest",
    tags=["Standardized Interop (Dark Sky / Open-Meteo / OWM / RainViewer / TileJSON)"],
)
async def get_rainviewer_standard_manifest(request: Request):
    base_url = str(request.base_url).rstrip("/")
    if request.headers.get("x-forwarded-proto") == "https" and base_url.startswith("http://"):
        base_url = "https://" + base_url[len("http://") :]
    manifest = await doppler_service.get_manifest(base_url=base_url)
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
    tags=["Standardized Interop (Dark Sky / Open-Meteo / OWM / RainViewer / TileJSON)"],
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
            "Cache-Control": "public, max-age=86400, s-maxage=86400, immutable",
            "CDN-Cache-Control": "public, max-age=86400, immutable",
            "Cloudflare-CDN-Cache-Control": "public, max-age=86400, immutable",
            "Access-Control-Allow-Origin": "*",
        },
    )


@router.get(
    "/api/v1/radar/tilejson.json",
    summary="OSGeo TileJSON 3.0.0 Standard Endpoint (for Mapbox GL, MapLibre, QGIS)",
    tags=["Standardized Interop (Dark Sky / Open-Meteo / OWM / RainViewer / TileJSON)"],
)
async def get_tilejson_standard(
    request: Request,
    frame_id: str = Query("latest", description="Radar frame ID or 'latest'"),
):
    base_url = str(request.base_url).rstrip("/")
    if request.headers.get("x-forwarded-proto") == "https" and base_url.startswith("http://"):
        base_url = "https://" + base_url[len("http://") :]
    return {
        "tilejson": "3.0.0",
        "name": "Züs 1km CONUS Doppler Radar & Nowcast",
        "description": "8-bit Züs Storm Palette Base Reflectivity (dBZ) + Optical Advection Nowcast",
        "version": "2.1.0",
        "attribution": "ZüsAPI / NOAA NEXRAD N0Q / IEM",
        "scheme": "xyz",
        "tiles": [f"{base_url}/api/v1/radar/tiles/{frame_id}/{{z}}/{{x}}/{{y}}.png"],
        "minzoom": 0,
        "maxzoom": 10,
        "bounds": [-126.0, 23.0, -65.0, 50.0],
        "center": [-95.5, 38.0, 5],
    }
