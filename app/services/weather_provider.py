import asyncio
import math
import re
import statistics
from collections import OrderedDict
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
from zoneinfo import ZoneInfo
import httpx
from app.config import settings
from app.models.weather import (
    WeatherDataModel,
    HourlyForecastModel,
    DailyForecastModel,
    WeatherCondition,
    WeatherResponse,
    ModelWeightDetail,
    WeatherAlertModel,
    NowcastSummary,
    NowcastPoint,
    RadarLocalTelemetry,
)
from app.services.db import db_manager, eval_diurnal_bias
from app.services.station_service import station_service, calc_relative_humidity
from app.services.doppler_service import doppler_service
from app.services.http_client import http_pool


def classify_aqi(aqi: Optional[int]) -> str:
    if aqi is None:
        return "Good"
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Sensitive"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def classify_dewpoint_comfort(dewpoint_c: float) -> str:
    # Convert to Fahrenheit for standard meteorological mugginess thresholds
    dp_f = dewpoint_c * 9.0 / 5.0 + 32.0
    if dp_f < 35:
        return "Very Dry & Crisp"
    if dp_f < 50:
        return "Crisp & Pleasant"
    if dp_f < 60:
        return "Optimal Comfort"
    if dp_f < 65:
        return "Slightly Humid"
    if dp_f < 70:
        return "Muggy"
    if dp_f < 75:
        return "Very Humid"
    return "Oppressive"


def calc_wet_bulb_c(temp_c: float, rh_pct: float) -> float:
    """
    Computes wet-bulb temperature (°C) using Stull's empirical formula (2011).
    Valid for RH between 1% and 100% and temperatures between -20°C and 50°C.
    """
    t = float(temp_c)
    rh = max(1.0, min(100.0, float(rh_pct)))
    tw = (
        t * math.atan(0.151977 * math.sqrt(rh + 8.313659))
        + math.atan(t + rh)
        - math.atan(rh - 1.676331)
        + 0.00391838 * (rh ** 1.5) * math.atan(0.023101 * rh)
        - 4.686035
    )
    return round(tw, 2)


def calc_snow_liquid_ratio(temp_c: float, precip_type: Optional[str] = None) -> float:
    """
    Computes temperature-dependent Kuchera/Cobb Snow-to-Liquid Ratio (SLR).
    ~10:1 near 0°C ramping to ~18:1 at -15°C, and 3:1 for sleet.
    """
    if precip_type == "sleet":
        return 3.0
    if precip_type == "rain":
        return 0.0

    t = float(temp_c)
    if t >= 0.5:
        slr = max(4.0, 10.0 - 3.0 * t)
    elif t >= -15.0:
        slr = 10.0 + (8.0 / 15.0) * (-t)
    elif t >= -25.0:
        slr = max(14.0, 18.0 - 0.3 * (-t - 15.0))
    else:
        slr = 14.0
    return round(float(slr), 1)


def classify_precip_type(
    temp_c: float,
    rh_pct: Optional[float] = None,
    wet_bulb_c: Optional[float] = None,
    wmo_code: Optional[int] = None,
    metar_wx: Optional[str] = None,
) -> str:
    """
    Classifies precipitation phase into 'rain', 'snow', or 'sleet' using:
    - METAR freezing/pellet tokens (FZRA, FZDZ, PL, IC)
    - WMO freezing/pellet codes (56, 57, 66, 67, 77)
    - Wet-bulb temperature Tw:
        Tw <= -0.4°C -> 'snow'
        -0.4°C < Tw <= +0.8°C -> 'sleet'
        Tw > +0.8°C -> 'rain'
    """
    if metar_wx:
        m = metar_wx.upper()
        if any(tok in m for tok in ("FZRA", "FZDZ", "PL", "IC")):
            return "sleet"
        if any(tok in m for tok in ("SN", "SG", "GS")):
            return "snow"

    if wmo_code in (56, 57, 66, 67, 77):
        return "sleet"

    if wet_bulb_c is None:
        if rh_pct is not None:
            wet_bulb_c = calc_wet_bulb_c(temp_c, rh_pct)
        else:
            wet_bulb_c = float(temp_c)

    if wet_bulb_c <= -0.4:
        return "snow"
    elif wet_bulb_c <= 0.8:
        return "sleet"
    else:
        if wmo_code in (71, 73, 75, 85, 86) and temp_c <= 2.0:
            return "snow"
        return "rain"


def calc_horizon_bias_scales(lead_hours: float) -> Tuple[float, float]:
    """
    Returns (mean_scale, harmonic_amp_scale) for learned station diurnal bias.
    Keeps 100% strength inside 0-36h, then tapers smoothly over 36-96h so a calm
    short-range regime bias does not distort Days 4-10 airmass shifts.
    """
    if lead_hours <= 36.0:
        return 1.0, 1.0
    if lead_hours >= 96.0:
        return 0.15, 0.0
    t = (lead_hours - 36.0) / 60.0
    return round(1.0 - 0.85 * t, 4), round(1.0 - t, 4)


def eval_decayed_diurnal_bias(
    b0: float, bsin: float, bcos: float, utc_hour: int, lead_hours: float
) -> float:
    mean_s, amp_s = calc_horizon_bias_scales(lead_hours)
    return eval_diurnal_bias(b0 * mean_s, bsin * amp_s, bcos * amp_s, utc_hour)


def _raw_model_series(
    raw_map: Dict[str, Any], key_base: str, suffix: str
) -> List[Any]:
    """Fetch a model-specific Open-Meteo series without falling back across models."""
    suffixed_key = f"{key_base}_{suffix}"
    if suffixed_key in raw_map:
        return raw_map.get(suffixed_key) or []
    if suffix == "best_match":
        return raw_map.get(key_base) or []
    return []


def blend_hourly_precip_and_snow(
    model_entries: List[Dict[str, Any]],
    total_w: float,
    blended_temp_c: float,
    blended_rh: float,
    blended_pop: int,
    top_code: int,
    lead_hours: float,
) -> Tuple[str, float]:
    """
    Compute hourly precip type and snow accumulation inside each individual model
    using its own bias-corrected T/RH before blending, eliminating cross-model artifacts.
    """
    blended_snow_cm = 0.0
    snow_model_w = 0.0
    sleet_model_w = 0.0
    rain_model_w = 0.0

    for entry in model_entries:
        w = float(entry["weight"])
        m_temp = float(entry["temp_c"])
        m_dew = float(entry["dew_c"])
        m_pr = float(entry["precip_mm"])
        m_pop = int(entry["precip_prob"])
        m_code = int(entry["wmo_code"])
        m_rh = (
            int(round(float(entry["rh_pct"])))
            if entry.get("rh_pct") is not None
            else calc_relative_humidity(m_temp, m_dew)
        )
        m_wb = calc_wet_bulb_c(m_temp, m_rh)
        m_pt = classify_precip_type(
            temp_c=m_temp, rh_pct=float(m_rh), wet_bulb_c=m_wb, wmo_code=m_code
        )
        m_slr = calc_snow_liquid_ratio(m_temp, m_pt)
        m_snow_cm = (m_pr / 10.0) * m_slr if (m_pt in ("snow", "sleet") and m_pr > 0.0) else 0.0
        blended_snow_cm += m_snow_cm * w

        if m_pr >= 0.05 or m_pop >= 25:
            if m_pt == "sleet":
                sleet_model_w += w
            elif m_pt == "snow":
                snow_model_w += w
            else:
                rain_model_w += w

    h_wb = calc_wet_bulb_c(blended_temp_c, blended_rh)
    if (snow_model_w + sleet_model_w + rain_model_w) > 0:
        if sleet_model_w >= snow_model_w and sleet_model_w >= rain_model_w:
            h_pt = "sleet"
        elif (
            snow_model_w > 0
            and rain_model_w > 0
            and total_w > 0
            and snow_model_w >= 0.20 * total_w
            and rain_model_w >= 0.20 * total_w
        ):
            h_pt = "sleet"
        elif snow_model_w > rain_model_w:
            h_pt = "snow"
        else:
            h_pt = "rain"
    else:
        h_pt = classify_precip_type(
            temp_c=blended_temp_c,
            rh_pct=float(blended_rh),
            wet_bulb_c=h_wb,
            wmo_code=top_code,
        )

    h_snow_cm = round(blended_snow_cm / total_w, 2) if total_w > 0 else 0.0
    if lead_hours > 48.0 and blended_pop < 22 and (snow_model_w + sleet_model_w) < 0.25 * max(total_w, 1e-6):
        h_snow_cm = 0.0

    return h_pt, h_snow_cm


def reconcile_daily_condition_and_snow(
    day_slice: List[HourlyForecastModel],
    d_idx: int,
    d_pop: int,
    d_psum: float,
    d_min: float,
    d_max: float,
    initial_day_code: int,
    tz_delta: timedelta,
) -> Tuple[int, str, float, float]:
    """
    Reconcile daily sky/precip condition code, precip type, confidence-gated snow
    accumulation, and SLR from the blended hourly slice for a single calendar day.
    """
    raw_snow_accum = round(sum((h.snowAccumulationCm or 0.0) for h in day_slice), 2)
    snow_allowed = (d_idx == 0 and (d_pop >= 20 or raw_snow_accum >= 0.25)) or (d_pop >= 25)
    d_snow_accum = raw_snow_accum if snow_allowed else 0.0

    sig_hours = [
        h
        for h in day_slice
        if (h.precipMm or 0.0) >= 0.08 or (h.precipitationChance or 0) >= 25
    ]
    sleet_h = sum(1 for h in sig_hours if h.precipType == "sleet")
    snow_h = sum(
        1
        for h in sig_hours
        if h.precipType == "snow" or (h.snowAccumulationCm or 0.0) >= 0.05
    )
    rain_h = sum(
        1
        for h in sig_hours
        if h.precipType == "rain" and (h.snowAccumulationCm or 0.0) < 0.05
    )

    if d_snow_accum >= 0.25:
        if sleet_h > snow_h or (rain_h >= 2 and snow_h >= 1):
            d_pt = "sleet"
        else:
            d_pt = "snow"
    elif sleet_h > 0 and d_pop >= 30:
        d_pt = "sleet"
    else:
        d_pt = "rain"

    wet_hours = [h.wmoCode for h in day_slice if h.wmoCode >= 51]
    allow_wet_icon = (d_pop >= 30) or (d_idx == 0 and d_pop >= 25 and len(wet_hours) >= 2)
    day_code = initial_day_code

    if allow_wet_icon and (wet_hours or initial_day_code >= 51):
        thunder_codes = [c for c in wet_hours if c >= 95]
        if thunder_codes and d_pop >= 40:
            day_code = max(thunder_codes)
        elif d_pt == "sleet":
            day_code = 67 if d_psum >= 5.0 else 66
        elif d_pt == "snow":
            day_code = 75 if d_snow_accum >= 8.0 else (73 if d_snow_accum >= 2.0 else 71)
        else:
            if d_psum >= 12.0:
                day_code = 65
            elif d_psum >= 4.0:
                day_code = 63
            elif d_psum >= 1.0 or d_pop >= 45:
                day_code = 61
            else:
                day_code = 51
    else:
        fog_hours = [h.wmoCode for h in day_slice if h.wmoCode in (45, 48)]
        if d_idx <= 1 and initial_day_code in (45, 48) and len(fog_hours) >= 3:
            day_code = max(fog_hours)
        else:
            cw_sum = 0.0
            cw_tot = 0.0
            for h in day_slice:
                is_daylight = (
                    h.isDay
                    if h.isDay is not None
                    else (7 <= (h.time + tz_delta).hour <= 17)
                )
                hw = 2.5 if is_daylight else 0.5
                cw_sum += float(h.cloudCover or 0) * hw
                cw_tot += hw
            eff_day_cloud = (cw_sum / cw_tot) if cw_tot > 0 else 0.0
            if eff_day_cloud >= 78.0:
                day_code = 3
            elif eff_day_cloud >= 26.0:
                day_code = 2
            else:
                day_code = 0

    if d_psum > 0 and d_snow_accum > 0:
        d_slr = round(d_snow_accum * 10.0 / d_psum, 1)
    else:
        d_slr = calc_snow_liquid_ratio((d_min + d_max) / 2.0, d_pt)

    return day_code, d_pt, d_snow_accum, d_slr


def calc_dobson_units(lat: float, surface_ozone_ugm3: Optional[float] = None) -> float:
    """
    Computes total column ozone in Dobson Units (DU), blending a latitude and
    seasonally modulated climatological baseline (260-360 DU) with real-time
    surface ozone measurements when available.
    """
    lat_rad = math.radians(lat)
    base_du = 290.0 + 45.0 * (math.sin(abs(lat_rad)) ** 2)
    doy = datetime.now(timezone.utc).timetuple().tm_yday
    if lat >= 0:
        phase = 2.0 * math.pi * (doy - 85) / 365.25
    else:
        phase = 2.0 * math.pi * (doy - 265) / 365.25
    amp = 30.0 * math.sin(abs(lat_rad))
    climatology_du = base_du + amp * math.cos(phase)

    if surface_ozone_ugm3 is not None and surface_ozone_ugm3 > 0:
        o3_delta = (float(surface_ozone_ugm3) - 60.0) * 0.22
        du = max(240.0, min(420.0, climatology_du + o3_delta))
    else:
        du = max(240.0, min(420.0, climatology_du))

    return round(float(du), 1)


def calc_local_sunrise_sunset_iso(
    lat: float, lon: float, local_date: datetime, utc_offset_sec: int
) -> Tuple[str, str]:
    """
    Computes astronomical local sunrise and sunset ISO strings ('YYYY-MM-DDTHH:MM')
    using NOAA solar ephemeris equations for the given latitude, longitude, local date,
    and UTC offset in seconds.
    """
    doy = local_date.timetuple().tm_yday
    gamma = (2.0 * math.pi / 365.0) * (doy - 1)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2.0 * gamma)
        - 0.040849 * math.sin(2.0 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2.0 * gamma)
        + 0.000907 * math.sin(2.0 * gamma)
        - 0.002697 * math.cos(3.0 * gamma)
        + 0.00148 * math.sin(3.0 * gamma)
    )
    lat_rad = math.radians(max(-89.5, min(89.5, lat)))
    zenith_rad = math.radians(90.833)
    cos_ha = (math.cos(zenith_rad) / (math.cos(lat_rad) * math.cos(decl))) - (
        math.tan(lat_rad) * math.tan(decl)
    )
    cos_ha = max(-1.0, min(1.0, cos_ha))
    ha_deg = math.degrees(math.acos(cos_ha))

    tz_min = utc_offset_sec / 60.0
    rise_min = int(round(720.0 - 4.0 * (lon + ha_deg) - eqtime + tz_min)) % 1440
    set_min = int(round(720.0 - 4.0 * (lon - ha_deg) - eqtime + tz_min)) % 1440
    if set_min <= rise_min:
        rise_min, set_min = 390, 1110

    date_prefix = local_date.strftime("%Y-%m-%d")
    sr_str = f"{date_prefix}T{rise_min // 60:02d}:{rise_min % 60:02d}"
    ss_str = f"{date_prefix}T{set_min // 60:02d}:{set_min % 60:02d}"
    return sr_str, ss_str


_CARDINAL_DEGREES = {
    "N": 0, "NNE": 23, "NE": 45, "ENE": 68,
    "E": 90, "ESE": 113, "SE": 135, "SSE": 158,
    "S": 180, "SSW": 203, "SW": 225, "WSW": 248,
    "W": 270, "WNW": 293, "NW": 315, "NNW": 338,
}


def _nws_short_forecast_to_wmo(short_forecast: Optional[str]) -> int:
    sf = str(short_forecast or "").lower()
    if "thunder" in sf:
        return 95
    if "heavy rain" in sf:
        return 65
    if any(tok in sf for tok in ("freezing", "sleet", "ice")):
        return 66
    if any(tok in sf for tok in ("snow", "flurr", "blizzard")):
        return 71
    if any(tok in sf for tok in ("rain", "shower")):
        return 61
    if "drizzle" in sf:
        return 51
    if any(tok in sf for tok in ("fog", "mist")):
        return 45
    if ("overcast" in sf or "cloudy" in sf) and "partly" not in sf:
        return 3
    if any(tok in sf for tok in ("partly", "mostly sunny", "mostly clear")):
        return 2
    return 0


def _parse_nws_wind_kmh(wind_str: Optional[str]) -> Tuple[float, float]:
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(wind_str or ""))]
    if not nums:
        return 10.0, 14.0
    factor = 1.0 if "km" in str(wind_str or "").lower() else 1.60934
    speed_kmh = round((sum(nums) / len(nums)) * factor, 1)
    max_kmh = round(max(nums) * factor, 1)
    gust_kmh = round(max(max_kmh * 1.25, speed_kmh * 1.35), 1)
    return speed_kmh, gust_kmh


def _calc_feels_like_c(temp_c: float, rh_pct: float, wind_kmh: float) -> float:
    if temp_c <= 10.0 and wind_kmh > 4.8:
        v_pow = wind_kmh ** 0.16
        wc = 13.12 + 0.6215 * temp_c - 11.37 * v_pow + 0.3965 * temp_c * v_pow
        return round(wc, 1)
    if temp_c >= 26.7 and rh_pct >= 40.0:
        t_f = temp_c * 9.0 / 5.0 + 32.0
        r = rh_pct
        hi_f = (
            -42.379 + 2.04901523 * t_f + 10.14333127 * r
            - 0.22475541 * t_f * r - 0.00683783 * t_f * t_f
            - 0.05481717 * r * r + 0.00122874 * t_f * t_f * r
            + 0.00085282 * t_f * r * r - 0.00000199 * t_f * t_f * r * r
        )
        return round((hi_f - 32.0) * 5.0 / 9.0, 1)
    return round(temp_c, 1)


class WeatherProviderService:
    def __init__(self):
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._max_cache_entries: int = 512
        self._inflight_locks: Dict[str, asyncio.Lock] = {}
        self._bg_tasks: set[asyncio.Task] = set()

    def _cache_key(self, lat: float, lon: float) -> str:
        return f"{lat:.2f}:{lon:.2f}"

    async def _fetch_open_meteo_ensemble(
        self,
        client: httpx.AsyncClient,
        lat: float,
        lon: float,
        include_past_24h: bool = False,
    ) -> Dict[str, Any]:
        in_conus = db_manager.is_conus_region(lat, lon)
        models_csv = (
            "best_match,ecmwf_ifs025,ncep_nbm_conus,gfs_seamless,icon_seamless,gem_seamless"
            if in_conus
            else "best_match,ecmwf_ifs025,gfs_seamless,icon_seamless,gem_seamless"
        )
        ensemble_params: Dict[str, Any] = {
            "latitude": lat,
            "longitude": lon,
            "hourly": (
                "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation_probability,"
                "precipitation,weather_code,wind_speed_10m,wind_direction_10m,wind_gusts_10m,"
                "uv_index,dew_point_2m,cloud_cover,visibility,pressure_msl,cape"
            ),
            "daily": (
                "weather_code,temperature_2m_max,temperature_2m_min,"
                "precipitation_probability_max,precipitation_sum,wind_speed_10m_max"
            ),
            "models": models_csv,
            "timezone": "UTC",
            "forecast_days": 10,
        }
        single_params: Dict[str, Any] = {
            "latitude": lat,
            "longitude": lon,
            "current": (
                "temperature_2m,relative_humidity_2m,apparent_temperature,dew_point_2m,"
                "weather_code,pressure_msl,surface_pressure,wind_speed_10m,wind_direction_10m,"
                "wind_gusts_10m,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,"
                "precipitation,cape,shortwave_radiation,vapour_pressure_deficit,is_day"
            ),
            "minutely_15": "precipitation,weather_code",
            "hourly": (
                "cloud_cover_low,cloud_cover_mid,cloud_cover_high,lifted_index,"
                "shortwave_radiation,vapour_pressure_deficit,freezing_level_height,is_day"
            ),
            "daily": "uv_index_max,wind_direction_10m_dominant,sunrise,sunset",
            "timezone": "UTC",
            "forecast_days": 10,
            "forecast_minutely_15": 8,  # Next 2 hours in 15-min intervals
        }
        if include_past_24h:
            ensemble_params["past_hours"] = 24
            single_params["past_hours"] = 24

        resp_ensemble, resp_single = await asyncio.gather(
            client.get(settings.open_meteo_forecast_url, params=ensemble_params),
            client.get(settings.open_meteo_forecast_url, params=single_params),
        )
        resp_ensemble.raise_for_status()
        raw_data = resp_ensemble.json()

        if resp_single.status_code == 200:
            single_data = resp_single.json()
            raw_data["current"] = single_data.get("current", {})
            raw_data["minutely_15"] = single_data.get("minutely_15", {})

            ens_hourly = raw_data.setdefault("hourly", {})
            for h_key, h_val in (single_data.get("hourly") or {}).items():
                if h_key != "time" or "time" not in ens_hourly:
                    ens_hourly[h_key] = h_val

            ens_daily = raw_data.setdefault("daily", {})
            for d_key, d_val in (single_data.get("daily") or {}).items():
                if d_key != "time" or "time" not in ens_daily:
                    ens_daily[d_key] = d_val

        return raw_data

    async def _fetch_air_quality_and_tz(
        self, client: httpx.AsyncClient, lat: float, lon: float
    ) -> Tuple[
        Optional[int],
        Optional[float],
        Optional[float],
        str,
        int,
        Optional[float],
        Optional[float],
        Optional[float],
    ]:
        default_offset = int(round(lon / 15.0)) * 3600
        default_ozone = calc_dobson_units(lat, None)
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "current": (
                    "us_aqi,pm2_5,ozone,alder_pollen,birch_pollen,"
                    "grass_pollen,mugwort_pollen,olive_pollen,ragweed_pollen"
                ),
                "timezone": "auto",
            }
            resp = await client.get(settings.open_meteo_air_quality_url, params=params, timeout=5.0)
            if resp.status_code == 200:
                body = resp.json()
                curr = body.get("current", {})
                aqi = curr.get("us_aqi")
                pm25 = curr.get("pm2_5")
                raw_ozone = curr.get("ozone")
                ozone_du = calc_dobson_units(lat, float(raw_ozone) if raw_ozone is not None else None)
                tz_name = str(body.get("timezone") or "UTC")
                utc_off = int(body.get("utc_offset_seconds") or default_offset)

                tree_vals = [
                    float(curr[k])
                    for k in ("alder_pollen", "birch_pollen", "olive_pollen")
                    if curr.get(k) is not None
                ]
                grass_vals = [float(curr["grass_pollen"])] if curr.get("grass_pollen") is not None else []
                weed_vals = [
                    float(curr[k])
                    for k in ("mugwort_pollen", "ragweed_pollen")
                    if curr.get(k) is not None
                ]
                tree_pollen = round(max(tree_vals), 1) if tree_vals else None
                grass_pollen = round(max(grass_vals), 1) if grass_vals else None
                weed_pollen = round(max(weed_vals), 1) if weed_vals else None

                return (
                    int(aqi) if aqi is not None else None,
                    round(float(pm25), 2) if pm25 is not None else None,
                    ozone_du,
                    tz_name,
                    utc_off,
                    tree_pollen,
                    grass_pollen,
                    weed_pollen,
                )
        except Exception:
            pass
        return None, None, default_ozone, "UTC", default_offset, None, None, None

    async def _fetch_nws_polygon_alerts(
        self, client: httpx.AsyncClient, lat: float, lon: float
    ) -> List[WeatherAlertModel]:
        if not db_manager.is_us_region(lat, lon):
            return []
        alerts: List[WeatherAlertModel] = []
        try:
            resp = await client.get(
                f"{settings.nws_base_url}/alerts/active",
                params={"point": f"{lat:.4f},{lon:.4f}"},
                timeout=5.0,
            )
            if resp.status_code == 200:
                features = resp.json().get("features", [])
                for f in features[:5]:
                    props = f.get("properties", {})
                    alerts.append(
                        WeatherAlertModel(
                            id=str(props.get("id") or f.get("id") or "alert"),
                            event=str(props.get("event") or "Weather Alert"),
                            headline=str(props.get("headline") or props.get("event") or ""),
                            severity=str(props.get("severity") or "Moderate"),
                            urgency=str(props.get("urgency") or "Expected"),
                            description=props.get("description"),
                            expires=props.get("expires"),
                        )
                    )
        except Exception:
            pass
        return alerts

    async def _run_verification_pipeline(
        self,
        cell_id: str,
        past_hourly_temps: Dict[datetime, float],
        model_hourly_streams: Dict[str, List[Dict[str, Any]]],
        ground_truth_temp: Optional[float],
        ground_truth_precip_active: bool,
        ground_truth_wind: Optional[float],
        ground_truth_precip_mmhr: Optional[float],
        radar_raw_mmhr: Optional[float],
    ) -> None:
        try:
            if past_hourly_temps:
                await db_manager.backfill_expired_snapshots(
                    cell_id=cell_id, past_hourly_temps=past_hourly_temps
                )
            await db_manager.save_forecast_snapshots_and_verify(
                cell_id=cell_id,
                model_forecasts=model_hourly_streams,
                ground_truth_temp_c=ground_truth_temp,
                ground_truth_precip_active=ground_truth_precip_active,
                ground_truth_wind_kmh=ground_truth_wind,
                ground_truth_precip_mmhr=ground_truth_precip_mmhr,
                radar_raw_mmhr=radar_raw_mmhr,
            )
        except Exception as e:
            print(f"⚠️ Async RLS verification warning for cell {cell_id}: {e}")

    def _build_nowcast(
        self,
        raw_data: Dict[str, Any],
        doppler: Optional[RadarLocalTelemetry] = None,
        station_composite: Optional[Dict[str, Any]] = None,
        radar_zr_factor: float = 1.0,
    ) -> NowcastSummary:
        minutely = raw_data.get("minutely_15", {})
        m15_times = minutely.get("time") or []
        precips = minutely.get("precipitation") or minutely.get("precipitation_best_match") or []

        # When past_hours=24 is set on Open-Meteo, minutely_15 starts 24h in the past (idx 96 = now)
        now_cutoff_str = (datetime.now(timezone.utc) - timedelta(minutes=15)).strftime("%Y-%m-%dT%H:%M")
        m15_start_idx = 0
        for i, t_str in enumerate(m15_times):
            if t_str >= now_cutoff_str:
                m15_start_idx = i
                break

        # Extract 5 15-minute NWP checkpoints (0, 15, 30, 45, 60m) in mm/hr
        m15_rates: List[float] = []
        for idx in range(5):
            p_idx = m15_start_idx + idx
            val = float(precips[p_idx]) * 4.0 if (p_idx < len(precips) and precips[p_idx] is not None) else 0.0
            m15_rates.append(max(0.0, val))

        # Ground-truth METAR precipitation rate (if a local station within 25km reports active rain/drizzle/snow)
        station_mmhr = (
            float(station_composite.get("precip_mmhr") or 0.0)
            if (station_composite and (station_composite.get("wx_station_dist_km") or 99.0) <= 25.0)
            else 0.0
        )
        station_checked_dry = bool(
            station_composite
            and (station_composite.get("closest_distance_km") or 99.0) <= 35.0
            and station_composite.get("wx_condition") is None
        )

        # Build 61 1-minute fused precipitation rates (0..60m) combining 1km Lagrangian Doppler + METAR + NWP
        has_conus_doppler = bool(doppler and doppler.inConusGrid and len(doppler.minuteByMinuteRates) >= 61)
        raw_61 = list(doppler.minuteByMinuteRates[:61]) if has_conus_doppler else [0.0] * 61

        fused_61: List[float] = []
        for t_min in range(61):
            seg = min(3, t_min // 15)
            frac = (t_min - seg * 15) / 15.0
            mu = (1.0 - math.cos(frac * math.pi)) / 2.0
            nwp_mmhr = m15_rates[seg] * (1.0 - mu) + m15_rates[seg + 1] * mu

            rad_mmhr = raw_61[t_min]
            # Assimilate sub-beam surface METAR precipitation over the first 20 minutes with exponential decay
            if station_mmhr > 0.0 and t_min <= 20:
                st_decay = math.exp(-t_min / 12.0)
                rad_mmhr = max(rad_mmhr, round(station_mmhr * st_decay, 2))

            if has_conus_doppler:
                if station_checked_dry and nwp_mmhr < 0.08 and (not doppler or doppler.currentDbz < 22.0):
                    # Nearby METAR station & NWP both report dry; suppress weak uncorroborated radar clutter/virga
                    blended = rad_mmhr * 0.35 if rad_mmhr >= 0.35 else 0.0
                else:
                    # Radar dominates minutes 0..35 (88%), gradually giving NWP more weight by minute 60 (72% radar / 28% NWP)
                    w_rad = 0.88 - 0.16 * (t_min / 60.0)
                    blended = max(rad_mmhr, rad_mmhr * w_rad + nwp_mmhr * (1.0 - w_rad))
            else:
                blended = max(rad_mmhr, nwp_mmhr)
            fused_61.append(round(blended, 2) if blended >= 0.04 else 0.0)

        if doppler:
            doppler.minuteByMinuteRates = fused_61
            doppler.minuteRates60m = [fused_61[t] for t in range(0, 61, 5)]
            if (doppler.inConusGrid or station_mmhr > 0.0) and station_mmhr > 0.0 and doppler.currentDbz < 18.0:
                doppler.currentDbz = 19.5 if station_mmhr >= 1.0 else 15.0
                doppler.precipRateMmHr = max(doppler.precipRateMmHr, station_mmhr)
                doppler.intensityCategory = "Light Rain" if station_mmhr >= 0.8 else "Light Drizzle"

        points: List[NowcastPoint] = []
        for idx in range(5):
            t_m = idx * 15
            b_mmhr = fused_61[t_m]
            prob = min(100, int(round(b_mmhr * 35))) if b_mmhr > 0.08 else 0
            points.append(
                NowcastPoint(
                    timeOffsetMinutes=t_m,
                    precipProbability=prob,
                    precipIntensityMmHr=b_mmhr,
                )
            )

        max_precip = max(fused_61) if fused_61 else 0.0
        peak_min = int(fused_61.index(max_precip)) if max_precip > 0.0 else 0
        trend_tag = ""
        if doppler and abs(doppler.echoTrendDbzPer10m) >= 1.5:
            trend_tag = ", intensifying" if doppler.echoTrendDbzPer10m > 0 else ", weakening"

        # Dark Sky 1-Minute Precision Summary Generator
        wet_threshold = 0.12
        if max_precip < 0.08:
            if doppler and doppler.nearestEchoKm is not None and doppler.nearestEchoKm <= 35.0:
                summary_text = (
                    f"Dry overhead — nearest radar echo ({doppler.maxDbzNearby:.0f} dBZ{trend_tag}) "
                    f"is {doppler.nearestEchoKm:.0f} km {doppler.nearestEchoCompass}, tracking {doppler.stormMotionCompass}."
                )
            else:
                summary_text = "Clear skies for the next 60 minutes — no precipitation echoes on Doppler radar."
        elif fused_61[0] >= wet_threshold:
            dbz_tag = f" ({doppler.currentDbz:.0f} dBZ{trend_tag})" if (doppler and doppler.currentDbz >= 12) else ""
            stop_min = None
            for m_i in range(1, 61):
                window = fused_61[m_i : min(61, m_i + 3)]
                if all(v < 0.08 for v in window):
                    stop_min = m_i
                    break
            intensity_word = (
                "Heavy rain" if fused_61[0] >= 4.0 else ("Moderate rain" if fused_61[0] >= 1.5 else "Light rain")
            )
            if stop_min is not None:
                summary_text = (
                    f"{intensity_word}{dbz_tag} ({fused_61[0]:.1f} mm/hr) — stopping in {stop_min} min."
                )
            elif peak_min >= 5 and max_precip >= fused_61[0] * 1.35:
                summary_text = (
                    f"{intensity_word}{dbz_tag} ({fused_61[0]:.1f} mm/hr) — peaking in {peak_min} min ({max_precip:.1f} mm/hr)."
                )
            else:
                summary_text = (
                    f"{intensity_word}{dbz_tag} ({fused_61[0]:.1f} mm/hr) continuing for the next hour."
                )
        else:
            start_min = next((m_i for m_i in range(1, 61) if fused_61[m_i] >= wet_threshold), None)
            if start_min is not None:
                stop_after_start = next(
                    (
                        m_i
                        for m_i in range(start_min + 2, 61)
                        if all(v < 0.08 for v in fused_61[m_i : min(61, m_i + 3)])
                    ),
                    None,
                )
                intensity_word = (
                    "Heavy rain" if max_precip >= 4.0 else ("Rain" if max_precip >= 1.2 else "Light rain")
                )
                if stop_after_start is not None:
                    duration_m = stop_after_start - start_min
                    summary_text = (
                        f"{intensity_word} starting in {start_min} min, lasting ~{duration_m} min "
                        f"(peak {max_precip:.1f} mm/hr{trend_tag})."
                    )
                else:
                    summary_text = (
                        f"{intensity_word} starting in {start_min} min "
                        f"(peaking at {max_precip:.1f} mm/hr in {peak_min} min{trend_tag})."
                    )
            else:
                summary_text = "Light mist or drizzle in the vicinity over the next hour."

        source_label = (
            "Züs 1km CONUS Lagrangian Doppler (1-Min Sub-Pixel) + ASOS/HRRR"
            if (doppler and doppler.inConusGrid)
            else "HRRR / 15-Min High-Res Radar Assimilation"
        )
        return NowcastSummary(
            summaryText=summary_text,
            source=source_label,
            points=points,
        )

    async def _assimilate_fallback_telemetry(
        self,
        latitude: float,
        longitude: float,
        now: datetime,
        cell_id: str,
        hourly_models: List[HourlyForecastModel],
        current_model: WeatherDataModel,
        station_composite: Optional[Dict[str, Any]],
        us_aqi: Optional[int],
        pm25_val: Optional[float],
        ozone_val: Optional[float],
        tree_pollen_val: Optional[float],
        grass_pollen_val: Optional[float],
        weed_pollen_val: Optional[float],
    ) -> Tuple[Optional[RadarLocalTelemetry], NowcastSummary]:
        rls_meta: Dict[str, Any] = {}
        try:
            rls_meta = await db_manager.get_cell_rls_metadata(cell_id)
        except Exception:
            rls_meta = {}

        h0 = hourly_models[0] if hourly_models else None
        est_temp_now = (
            float(station_composite["temperature_c"])
            if (station_composite and station_composite.get("temperature_c") is not None)
            else (h0.temperature if h0 else current_model.temperature)
        )
        est_dewp_now = (
            float(station_composite["dewpoint_c"])
            if (station_composite and station_composite.get("dewpoint_c") is not None)
            else (h0.dewPoint if (h0 and h0.dewPoint is not None) else (est_temp_now - 4.0))
        )
        st_checked_now = bool(
            station_composite and (station_composite.get("closest_distance_km") or 99.0) <= 35.0
        )
        st_wet_now = bool(station_composite and station_composite.get("wx_condition") is not None)
        nwp_precip_now = bool(
            h0
            and (
                (h0.precipMm or 0.0) >= 0.1
                or (h0.precipitationChance or 0) >= 25
                or h0.wmoCode >= 51
            )
        )
        radar_zr_factor = float(rls_meta.get("radar_zr_factor") or 1.0)
        doppler_telemetry = await doppler_service.get_point_telemetry(
            latitude,
            longitude,
            temp_c=est_temp_now,
            dewpoint_c=est_dewp_now,
            station_wet=st_wet_now,
            radar_zr_factor=radar_zr_factor,
            station_checked=st_checked_now,
            cloud_cover_low=(float(h0.cloudCoverLow) if (h0 and h0.cloudCoverLow is not None) else None),
            nwp_precip_active=nwp_precip_now,
        )
        nowcast_obj = self._build_nowcast(
            {"minutely_15": {}},
            doppler_telemetry,
            station_composite,
            radar_zr_factor=radar_zr_factor,
        )

        ensemble_temp = h0.temperature if h0 else current_model.temperature
        if station_composite and station_composite["closest_distance_km"] <= 35.0:
            dist = station_composite["closest_distance_km"]
            station_weight = max(0.35, min(0.75, 1.0 - (dist / 50.0)))
            final_temp = round(
                station_composite["temperature_c"] * station_weight
                + ensemble_temp * (1.0 - station_weight),
                1,
            )
            current_model.localStationName = station_composite["closest_station_name"]
            current_model.localStationDistanceKm = dist
            current_model.localStationWx = station_composite.get("wx_string")
        else:
            final_temp = round(ensemble_temp, 1)

        temp_delta_now = round(final_temp - ensemble_temp, 2)
        base_feels = (h0.feelsLike if (h0 and h0.feelsLike is not None) else current_model.feelsLike)
        final_feels = round(float(base_feels) + temp_delta_now, 1)
        dewpoint_c = round(float(est_dewp_now), 1)
        comfort_label = classify_dewpoint_comfort(dewpoint_c)

        min_curr_doppler_dbz = 24.0 if (st_checked_now and not st_wet_now) else 18.0
        if station_composite and station_composite.get("wx_condition") is not None:
            observed_cond = station_composite["wx_condition"]
            observed_wmo = int(station_composite.get("wx_wmo") or 61)
        elif (
            doppler_telemetry
            and doppler_telemetry.currentDbz >= min_curr_doppler_dbz
            and doppler_telemetry.precipRateMmHr >= 0.15
        ):
            if final_temp <= 0.5:
                observed_cond = WeatherCondition.SNOW
                observed_wmo = 71
            elif doppler_telemetry.currentDbz >= 35.0:
                observed_cond = WeatherCondition.HEAVY_RAIN
                observed_wmo = 65
            elif doppler_telemetry.currentDbz >= 20.0:
                observed_cond = WeatherCondition.RAIN
                observed_wmo = 61
            else:
                observed_cond = WeatherCondition.DRIZZLE
                observed_wmo = 51
        elif h0 is not None:
            observed_cond = h0.condition
            observed_wmo = h0.wmoCode
        else:
            observed_cond = current_model.condition
            observed_wmo = current_model.observedWeatherCode or 0

        wind_kmh = float(
            station_composite["wind_kmh"]
            if (station_composite and station_composite.get("wind_kmh") is not None)
            else (h0.windSpeed if h0 else current_model.windSpeed)
        )
        gust_kmh = float(
            station_composite["gust_kmh"]
            if (station_composite and station_composite.get("gust_kmh") is not None)
            else (h0.windGusts if (h0 and h0.windGusts is not None) else round(wind_kmh * 1.25, 1))
        )
        wind_dir = int(
            station_composite["wind_dir"]
            if (station_composite and station_composite.get("wind_dir") is not None)
            else (h0.windDirection if (h0 and h0.windDirection is not None) else current_model.windDirection)
        )
        msl_pressure = (
            (station_composite and station_composite.get("pressure_hpa"))
            or (h0.pressure if (h0 and h0.pressure is not None) else None)
            or current_model.airPressure
        )
        visibility_km = float(
            station_composite["visibility_km"]
            if (station_composite and station_composite.get("visibility_km") is not None)
            else (h0.visibilityKm if (h0 and h0.visibilityKm is not None) else current_model.visibility)
        )
        curr_hum = int(
            station_composite["humidity"]
            if (station_composite and station_composite.get("humidity") is not None)
            else (h0.humidity if (h0 and h0.humidity is not None) else calc_relative_humidity(final_temp, dewpoint_c))
        )
        curr_precip_mmhr = float(
            max(
                nowcast_obj.points[0].precipIntensityMmHr if nowcast_obj.points else 0.0,
                doppler_telemetry.precipRateMmHr if doppler_telemetry else 0.0,
                float(station_composite.get("precip_mmhr") or 0.0) if station_composite else 0.0,
            )
        )
        curr_wb = calc_wet_bulb_c(final_temp, float(curr_hum))
        curr_pt = classify_precip_type(
            temp_c=final_temp,
            rh_pct=float(curr_hum),
            wet_bulb_c=curr_wb,
            wmo_code=observed_wmo,
            metar_wx=current_model.localStationWx,
        )

        current_model.temperature = final_temp
        current_model.feelsLike = final_feels
        current_model.condition = observed_cond
        current_model.observedWeatherCode = observed_wmo
        current_model.tempMin = min(current_model.tempMin, final_temp)
        current_model.tempMax = max(current_model.tempMax, final_temp)
        current_model.humidity = curr_hum
        current_model.windSpeed = round(wind_kmh, 1)
        current_model.windGust = round(gust_kmh, 1)
        current_model.windDirection = wind_dir
        current_model.airPressure = round(float(msl_pressure), 1)
        current_model.visibility = round(visibility_km, 1)
        current_model.dewPoint = dewpoint_c
        current_model.comfortLevel = comfort_label
        current_model.precipIntensityMmHr = round(curr_precip_mmhr, 2)
        current_model.wetBulbTemperature = curr_wb
        current_model.precipType = curr_pt
        current_model.nowcastSummary = nowcast_obj.summaryText
        current_model.timestamp = now
        if us_aqi is not None:
            current_model.usAqi = us_aqi
            current_model.aqiCategory = classify_aqi(us_aqi)
        if pm25_val is not None:
            current_model.pm25 = pm25_val
        if ozone_val is not None:
            current_model.ozone = ozone_val
        if tree_pollen_val is not None:
            current_model.treePollen = tree_pollen_val
        if grass_pollen_val is not None:
            current_model.grassPollen = grass_pollen_val
        if weed_pollen_val is not None:
            current_model.weedPollen = weed_pollen_val

        if h0 is not None:
            h0.temperature = final_temp
            h0.tempMin = min(h0.tempMin if h0.tempMin is not None else final_temp, final_temp)
            h0.tempMax = max(h0.tempMax if h0.tempMax is not None else final_temp, final_temp)
            h0.feelsLike = final_feels
            h0.wmoCode = observed_wmo
            h0.condition = observed_cond
            h0.humidity = curr_hum
            h0.dewPoint = dewpoint_c
            h0.windSpeed = round(wind_kmh, 1)
            h0.windGusts = round(gust_kmh, 1)
            h0.windDirection = wind_dir
            h0.pressure = round(float(msl_pressure), 1)
            h0.visibilityKm = round(visibility_km, 2)
            h0.wetBulbTemperature = curr_wb
            h0.precipType = curr_pt
            if abs(temp_delta_now) >= 0.1:
                for decay_idx, decay_factor in ((1, 0.55), (2, 0.25)):
                    if decay_idx < len(hourly_models):
                        hk = hourly_models[decay_idx]
                        adj_delta = round(temp_delta_now * decay_factor, 1)
                        hk.temperature = round(hk.temperature + adj_delta, 1)
                        if hk.feelsLike is not None:
                            hk.feelsLike = round(hk.feelsLike + adj_delta, 1)

        return doppler_telemetry, nowcast_obj

    async def _refresh_cached_fallback(
        self,
        key: str,
        latitude: float,
        longitude: float,
        city_name: str,
        cell_id: str,
        now: datetime,
        station_composite: Optional[Dict[str, Any]],
        nearby_stations: List[Any],
        active_alerts: List[WeatherAlertModel],
        us_aqi: Optional[int],
        pm25_val: Optional[float],
        ozone_val: Optional[float],
        tree_pollen_val: Optional[float],
        grass_pollen_val: Optional[float],
        weed_pollen_val: Optional[float],
    ) -> Optional[WeatherResponse]:
        entry = self._cache.get(key)
        if not entry or (now - entry["cached_at"]).total_seconds() > 10800:
            return None

        cached_resp: WeatherResponse = entry["data"].model_copy(deep=True)
        now_hour = now.replace(minute=0, second=0, microsecond=0)
        future_hourly = [
            h
            for h in cached_resp.hourly
            if (h.time.astimezone(timezone.utc) if h.time.tzinfo else h.time.replace(tzinfo=timezone.utc)) >= now_hour
        ]
        if not future_hourly:
            return None
        cached_resp.hourly = future_hourly

        tz_delta = timedelta(seconds=cached_resp.utcOffsetSeconds or 0)
        local_today = (now + tz_delta).date()
        future_daily = [
            d
            for d in cached_resp.daily
            if ((d.date.astimezone(timezone.utc) if d.date.tzinfo else d.date.replace(tzinfo=timezone.utc)) + tz_delta).date() >= local_today
        ]
        if future_daily:
            cached_resp.daily = future_daily

        if city_name and city_name != "Unknown Location":
            cached_resp.current.cityName = city_name

        doppler_telemetry, nowcast_obj = await self._assimilate_fallback_telemetry(
            latitude=latitude,
            longitude=longitude,
            now=now,
            cell_id=cell_id,
            hourly_models=cached_resp.hourly,
            current_model=cached_resp.current,
            station_composite=station_composite,
            us_aqi=us_aqi,
            pm25_val=pm25_val,
            ozone_val=ozone_val,
            tree_pollen_val=tree_pollen_val,
            grass_pollen_val=grass_pollen_val,
            weed_pollen_val=weed_pollen_val,
        )
        cached_resp.nearbyStations = nearby_stations
        cached_resp.alerts = active_alerts
        cached_resp.dopplerTelemetry = doppler_telemetry
        cached_resp.nowcast = nowcast_obj
        cached_resp.generatedAt = now
        cached_resp.cached = True
        return cached_resp

    async def _fetch_nws_fallback_forecast(
        self,
        client: httpx.AsyncClient,
        key: str,
        latitude: float,
        longitude: float,
        city_name: str,
        cell_id: str,
        now: datetime,
        tz_name: str,
        utc_offset_sec: int,
        nearby_stations: List[Any],
        station_composite: Optional[Dict[str, Any]],
        active_alerts: List[WeatherAlertModel],
        us_aqi: Optional[int],
        pm25_val: Optional[float],
        ozone_val: Optional[float],
        tree_pollen_val: Optional[float],
        grass_pollen_val: Optional[float],
        weed_pollen_val: Optional[float],
    ) -> WeatherResponse:
        nws_headers = {"Accept": "application/geo+json"}
        pts_resp = await client.get(
            f"{settings.nws_base_url}/points/{latitude:.4f},{longitude:.4f}",
            headers=nws_headers,
            timeout=6.0,
        )
        pts_resp.raise_for_status()
        pts_props = pts_resp.json().get("properties", {})
        hourly_url = pts_props.get("forecastHourly")
        daily_url = pts_props.get("forecast")
        if not hourly_url:
            raise RuntimeError("NWS gridpoint forecastHourly URL unavailable")

        eff_tz = str(pts_props.get("timeZone") or tz_name or "UTC")
        eff_offset = utc_offset_sec
        try:
            tz_obj = ZoneInfo(eff_tz)
            off = datetime.now(tz_obj).utcoffset()
            if off is not None:
                eff_offset = int(off.total_seconds())
        except Exception:
            tz_obj = timezone(timedelta(seconds=eff_offset))

        async def _safe_daily_get() -> Optional[Dict[str, Any]]:
            if not daily_url:
                return None
            try:
                r = await client.get(daily_url, headers=nws_headers, timeout=6.5)
                if r.status_code == 200:
                    return r.json()
            except Exception:
                pass
            return None

        hourly_resp, daily_json = await asyncio.gather(
            client.get(hourly_url, headers=nws_headers, timeout=6.5),
            _safe_daily_get(),
        )
        hourly_resp.raise_for_status()
        hourly_json = hourly_resp.json()
        hourly_props = hourly_json.get("properties", {})
        periods = hourly_props.get("periods") or []
        if not periods:
            raise RuntimeError("NWS forecastHourly returned zero periods")

        elev_val = (hourly_props.get("elevation") or {}).get("value")
        target_elevation = float(elev_val) if elev_val is not None else 0.0
        if target_elevation > 5.0 and nearby_stations:
            nearby_stations, station_composite = await station_service.fetch_nearby_stations(
                lat=latitude,
                lon=longitude,
                target_elev_m=target_elevation,
            )

        now_hour_utc = now.replace(minute=0, second=0, microsecond=0)
        parsed_periods: List[Tuple[datetime, Dict[str, Any]]] = []
        for p in periods:
            raw_start = p.get("startTime")
            if not raw_start:
                continue
            try:
                p_dt = datetime.fromisoformat(str(raw_start))
                if p_dt.tzinfo is None:
                    p_dt = p_dt.replace(tzinfo=tz_obj)
                p_utc = p_dt.astimezone(timezone.utc)
                if p_utc >= now_hour_utc:
                    parsed_periods.append((p_utc, p))
            except Exception:
                continue
        if not parsed_periods and periods:
            last_p = periods[-1]
            parsed_periods.append((now_hour_utc, last_p))

        hourly_models: List[HourlyForecastModel] = []
        for p_utc, p in parsed_periods[:156]:
            raw_t = p.get("temperature")
            t_unit = str(p.get("temperatureUnit") or "F").upper()
            temp_c = (
                round((float(raw_t) - 32.0) * 5.0 / 9.0, 1)
                if (raw_t is not None and t_unit == "F")
                else round(float(raw_t if raw_t is not None else 15.0), 1)
            )
            pop_val = int((p.get("probabilityOfPrecipitation") or {}).get("value") or 0)
            dew_raw = (p.get("dewpoint") or {}).get("value")
            dew_c = round(float(dew_raw), 1) if dew_raw is not None else round(temp_c - 4.0, 1)
            rh_raw = (p.get("relativeHumidity") or {}).get("value")
            rh_pct = int(round(float(rh_raw))) if rh_raw is not None else calc_relative_humidity(temp_c, dew_c)
            wind_kmh, gust_kmh = _parse_nws_wind_kmh(p.get("windSpeed"))
            wdir_deg = _CARDINAL_DEGREES.get(str(p.get("windDirection") or "").upper(), 180)
            wmo_code = _nws_short_forecast_to_wmo(p.get("shortForecast"))
            if wmo_code >= 51 and pop_val < 20:
                pop_val = 30
            is_day_flag = bool(p.get("isDaytime")) if p.get("isDaytime") is not None else (6 <= (p_utc + timedelta(seconds=eff_offset)).hour < 20)
            cloud_cov = 90 if wmo_code >= 51 else (95 if wmo_code in (45, 48) else (88 if wmo_code == 3 else (45 if wmo_code == 2 else 10)))
            feels_c = _calc_feels_like_c(temp_c, float(rh_pct), wind_kmh)
            est_pr_mm = round(max(0.0, (pop_val / 100.0) * (1.2 if wmo_code >= 61 else 0.3)), 2) if wmo_code >= 51 else 0.0
            h_wb = calc_wet_bulb_c(temp_c, float(rh_pct))
            h_pt = classify_precip_type(temp_c=temp_c, rh_pct=float(rh_pct), wet_bulb_c=h_wb, wmo_code=wmo_code)
            h_snow_cm = round((est_pr_mm / 10.0) * calc_snow_liquid_ratio(temp_c, h_pt), 2) if (h_pt in ("snow", "sleet") and est_pr_mm > 0.0) else 0.0

            hourly_models.append(
                HourlyForecastModel(
                    time=p_utc,
                    temperature=temp_c,
                    tempMin=temp_c,
                    tempMax=temp_c,
                    feelsLike=feels_c,
                    feelsMin=feels_c,
                    feelsMax=feels_c,
                    condition=WeatherCondition.from_wmo_code(wmo_code),
                    wmoCode=wmo_code,
                    precipitationChance=max(0, min(100, pop_val)),
                    precipMm=est_pr_mm,
                    windSpeed=wind_kmh,
                    windGusts=gust_kmh,
                    windDirection=wdir_deg,
                    humidity=max(5, min(100, rh_pct)),
                    uvIndex=2.0 if is_day_flag else 0.0,
                    dewPoint=dew_c,
                    cloudCover=cloud_cov,
                    visibilityKm=16.0,
                    pressure=1013.2,
                    wetBulbTemperature=h_wb,
                    precipType=h_pt,
                    snowAccumulationCm=h_snow_cm,
                    isDay=is_day_flag,
                )
            )

        h0 = hourly_models[0]
        current_model = WeatherDataModel(
            cityName=city_name,
            temperature=h0.temperature,
            feelsLike=h0.feelsLike or h0.temperature,
            condition=h0.condition,
            tempMin=h0.temperature,
            tempMax=h0.temperature,
            humidity=h0.humidity or 65,
            windSpeed=h0.windSpeed,
            windGust=h0.windGusts,
            windDirection=h0.windDirection or 180,
            uvIndex=h0.uvIndex or 0.0,
            airPressure=h0.pressure or 1013.2,
            visibility=h0.visibilityKm or 16.0,
            timestamp=now,
            consensusScore=88,
            dominantModel="Regional Consensus",
            elevationMeters=round(target_elevation, 1),
            usAqi=us_aqi,
            aqiCategory=classify_aqi(us_aqi),
            pm25=pm25_val,
            dewPoint=h0.dewPoint,
            comfortLevel=classify_dewpoint_comfort(h0.dewPoint if h0.dewPoint is not None else h0.temperature - 4.0),
            cloudCover=h0.cloudCover,
            observedWeatherCode=h0.wmoCode,
            wetBulbTemperature=h0.wetBulbTemperature,
            precipType=h0.precipType,
            ozone=ozone_val,
            isDay=h0.isDay,
            treePollen=tree_pollen_val,
            grassPollen=grass_pollen_val,
            weedPollen=weed_pollen_val,
        )

        doppler_telemetry, nowcast_obj = await self._assimilate_fallback_telemetry(
            latitude=latitude,
            longitude=longitude,
            now=now,
            cell_id=cell_id,
            hourly_models=hourly_models,
            current_model=current_model,
            station_composite=station_composite,
            us_aqi=us_aqi,
            pm25_val=pm25_val,
            ozone_val=ozone_val,
            tree_pollen_val=tree_pollen_val,
            grass_pollen_val=grass_pollen_val,
            weed_pollen_val=weed_pollen_val,
        )

        tz_delta = timedelta(seconds=eff_offset)
        hourly_by_local_date: Dict[str, List[HourlyForecastModel]] = {}
        for hm in hourly_models:
            lk = (hm.time + tz_delta).strftime("%Y-%m-%d")
            hourly_by_local_date.setdefault(lk, []).append(hm)

        nws_daily_by_date: Dict[str, List[Dict[str, Any]]] = {}
        for dp in ((daily_json or {}).get("properties") or {}).get("periods") or []:
            raw_s = dp.get("startTime")
            if not raw_s:
                continue
            try:
                dt_d = datetime.fromisoformat(str(raw_s))
                if dt_d.tzinfo is None:
                    dt_d = dt_d.replace(tzinfo=tz_obj)
                lk = dt_d.astimezone(tz_obj).strftime("%Y-%m-%d")
                nws_daily_by_date.setdefault(lk, []).append(dp)
            except Exception:
                continue

        local_now = now + tz_delta
        daily_models: List[DailyForecastModel] = []
        for d_idx in range(7):
            local_day_dt = local_now + timedelta(days=d_idx)
            local_date_key = local_day_dt.strftime("%Y-%m-%d")
            day_date_utc = datetime(
                local_day_dt.year,
                local_day_dt.month,
                local_day_dt.day,
                12,
                0,
                0,
                tzinfo=timezone.utc,
            )
            day_slice = hourly_by_local_date.get(local_date_key, [])
            nws_day_periods = nws_daily_by_date.get(local_date_key, [])
            if not day_slice and not nws_day_periods:
                continue

            slice_temps = [h.temperature for h in day_slice]
            period_temps: List[float] = []
            period_pops: List[int] = []
            period_winds: List[float] = []
            period_codes: List[int] = []
            for np_item in nws_day_periods:
                nt = np_item.get("temperature")
                if nt is not None:
                    nu = str(np_item.get("temperatureUnit") or "F").upper()
                    period_temps.append(round((float(nt) - 32.0) * 5.0 / 9.0, 1) if nu == "F" else round(float(nt), 1))
                npop = (np_item.get("probabilityOfPrecipitation") or {}).get("value")
                if npop is not None:
                    period_pops.append(int(npop))
                nw_kmh, _ = _parse_nws_wind_kmh(np_item.get("windSpeed"))
                period_winds.append(nw_kmh)
                period_codes.append(_nws_short_forecast_to_wmo(np_item.get("shortForecast")))

            all_temps = slice_temps + ([current_model.temperature] if d_idx == 0 else []) + (period_temps if not slice_temps else [])
            if d_idx == 0 and period_temps:
                all_temps.extend(period_temps)
            d_min = round(min(all_temps), 1) if all_temps else current_model.temperature
            d_max = round(max(all_temps), 1) if all_temps else current_model.temperature
            d_pop = max([int(h.precipitationChance) for h in day_slice] + period_pops + [0])
            d_psum = round(sum((h.precipMm or 0.0) for h in day_slice), 2)
            d_wmax = round(max([h.windSpeed for h in day_slice] + period_winds + [10.0]), 1)
            d_wdir = day_slice[0].windDirection if (day_slice and day_slice[0].windDirection is not None) else current_model.windDirection
            initial_code = max([h.wmoCode for h in day_slice] + period_codes + [current_model.observedWeatherCode or 0])

            if day_slice:
                day_code, d_pt, d_snow_accum, d_slr = reconcile_daily_condition_and_snow(
                    day_slice=day_slice,
                    d_idx=d_idx,
                    d_pop=d_pop,
                    d_psum=d_psum,
                    d_min=d_min,
                    d_max=d_max,
                    initial_day_code=initial_code,
                    tz_delta=tz_delta,
                )
            else:
                day_code = initial_code if d_pop >= 30 or initial_code < 51 else 2
                d_pt = classify_precip_type(temp_c=(d_min + d_max) / 2.0, wmo_code=day_code)
                d_slr = calc_snow_liquid_ratio((d_min + d_max) / 2.0, d_pt)
                d_snow_accum = 0.0

            astro_sr, astro_ss = calc_local_sunrise_sunset_iso(
                latitude, longitude, local_day_dt, eff_offset
            )
            daily_models.append(
                DailyForecastModel(
                    date=day_date_utc,
                    condition=WeatherCondition.from_wmo_code(day_code),
                    wmoCode=day_code,
                    minTemperature=d_min,
                    maxTemperature=max(d_min, d_max),
                    precipitationChance=max(0, min(100, d_pop)),
                    precipSumMm=d_psum,
                    uvIndexMax=5.0,
                    windSpeedMax=d_wmax,
                    windDirectionDominant=d_wdir,
                    sunrise=astro_sr,
                    sunset=astro_ss,
                    precipType=d_pt,
                    snowAccumulationCm=d_snow_accum,
                    snowToLiquidRatio=d_slr,
                )
            )

        if daily_models:
            current_model.tempMin = min(daily_models[0].minTemperature, current_model.temperature)
            current_model.tempMax = max(daily_models[0].maxTemperature, current_model.temperature)

        result = WeatherResponse(
            current=current_model,
            hourly=hourly_models,
            daily=daily_models,
            modelWeights=[],
            nearbyStations=nearby_stations,
            alerts=active_alerts,
            nowcast=nowcast_obj,
            dopplerTelemetry=doppler_telemetry,
            regionMode="US_HYPERLOCAL",
            timezone=eff_tz,
            utcOffsetSeconds=eff_offset,
            source="ZüsAPI (Regional Consensus)",
            cached=False,
            generatedAt=now,
        )
        self._cache[key] = {
            "cached_at": now,
            "data": result,
        }
        while len(self._cache) > self._max_cache_entries:
            self._cache.popitem(last=False)
        return result

    async def get_forecast(
        self,
        latitude: float,
        longitude: float,
        city_name: str,
        sync_verify: bool = False,
    ) -> WeatherResponse:
        key = self._cache_key(latitude, longitude)
        now = datetime.now(timezone.utc)

        if key in self._cache:
            entry = self._cache[key]
            if (now - entry["cached_at"]).total_seconds() < settings.cache_ttl_seconds:
                self._cache.move_to_end(key)
                response: WeatherResponse = entry["data"]
                if city_name and city_name != "Unknown Location" and response.current.cityName != city_name:
                    cur_copy = response.current.model_copy(update={"cityName": city_name})
                    return response.model_copy(update={"cached": True, "current": cur_copy})
                return response.model_copy(update={"cached": True})

        lock = self._inflight_locks.setdefault(key, asyncio.Lock())
        async with lock:
            # Double-check cache after acquiring singleflight lock
            now = datetime.now(timezone.utc)
            if key in self._cache:
                entry = self._cache[key]
                if (now - entry["cached_at"]).total_seconds() < settings.cache_ttl_seconds:
                    self._cache.move_to_end(key)
                    response = entry["data"]
                    if city_name and city_name != "Unknown Location" and response.current.cityName != city_name:
                        cur_copy = response.current.model_copy(update={"cityName": city_name})
                        return response.model_copy(update={"cached": True, "current": cur_copy})
                    return response.model_copy(update={"cached": True})

            is_us = db_manager.is_us_region(latitude, longitude)
            in_conus = db_manager.is_conus_region(latitude, longitude)
            region_mode = "US_HYPERLOCAL" if is_us else "GLOBAL_ENSEMBLE"
            cell_id = db_manager.make_cell_id(latitude, longitude)

            # Fast O(1) check if this cell has expired snapshots needing past_hours=24 backfill
            need_past_24h = await db_manager.has_expired_unverified_snapshots(cell_id)

            # 1. Execute ALL upstream queries in parallel (Ensemble + AQI/TZ + NWS Alerts + Nearby METAR/PWS Stations)
            client = http_pool.get_client()

            async def _safe_ensemble() -> Tuple[Optional[Dict[str, Any]], Optional[Exception]]:
                try:
                    data = await self._fetch_open_meteo_ensemble(
                        client, latitude, longitude, include_past_24h=need_past_24h
                    )
                    return data, None
                except Exception as exc:
                    return None, exc

            ensemble_task = _safe_ensemble()
            aqi_task = self._fetch_air_quality_and_tz(client, latitude, longitude)
            alerts_task = self._fetch_nws_polygon_alerts(client, latitude, longitude)
            stations_task = station_service.fetch_nearby_stations(
                lat=latitude, lon=longitude, target_elev_m=0.0
            )
            cell_reg_task = db_manager.register_active_cell(
                lat=latitude, lon=longitude, city_name=city_name, elevation_m=0.0
            )

            (
                (raw_data, ensemble_err),
                (
                    us_aqi,
                    pm25_val,
                    ozone_val,
                    tz_name,
                    utc_offset_sec,
                    tree_pollen_val,
                    grass_pollen_val,
                    weed_pollen_val,
                ),
                active_alerts,
                (nearby_stations, station_composite),
                _,
            ) = await asyncio.gather(
                ensemble_task, aqi_task, alerts_task, stations_task, cell_reg_task
            )

            if raw_data is None:
                # Fallback Tier 1: Recent In-Memory Cache (<= 3 hours) enriched with live station + 1km Doppler radar
                cached_fallback = await self._refresh_cached_fallback(
                    key=key,
                    latitude=latitude,
                    longitude=longitude,
                    city_name=city_name,
                    cell_id=cell_id,
                    now=now,
                    station_composite=station_composite,
                    nearby_stations=nearby_stations,
                    active_alerts=active_alerts,
                    us_aqi=us_aqi,
                    pm25_val=pm25_val,
                    ozone_val=ozone_val,
                    tree_pollen_val=tree_pollen_val,
                    grass_pollen_val=grass_pollen_val,
                    weed_pollen_val=weed_pollen_val,
                )
                if cached_fallback is not None:
                    return cached_fallback

                # Fallback Tier 2: Official NOAA/NWS forecast + live METAR/Doppler when inside US
                if is_us:
                    try:
                        return await self._fetch_nws_fallback_forecast(
                            client=client,
                            key=key,
                            latitude=latitude,
                            longitude=longitude,
                            city_name=city_name,
                            cell_id=cell_id,
                            now=now,
                            tz_name=tz_name,
                            utc_offset_sec=utc_offset_sec,
                            nearby_stations=nearby_stations,
                            station_composite=station_composite,
                            active_alerts=active_alerts,
                            us_aqi=us_aqi,
                            pm25_val=pm25_val,
                            ozone_val=ozone_val,
                            tree_pollen_val=tree_pollen_val,
                            grass_pollen_val=grass_pollen_val,
                            weed_pollen_val=weed_pollen_val,
                        )
                    except Exception as nws_err:
                        print(f"⚠️ NWS fallback failed after upstream error ({ensemble_err}): {nws_err}")

                if ensemble_err is not None:
                    raise ensemble_err
                raise RuntimeError("Upstream forecast data unavailable")

            target_elevation = float(raw_data.get("elevation") or 0.0)
            if target_elevation > 5.0 and nearby_stations:
                # Re-normalize station lapse-rate adjustment in-memory (<0.05ms via cached sector)
                nearby_stations, station_composite = await station_service.fetch_nearby_stations(
                    lat=latitude,
                    lon=longitude,
                    target_elev_m=target_elevation,
                )

            # 2. Extract individual model hourly streams from Open-Meteo Multi-Model payload
            hourly_raw = raw_data.get("hourly", {})
            daily_raw = raw_data.get("daily", {})
            current_raw = raw_data.get("current", {})

            model_suffixes = {
                "HRRR": "best_match",
                "ECMWF": "ecmwf_ifs025",
                "GFS": "gfs_seamless",
                "ICON": "icon_seamless",
                "GEM": "gem_seamless",
            }
            if not is_us:
                model_suffixes.pop("HRRR", None)
            if in_conus and any(
                v is not None for v in (hourly_raw.get("temperature_2m_ncep_nbm_conus") or [])
            ):
                model_suffixes["NBM"] = "ncep_nbm_conus"

            times_list = hourly_raw.get("time", [])
            current_hour_str = now.strftime("%Y-%m-%dT%H:00")
            start_idx = 0
            for idx, t_str in enumerate(times_list):
                if t_str >= current_hour_str:
                    start_idx = idx
                    break

            wdir_hourly_arr = (
                hourly_raw.get("wind_direction_10m_best_match")
                or hourly_raw.get("wind_direction_10m")
                or []
            )
            rh_hourly_arr = (
                hourly_raw.get("relative_humidity_2m_best_match")
                or hourly_raw.get("relative_humidity_2m")
                or []
            )
            cape_hourly_arr = (
                hourly_raw.get("cape_best_match")
                or hourly_raw.get("cape")
                or []
            )
            li_hourly_arr = (
                hourly_raw.get("lifted_index_best_match")
                or hourly_raw.get("lifted_index")
                or []
            )
            sw_hourly_arr = (
                hourly_raw.get("shortwave_radiation_best_match")
                or hourly_raw.get("shortwave_radiation_ecmwf_ifs025")
                or hourly_raw.get("shortwave_radiation")
                or []
            )
            vpd_hourly_arr = (
                hourly_raw.get("vapour_pressure_deficit_best_match")
                or hourly_raw.get("vapour_pressure_deficit_ecmwf_ifs025")
                or hourly_raw.get("vapour_pressure_deficit")
                or []
            )
            flh_hourly_arr = (
                hourly_raw.get("freezing_level_height_best_match")
                or hourly_raw.get("freezing_level_height_ecmwf_ifs025")
                or hourly_raw.get("freezing_level_height")
                or []
            )
            isday_hourly_arr = (
                hourly_raw.get("is_day_best_match")
                or hourly_raw.get("is_day_ecmwf_ifs025")
                or hourly_raw.get("is_day")
                or []
            )
            cclow_hourly_arr = (
                hourly_raw.get("cloud_cover_low_best_match")
                or hourly_raw.get("cloud_cover_low_ecmwf_ifs025")
                or hourly_raw.get("cloud_cover_low")
                or []
            )
            ccmid_hourly_arr = (
                hourly_raw.get("cloud_cover_mid_best_match")
                or hourly_raw.get("cloud_cover_mid_ecmwf_ifs025")
                or hourly_raw.get("cloud_cover_mid")
                or []
            )
            cchigh_hourly_arr = (
                hourly_raw.get("cloud_cover_high_best_match")
                or hourly_raw.get("cloud_cover_high_ecmwf_ifs025")
                or hourly_raw.get("cloud_cover_high")
                or []
            )

            model_hourly_streams: Dict[str, List[Dict[str, Any]]] = {}
            for m_name, suffix in model_suffixes.items():
                t_arr = _raw_model_series(hourly_raw, "temperature_2m", suffix)
                fl_arr = _raw_model_series(hourly_raw, "apparent_temperature", suffix)
                p_arr = _raw_model_series(hourly_raw, "precipitation_probability", suffix)
                pr_arr = _raw_model_series(hourly_raw, "precipitation", suffix)
                w_arr = _raw_model_series(hourly_raw, "wind_speed_10m", suffix)
                wdir_arr = _raw_model_series(hourly_raw, "wind_direction_10m", suffix)
                wg_arr = _raw_model_series(hourly_raw, "wind_gusts_10m", suffix)
                uv_arr = _raw_model_series(hourly_raw, "uv_index", suffix)
                dp_arr = _raw_model_series(hourly_raw, "dew_point_2m", suffix)
                rh_arr = _raw_model_series(hourly_raw, "relative_humidity_2m", suffix)
                cc_arr = _raw_model_series(hourly_raw, "cloud_cover", suffix)
                vis_arr = _raw_model_series(hourly_raw, "visibility", suffix)
                pr_msl_arr = _raw_model_series(hourly_raw, "pressure_msl", suffix)
                cape_arr = _raw_model_series(hourly_raw, "cape", suffix)
                c_arr = _raw_model_series(hourly_raw, "weather_code", suffix)

                stream: List[Dict[str, Any]] = []
                max_offsets = max(25, len(times_list) - start_idx)
                for offset in range(max_offsets):
                    idx = start_idx + offset
                    if idx >= len(t_arr):
                        break
                    temp_val = t_arr[idx]
                    if temp_val is None:
                        continue
                    w_val = float(w_arr[idx]) if idx < len(w_arr) and w_arr[idx] is not None else 0.0
                    stream.append(
                        {
                            "time": now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=offset),
                            "temp_c": float(temp_val),
                            "feels_c": float(fl_arr[idx]) if idx < len(fl_arr) and fl_arr[idx] is not None else float(temp_val),
                            "precip_prob": int(p_arr[idx]) if idx < len(p_arr) and p_arr[idx] is not None else 0,
                            "precip_mm": float(pr_arr[idx]) if idx < len(pr_arr) and pr_arr[idx] is not None else 0.0,
                            "wind_kmh": w_val,
                            "wind_dir": float(wdir_arr[idx]) if idx < len(wdir_arr) and wdir_arr[idx] is not None else None,
                            "gust_kmh": float(wg_arr[idx]) if idx < len(wg_arr) and wg_arr[idx] is not None else round(w_val * 1.3, 1),
                            "uv_index": float(uv_arr[idx]) if idx < len(uv_arr) and uv_arr[idx] is not None else None,
                            "dew_c": float(dp_arr[idx]) if idx < len(dp_arr) and dp_arr[idx] is not None else float(temp_val) - 4.0,
                            "rh_pct": float(rh_arr[idx]) if idx < len(rh_arr) and rh_arr[idx] is not None else None,
                            "cloud_pct": int(cc_arr[idx]) if idx < len(cc_arr) and cc_arr[idx] is not None else 0,
                            "vis_km": round(float(vis_arr[idx]) / 1000.0, 2) if idx < len(vis_arr) and vis_arr[idx] is not None else None,
                            "press_hpa": float(pr_msl_arr[idx]) if idx < len(pr_msl_arr) and pr_msl_arr[idx] is not None else None,
                            "cape_jkg": float(cape_arr[idx]) if idx < len(cape_arr) and cape_arr[idx] is not None else None,
                            "wmo_code": int(c_arr[idx]) if idx < len(c_arr) and c_arr[idx] is not None else 0,
                        }
                    )
                if stream:
                    model_hourly_streams[m_name] = stream

            # 3. Query cell RLS metadata & weights in parallel, plus 1km Doppler telemetry
            rls_meta, db_weights = await asyncio.gather(
                db_manager.get_cell_rls_metadata(cell_id),
                db_manager.get_model_weights(cell_id),
            )

            est_temp_now = (
                station_composite["temperature_c"]
                if station_composite and station_composite.get("temperature_c") is not None
                else float(current_raw.get("temperature_2m") or current_raw.get("temperature_2m_best_match") or 15.0)
            )
            est_dewp_now = (
                station_composite["dewpoint_c"]
                if station_composite and station_composite.get("dewpoint_c") is not None
                else float(current_raw.get("dew_point_2m") or current_raw.get("dew_point_2m_best_match") or (est_temp_now - 4.0))
            )
            st_checked_now = bool(
                station_composite and (station_composite.get("closest_distance_km") or 99.0) <= 35.0
            )
            st_wet_now = bool(station_composite and station_composite.get("wx_condition") is not None)
            est_cclow_now = (
                float(cclow_hourly_arr[start_idx])
                if (start_idx < len(cclow_hourly_arr) and cclow_hourly_arr[start_idx] is not None)
                else None
            )
            nwp_precip_now = any(
                (
                    s[0].get("precip_mm", 0.0) >= 0.1
                    or s[0].get("precip_prob", 0) >= 25
                    or s[0].get("wmo_code", 0) >= 51
                )
                for s in model_hourly_streams.values()
                if s
            )
            doppler_telemetry = await doppler_service.get_point_telemetry(
                latitude,
                longitude,
                temp_c=est_temp_now,
                dewpoint_c=est_dewp_now,
                station_wet=st_wet_now,
                radar_zr_factor=float(rls_meta.get("radar_zr_factor") or 1.0),
                station_checked=st_checked_now,
                cloud_cover_low=est_cclow_now,
                nwp_precip_active=nwp_precip_now,
            )

            # 4. Schedule or Run RLS Verification & Snapshot Persistence
            past_hourly_temps: Dict[datetime, float] = {}
            if need_past_24h and start_idx > 0:
                best_past_arr = (
                    hourly_raw.get("temperature_2m_best_match")
                    or hourly_raw.get("temperature_2m_ecmwf_ifs025")
                    or hourly_raw.get("temperature_2m")
                    or []
                )
                for p_idx in range(min(start_idx, len(times_list), len(best_past_arr))):
                    t_val = best_past_arr[p_idx]
                    if t_val is not None:
                        try:
                            dt_parsed = datetime.strptime(times_list[p_idx], "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
                            past_hourly_temps[dt_parsed] = float(t_val)
                        except Exception:
                            pass

            ground_truth_temp = (
                station_composite["temperature_c"]
                if station_composite and station_composite["closest_distance_km"] <= 45.0
                else None
            )
            ground_truth_wind = (
                float(station_composite["wind_kmh"])
                if (station_composite and station_composite.get("wind_kmh") is not None and station_composite["closest_distance_km"] <= 45.0)
                else None
            )
            ground_truth_precip_mmhr = (
                float(station_composite.get("precip_mmhr") or 0.0)
                if (station_composite and station_composite["closest_distance_km"] <= 35.0)
                else None
            )
            radar_raw_mmhr = (
                float(doppler_telemetry.precipRateMmHr)
                if (doppler_telemetry and doppler_telemetry.inConusGrid)
                else None
            )
            ground_truth_precip_active = bool(
                (station_composite and station_composite.get("wx_condition") is not None)
                or (
                    doppler_telemetry
                    and doppler_telemetry.currentDbz >= (24.0 if (st_checked_now and not st_wet_now) else 18.0)
                    and doppler_telemetry.precipRateMmHr >= 0.15
                )
            )

            if sync_verify:
                await self._run_verification_pipeline(
                    cell_id=cell_id,
                    past_hourly_temps=past_hourly_temps,
                    model_hourly_streams=model_hourly_streams,
                    ground_truth_temp=ground_truth_temp,
                    ground_truth_precip_active=ground_truth_precip_active,
                    ground_truth_wind=ground_truth_wind,
                    ground_truth_precip_mmhr=ground_truth_precip_mmhr,
                    radar_raw_mmhr=radar_raw_mmhr,
                )
                db_weights = await db_manager.get_model_weights(cell_id)
            else:
                bg_task = asyncio.create_task(
                    self._run_verification_pipeline(
                        cell_id=cell_id,
                        past_hourly_temps=past_hourly_temps,
                        model_hourly_streams=model_hourly_streams,
                        ground_truth_temp=ground_truth_temp,
                        ground_truth_precip_active=ground_truth_precip_active,
                        ground_truth_wind=ground_truth_wind,
                        ground_truth_precip_mmhr=ground_truth_precip_mmhr,
                        radar_raw_mmhr=radar_raw_mmhr,
                    )
                )
                self._bg_tasks.add(bg_task)
                bg_task.add_done_callback(self._bg_tasks.discard)

            nowcast_obj = self._build_nowcast(
                raw_data,
                doppler_telemetry,
                station_composite,
                radar_zr_factor=float(rls_meta.get("radar_zr_factor") or 1.0),
            )

            # 5. Prepare RLS horizon weights & learned Diurnal Harmonic MOS bias
            weights_map: Dict[str, Dict[str, Any]] = {}
            model_weight_details: List[ModelWeightDetail] = []

            for w_row in db_weights:
                m = w_row["model_name"]
                if m in model_hourly_streams:
                    weights_map[m] = w_row
                    b0 = float(w_row.get("bias_temp_c") or 0.0)
                    bsin = float(w_row.get("bias_sin_c") or 0.0)
                    bcos = float(w_row.get("bias_cos_c") or 0.0)
                    diurnal_now = round(eval_diurnal_bias(b0, bsin, bcos, now.hour), 2)
                    m_stream0 = model_hourly_streams[m][0] if model_hourly_streams[m] else None
                    m_wind_bias = float(w_row.get("bias_wind_kmh") or 0.0) * 0.4
                    m_cur_temp = round(float(m_stream0["temp_c"]) - diurnal_now, 1) if m_stream0 else None
                    m_cur_pop = int(m_stream0.get("precip_prob", 0)) if m_stream0 else None
                    m_cur_wind = round(max(0.0, float(m_stream0.get("wind_kmh", 0.0)) - m_wind_bias), 1) if m_stream0 else None
                    model_weight_details.append(
                        ModelWeightDetail(
                            modelName=m,
                            weight=round(float(w_row["dynamic_weight"]), 3),
                            weightShort=round(float(w_row.get("weight_short") or w_row["dynamic_weight"]), 3),
                            weightMed=round(float(w_row.get("weight_med") or w_row["dynamic_weight"]), 3),
                            weightExt=round(float(w_row.get("weight_ext") or w_row["dynamic_weight"]), 3),
                            accuracyPercent=round(float(w_row["accuracy_pct"]), 1),
                            maeCelsius=round(float(w_row["mae_temp_c"]), 2),
                            biasCelsius=round(b0, 2),
                            diurnalBiasNowCelsius=diurnal_now,
                            maeWindKmh=round(float(w_row.get("mae_wind_kmh") or 3.5), 2),
                            brierPrecip=round(float(w_row.get("brier_precip") or 0.10), 3),
                            samplesVerified=int(w_row["samples_verified"]),
                            currentTempCelsius=m_cur_temp,
                            currentPrecipProb=m_cur_pop,
                            currentWindKmh=m_cur_wind,
                        )
                    )

            if not weights_map:
                equal_w = round(1.0 / max(1, len(model_hourly_streams)), 3)
                for m in model_hourly_streams:
                    m_stream0 = model_hourly_streams[m][0] if model_hourly_streams[m] else None
                    weights_map[m] = {
                        "dynamic_weight": equal_w,
                        "weight_short": equal_w,
                        "weight_med": equal_w,
                        "weight_ext": equal_w,
                        "bias_temp_c": 0.0,
                        "bias_sin_c": 0.0,
                        "bias_cos_c": 0.0,
                        "bias_wind_kmh": 0.0,
                        "mae_temp_c": 1.0,
                        "accuracy_pct": 94.0,
                    }
                    model_weight_details.append(
                        ModelWeightDetail(
                            modelName=m,
                            weight=equal_w,
                            weightShort=equal_w,
                            weightMed=equal_w,
                            weightExt=equal_w,
                            accuracyPercent=94.0,
                            maeCelsius=1.0,
                            biasCelsius=0.0,
                            diurnalBiasNowCelsius=0.0,
                            samplesVerified=1,
                            currentTempCelsius=round(float(m_stream0["temp_c"]), 1) if m_stream0 else None,
                            currentPrecipProb=int(m_stream0.get("precip_prob", 0)) if m_stream0 else None,
                            currentWindKmh=round(float(m_stream0.get("wind_kmh", 0.0)), 1) if m_stream0 else None,
                        )
                    )

            for bucket_key, norm_key in (
                ("dynamic_weight", "norm_w"),
                ("weight_short", "norm_short"),
                ("weight_med", "norm_med"),
                ("weight_ext", "norm_ext"),
            ):
                b_sum = sum(float(weights_map[m].get(bucket_key) or weights_map[m]["dynamic_weight"]) for m in weights_map) or 1.0
                for m in weights_map:
                    weights_map[m][norm_key] = float(weights_map[m].get(bucket_key) or weights_map[m]["dynamic_weight"]) / b_sum

            # Cap secondary global models in extended range when primary extended guidance is present
            if len(weights_map) >= 4 and ("ECMWF" in weights_map or "NBM" in weights_map):
                ext_cap = 0.12
                capped_models = [m for m in ("GEM", "ICON") if m in weights_map and weights_map[m]["norm_ext"] > ext_cap]
                if capped_models:
                    excess_ext = sum(weights_map[m]["norm_ext"] - ext_cap for m in capped_models)
                    for m in capped_models:
                        weights_map[m]["norm_ext"] = ext_cap
                    uncapped_models = [m for m in weights_map if m not in ("GEM", "ICON")]
                    uncapped_sum = sum(weights_map[m]["norm_ext"] for m in uncapped_models)
                    if uncapped_sum > 0 and excess_ext > 0:
                        for m in uncapped_models:
                            weights_map[m]["norm_ext"] += excess_ext * (weights_map[m]["norm_ext"] / uncapped_sum)

            # 6. Compute Horizon-Weighted & Diurnal-Bias-Corrected Multi-Model Hourly Blend + Oracle Consensus Score
            hourly_models: List[HourlyForecastModel] = []
            hourly_spreads: List[float] = []
            total_hourly_horizon = max(24, min(240, len(times_list) - start_idx))

            for h_offset in range(total_hourly_horizon):
                blended_t = 0.0
                blended_fl = 0.0
                blended_p = 0.0
                blended_pr_mm = 0.0
                blended_wind = 0.0
                blended_gust = 0.0
                blended_wdir_sin = 0.0
                blended_wdir_cos = 0.0
                wdir_w_total = 0.0
                blended_uv = 0.0
                uv_w_total = 0.0
                blended_dew = 0.0
                blended_rh = 0.0
                rh_w_total = 0.0
                blended_cloud = 0.0
                blended_vis = 0.0
                vis_w_total = 0.0
                blended_press = 0.0
                press_w_total = 0.0
                blended_cape = 0.0
                cape_w_total = 0.0
                total_w = 0.0
                wet_w = 0.0
                hour_temps: List[float] = []
                hour_feels: List[float] = []
                hour_pops: List[int] = []
                hour_pr_mms: List[float] = []
                hour_winds: List[float] = []
                hour_gusts: List[float] = []
                hour_uvs: List[float] = []
                hour_dews: List[float] = []
                hour_clouds: List[int] = []
                hour_vis: List[float] = []
                hour_press: List[float] = []
                hour_model_entries: List[Dict[str, Any]] = []
                top_code = 0
                top_w = -1.0
                top_wet_code = 0
                top_wet_w = -1.0
                dt = now + timedelta(hours=h_offset)
                target_utc_hour = dt.hour
                _, wind_bias_scale = calc_horizon_bias_scales(float(h_offset))

                for m_name, stream in model_hourly_streams.items():
                    if h_offset < len(stream) and m_name in weights_map:
                        pt = stream[h_offset]
                        w_info = weights_map[m_name]
                        if h_offset <= 6:
                            w = w_info["norm_short"]
                        elif h_offset <= 24:
                            alpha_h = (h_offset - 6.0) / 18.0
                            w = (1.0 - alpha_h) * w_info["norm_short"] + alpha_h * w_info["norm_med"]
                        elif h_offset <= 48:
                            alpha_h = (h_offset - 24.0) / 24.0
                            w = (1.0 - alpha_h) * w_info["norm_med"] + alpha_h * w_info["norm_ext"]
                        else:
                            w = w_info["norm_ext"]

                        bias = eval_decayed_diurnal_bias(
                            float(w_info.get("bias_temp_c") or 0.0),
                            float(w_info.get("bias_sin_c") or 0.0),
                            float(w_info.get("bias_cos_c") or 0.0),
                            target_utc_hour,
                            float(h_offset),
                        )
                        wind_bias = float(w_info.get("bias_wind_kmh") or 0.0) * 0.4 * wind_bias_scale
                        bias_corrected_t = pt["temp_c"] - bias
                        bias_corrected_fl = pt.get("feels_c", pt["temp_c"]) - bias
                        bias_corrected_dew = pt.get("dew_c", pt["temp_c"] - 4.0) - bias
                        pop_val = int(pt["precip_prob"])
                        pr_mm_val = float(pt.get("precip_mm", 0.0))
                        w_val = max(0.0, float(pt.get("wind_kmh", 0.0)) - wind_bias)
                        g_val = max(w_val, float(pt.get("gust_kmh", round(w_val * 1.3, 1))) - wind_bias)
                        c_val = int(pt.get("cloud_pct", 0))
                        code = int(pt["wmo_code"])

                        blended_t += bias_corrected_t * w
                        blended_fl += bias_corrected_fl * w
                        blended_p += pop_val * w
                        blended_pr_mm += pr_mm_val * w
                        blended_wind += w_val * w
                        blended_gust += g_val * w
                        blended_dew += bias_corrected_dew * w
                        blended_cloud += c_val * w
                        total_w += w

                        hour_temps.append(bias_corrected_t)
                        hour_feels.append(bias_corrected_fl)
                        hour_pops.append(pop_val)
                        hour_pr_mms.append(pr_mm_val)
                        hour_winds.append(w_val)
                        hour_gusts.append(g_val)
                        hour_dews.append(bias_corrected_dew)
                        hour_clouds.append(c_val)
                        hour_model_entries.append(
                            {
                                "weight": w,
                                "temp_c": bias_corrected_t,
                                "dew_c": bias_corrected_dew,
                                "rh_pct": pt.get("rh_pct"),
                                "precip_mm": pr_mm_val,
                                "precip_prob": pop_val,
                                "wmo_code": code,
                            }
                        )

                        if pt.get("press_hpa") is not None:
                            p_val = float(pt["press_hpa"])
                            blended_press += p_val * w
                            press_w_total += w
                            hour_press.append(p_val)

                        if pt.get("wind_dir") is not None:
                            wd_rad = math.radians(float(pt["wind_dir"]))
                            blended_wdir_sin += math.sin(wd_rad) * w
                            blended_wdir_cos += math.cos(wd_rad) * w
                            wdir_w_total += w

                        if pt.get("rh_pct") is not None:
                            blended_rh += float(pt["rh_pct"]) * w
                            rh_w_total += w

                        if pt.get("cape_jkg") is not None:
                            blended_cape += float(pt["cape_jkg"]) * w
                            cape_w_total += w

                        if pt.get("uv_index") is not None:
                            uv_v = float(pt["uv_index"])
                            blended_uv += uv_v * w
                            uv_w_total += w
                            hour_uvs.append(uv_v)

                        if pt.get("vis_km") is not None:
                            vis_v = float(pt["vis_km"])
                            blended_vis += vis_v * w
                            vis_w_total += w
                            hour_vis.append(vis_v)

                        if w > top_w:
                            top_w = w
                            top_code = code
                        if code >= 51:
                            eff_wet_w = w * (1.35 if m_name == "HRRR" else 1.0)
                            wet_w += eff_wet_w
                            if eff_wet_w > top_wet_w:
                                top_wet_w = eff_wet_w
                                top_wet_code = code

                if total_w > 0:
                    final_h_temp = round(blended_t / total_w, 1)
                    final_h_feels = round(blended_fl / total_w, 1)
                    final_h_precip = int(round(blended_p / total_w))
                    final_h_pr_mm = round(blended_pr_mm / total_w, 2)
                    final_h_wind = round(blended_wind / total_w, 1)
                    final_h_gust = round(blended_gust / total_w, 1)
                    final_h_dew = round(blended_dew / total_w, 1)
                    final_h_cloud = int(round(blended_cloud / total_w))
                    final_h_press = round(blended_press / press_w_total, 1) if press_w_total > 0 else 1013.2
                else:
                    final_h_temp, final_h_feels, final_h_precip, final_h_pr_mm, final_h_wind = 0.0, 0.0, 0, 0.0, 0.0
                    final_h_gust, final_h_dew, final_h_cloud, final_h_press = 0.0, 0.0, 0, 1013.2

                final_h_uv = round(blended_uv / uv_w_total, 2) if uv_w_total > 0 else 0.0
                final_h_vis = round(blended_vis / vis_w_total, 2) if vis_w_total > 0 else 16.0

                raw_idx = start_idx + h_offset
                if wdir_w_total > 0:
                    h_wdir = int(round(math.degrees(math.atan2(blended_wdir_sin, blended_wdir_cos)))) % 360
                else:
                    h_wdir = (
                        int(wdir_hourly_arr[raw_idx])
                        if (raw_idx < len(wdir_hourly_arr) and wdir_hourly_arr[raw_idx] is not None)
                        else 180
                    )

                thermo_rh = calc_relative_humidity(final_h_temp, final_h_dew)
                if rh_w_total > 0:
                    model_rh = blended_rh / rh_w_total
                    h_rh = int(round(0.65 * thermo_rh + 0.35 * model_rh))
                else:
                    h_rh = thermo_rh

                if cape_w_total > 0:
                    h_cape = round(blended_cape / cape_w_total, 1)
                else:
                    h_cape = (
                        round(float(cape_hourly_arr[raw_idx]), 1)
                        if (raw_idx < len(cape_hourly_arr) and cape_hourly_arr[raw_idx] is not None)
                        else 0.0
                    )

                if top_wet_code >= 51 and (
                    (wet_w >= 0.32 and (final_h_precip >= 20 or final_h_pr_mm >= 0.08))
                    or final_h_precip >= 40
                    or final_h_pr_mm >= 0.12
                ):
                    top_code = top_wet_code
                elif top_code < 45:
                    # Reconcile sky condition code with RLS-blended cloud cover
                    if final_h_cloud >= 80:
                        top_code = 3
                    elif final_h_cloud >= 26:
                        top_code = 2
                    else:
                        top_code = 0

                # Assimilate live ground-truth METAR / 1km Doppler radar into hour 0
                live_wet_override = False
                if h_offset == 0:
                    min_doppler_dbz = 24.0 if (st_checked_now and not st_wet_now) else 18.0
                    if station_composite and station_composite.get("wx_wmo"):
                        top_code = int(station_composite["wx_wmo"])
                        final_h_precip = max(final_h_precip, 85)
                        final_h_pr_mm = max(final_h_pr_mm, float(station_composite.get("precip_mmhr") or 0.5))
                        live_wet_override = top_code >= 51
                    elif (
                        doppler_telemetry
                        and doppler_telemetry.currentDbz >= min_doppler_dbz
                        and doppler_telemetry.precipRateMmHr >= 0.15
                    ):
                        top_code = 65 if doppler_telemetry.currentDbz >= 35.0 else (61 if doppler_telemetry.currentDbz >= 20.0 else 51)
                        if final_h_temp <= 0.5:
                            top_code = 71
                        final_h_precip = max(final_h_precip, 75)
                        final_h_pr_mm = max(final_h_pr_mm, doppler_telemetry.precipRateMmHr)
                        live_wet_override = True

                if len(hour_temps) >= 2:
                    hourly_spreads.append(statistics.pstdev(hour_temps))

                h_wb = calc_wet_bulb_c(final_h_temp, h_rh)
                h_pt, h_snow_cm = blend_hourly_precip_and_snow(
                    model_entries=hour_model_entries,
                    total_w=total_w,
                    blended_temp_c=final_h_temp,
                    blended_rh=float(h_rh),
                    blended_pop=final_h_precip,
                    top_code=top_code,
                    lead_hours=float(h_offset),
                )
                if h_offset == 0 and live_wet_override:
                    live_pt = classify_precip_type(
                        temp_c=final_h_temp, rh_pct=float(h_rh), wet_bulb_c=h_wb, wmo_code=top_code
                    )
                    if live_pt in ("snow", "sleet"):
                        h_pt = live_pt
                        h_snow_cm = max(
                            h_snow_cm,
                            round((final_h_pr_mm / 10.0) * calc_snow_liquid_ratio(final_h_temp, live_pt), 2),
                        )

                h_sw = (
                    round(float(sw_hourly_arr[raw_idx]), 1)
                    if (raw_idx < len(sw_hourly_arr) and sw_hourly_arr[raw_idx] is not None)
                    else None
                )
                h_vpd = (
                    round(float(vpd_hourly_arr[raw_idx]), 2)
                    if (raw_idx < len(vpd_hourly_arr) and vpd_hourly_arr[raw_idx] is not None)
                    else None
                )
                h_flh = (
                    round(float(flh_hourly_arr[raw_idx]), 0)
                    if (raw_idx < len(flh_hourly_arr) and flh_hourly_arr[raw_idx] is not None)
                    else None
                )
                h_isday = (
                    bool(int(isday_hourly_arr[raw_idx]))
                    if (raw_idx < len(isday_hourly_arr) and isday_hourly_arr[raw_idx] is not None)
                    else None
                )
                h_cclow = (
                    int(round(float(cclow_hourly_arr[raw_idx])))
                    if (raw_idx < len(cclow_hourly_arr) and cclow_hourly_arr[raw_idx] is not None)
                    else None
                )
                h_ccmid = (
                    int(round(float(ccmid_hourly_arr[raw_idx])))
                    if (raw_idx < len(ccmid_hourly_arr) and ccmid_hourly_arr[raw_idx] is not None)
                    else None
                )
                h_cchigh = (
                    int(round(float(cchigh_hourly_arr[raw_idx])))
                    if (raw_idx < len(cchigh_hourly_arr) and cchigh_hourly_arr[raw_idx] is not None)
                    else None
                )

                hourly_models.append(
                    HourlyForecastModel(
                        time=dt,
                        temperature=final_h_temp,
                        tempMin=round(min(hour_temps), 1) if hour_temps else final_h_temp,
                        tempMax=round(max(hour_temps), 1) if hour_temps else final_h_temp,
                        feelsLike=final_h_feels,
                        feelsMin=round(min(hour_feels), 1) if hour_feels else final_h_feels,
                        feelsMax=round(max(hour_feels), 1) if hour_feels else final_h_feels,
                        condition=WeatherCondition.from_wmo_code(top_code),
                        wmoCode=top_code,
                        precipitationChance=max(0, min(100, final_h_precip)),
                        precipProbMin=max(0, min(hour_pops)) if hour_pops else final_h_precip,
                        precipProbMax=min(100, max(hour_pops)) if hour_pops else final_h_precip,
                        precipMm=final_h_pr_mm,
                        precipMmMin=round(min(hour_pr_mms), 2) if hour_pr_mms else final_h_pr_mm,
                        precipMmMax=round(max(hour_pr_mms), 2) if hour_pr_mms else final_h_pr_mm,
                        windSpeed=final_h_wind,
                        windMin=round(min(hour_winds), 1) if hour_winds else final_h_wind,
                        windMax=round(max(hour_winds), 1) if hour_winds else final_h_wind,
                        windGusts=final_h_gust,
                        gustMin=round(min(hour_gusts), 1) if hour_gusts else final_h_gust,
                        gustMax=round(max(hour_gusts), 1) if hour_gusts else final_h_gust,
                        windDirection=h_wdir,
                        humidity=max(5, min(100, h_rh)),
                        uvIndex=final_h_uv,
                        uvMin=round(min(hour_uvs), 2) if hour_uvs else final_h_uv,
                        uvMax=round(max(hour_uvs), 2) if hour_uvs else final_h_uv,
                        dewPoint=final_h_dew,
                        dewMin=round(min(hour_dews), 1) if hour_dews else final_h_dew,
                        dewMax=round(max(hour_dews), 1) if hour_dews else final_h_dew,
                        cloudCover=final_h_cloud,
                        cloudMin=max(0, min(hour_clouds)) if hour_clouds else final_h_cloud,
                        cloudMax=min(100, max(hour_clouds)) if hour_clouds else final_h_cloud,
                        visibilityKm=final_h_vis,
                        visMin=round(min(hour_vis), 2) if hour_vis else final_h_vis,
                        visMax=round(max(hour_vis), 2) if hour_vis else final_h_vis,
                        pressure=final_h_press,
                        pressMin=round(min(hour_press), 1) if hour_press else final_h_press,
                        pressMax=round(max(hour_press), 1) if hour_press else final_h_press,
                        capeJkg=h_cape,
                        wetBulbTemperature=h_wb,
                        precipType=h_pt,
                        snowAccumulationCm=h_snow_cm,
                        isDay=h_isday,
                        shortwaveRadiation=h_sw,
                        vpdKpa=h_vpd,
                        cloudCoverLow=h_cclow,
                        cloudCoverMid=h_ccmid,
                        cloudCoverHigh=h_cchigh,
                        freezingLevelMeters=h_flh,
                    )
                )

            avg_spread = statistics.mean(hourly_spreads) if hourly_spreads else 0.8
            consensus_score = int(round(max(62.0, min(99.0, 100.0 - (avg_spread * 8.5)))))

            # 7. Assimilate Hyperlocal Station Ground-Truth with Bias-Corrected Model Ensemble for Current Weather & Anchor NOW
            ensemble_current_temp = (
                hourly_models[0].temperature
                if hourly_models
                else float(current_raw.get("temperature_2m") or current_raw.get("temperature_2m_best_match") or 0.0)
            )

            local_station_name: Optional[str] = None
            local_station_dist: Optional[float] = None
            local_station_wx: Optional[str] = None

            if station_composite and station_composite["closest_distance_km"] <= 35.0:
                dist = station_composite["closest_distance_km"]
                station_weight = max(0.35, min(0.75, 1.0 - (dist / 50.0)))
                final_current_temp = round(
                    station_composite["temperature_c"] * station_weight
                    + ensemble_current_temp * (1.0 - station_weight),
                    1,
                )
                local_station_name = station_composite["closest_station_name"]
                local_station_dist = dist
                local_station_wx = station_composite.get("wx_string")
            else:
                final_current_temp = round(ensemble_current_temp, 1)

            temp_delta_now = round(final_current_temp - ensemble_current_temp, 2)

            raw_dewp = (
                station_composite["dewpoint_c"]
                if (station_composite and station_composite.get("dewpoint_c") is not None)
                else (
                    hourly_models[0].dewPoint
                    if (hourly_models and hourly_models[0].dewPoint is not None)
                    else (final_current_temp - 5.0)
                )
            )
            dewpoint_c = round(float(raw_dewp), 1)
            comfort_label = classify_dewpoint_comfort(dewpoint_c)

            min_curr_doppler_dbz = 24.0 if (st_checked_now and not st_wet_now) else 18.0
            if station_composite and station_composite.get("wx_condition") is not None:
                current_condition = station_composite["wx_condition"]
                observed_wmo = int(station_composite.get("wx_wmo") or 61)
            elif (
                doppler_telemetry
                and doppler_telemetry.currentDbz >= min_curr_doppler_dbz
                and doppler_telemetry.precipRateMmHr >= 0.15
            ):
                if final_current_temp <= 0.5:
                    current_condition = WeatherCondition.SNOW
                    observed_wmo = 71
                elif doppler_telemetry.currentDbz >= 35.0:
                    current_condition = WeatherCondition.HEAVY_RAIN
                    observed_wmo = 65
                elif doppler_telemetry.currentDbz >= 20.0:
                    current_condition = WeatherCondition.RAIN
                    observed_wmo = 61
                else:
                    current_condition = WeatherCondition.DRIZZLE
                    observed_wmo = 51
            elif hourly_models:
                current_condition = hourly_models[0].condition
                observed_wmo = hourly_models[0].wmoCode
            else:
                wmo_fallback = int(current_raw.get("weather_code") or 0)
                current_condition = WeatherCondition.from_wmo_code(wmo_fallback)
                observed_wmo = wmo_fallback

            current_uv = float(hourly_models[0].uvIndex) if hourly_models and hourly_models[0].uvIndex is not None else 0.0

            top_models = [m.modelName for m in model_weight_details[:2]]
            dominant_str = " + ".join(top_models) + " Blend" if top_models else "Multi-Model Blend"

            msl_pressure = (
                (station_composite and station_composite.get("pressure_hpa"))
                or (hourly_models[0].pressure if hourly_models and hourly_models[0].pressure is not None else None)
                or current_raw.get("pressure_msl")
                or 1013.25
            )

            wind_speed_kmh = float(
                (station_composite and station_composite.get("wind_kmh"))
                if (station_composite and station_composite.get("wind_kmh") is not None)
                else (
                    hourly_models[0].windSpeed
                    if hourly_models
                    else (current_raw.get("wind_speed_10m") or 0.0)
                )
            )
            wind_gust_kmh = float(
                (station_composite and station_composite.get("gust_kmh"))
                if (station_composite and station_composite.get("gust_kmh") is not None)
                else (
                    hourly_models[0].windGusts
                    if (hourly_models and hourly_models[0].windGusts is not None)
                    else round(wind_speed_kmh * 1.25, 1)
                )
            )
            wind_dir_deg = int(
                (station_composite and station_composite.get("wind_dir"))
                if (station_composite and station_composite.get("wind_dir") is not None)
                else (
                    hourly_models[0].windDirection
                    if (hourly_models and hourly_models[0].windDirection is not None)
                    else 180
                )
            )

            visibility_km = float(
                (station_composite and station_composite.get("visibility_km"))
                if (station_composite and station_composite.get("visibility_km") is not None)
                else (
                    hourly_models[0].visibilityKm
                    if (hourly_models and hourly_models[0].visibilityKm is not None)
                    else 16.0
                )
            )

            cloud_cover_pct = int(
                hourly_models[0].cloudCover
                if (hourly_models and hourly_models[0].cloudCover is not None)
                else (95 if observed_wmo >= 3 else 25)
            )
            if observed_wmo >= 51:
                cloud_cover_pct = max(cloud_cover_pct, 88)
            elif observed_wmo == 3:
                cloud_cover_pct = max(cloud_cover_pct, 85)

            curr_precip_mmhr = float(
                max(
                    nowcast_obj.points[0].precipIntensityMmHr if nowcast_obj.points else 0.0,
                    doppler_telemetry.precipRateMmHr if doppler_telemetry else 0.0,
                    float(station_composite.get("precip_mmhr") or 0.0) if station_composite else 0.0,
                )
            )

            base_feels = (
                hourly_models[0].feelsLike
                if (hourly_models and hourly_models[0].feelsLike is not None)
                else final_current_temp
            )
            final_current_feels = round(float(base_feels) + temp_delta_now, 1)

            curr_cape = float(
                hourly_models[0].capeJkg
                if (hourly_models and hourly_models[0].capeJkg is not None)
                else (current_raw.get("cape") or 0.0)
            )
            curr_li = float(
                li_hourly_arr[start_idx]
                if (start_idx < len(li_hourly_arr) and li_hourly_arr[start_idx] is not None)
                else 2.0
            )

            curr_hum = int(
                (station_composite and station_composite.get("humidity"))
                if (station_composite and station_composite.get("humidity") is not None)
                else (
                    hourly_models[0].humidity
                    if (hourly_models and hourly_models[0].humidity is not None)
                    else calc_relative_humidity(final_current_temp, dewpoint_c)
                )
            )
            curr_wb = calc_wet_bulb_c(final_current_temp, float(curr_hum))
            curr_pt = classify_precip_type(
                temp_c=final_current_temp,
                rh_pct=float(curr_hum),
                wet_bulb_c=curr_wb,
                wmo_code=observed_wmo,
                metar_wx=local_station_wx,
            )

            # Anchor hourly[0] ("NOW") to assimilated current state and smoothly decay station delta into +1h/+2h
            if hourly_models:
                h0 = hourly_models[0]
                h0.temperature = final_current_temp
                h0.tempMin = min(h0.tempMin if h0.tempMin is not None else final_current_temp, final_current_temp)
                h0.tempMax = max(h0.tempMax if h0.tempMax is not None else final_current_temp, final_current_temp)
                h0.feelsLike = final_current_feels
                h0.feelsMin = min(h0.feelsMin if h0.feelsMin is not None else final_current_feels, final_current_feels)
                h0.feelsMax = max(h0.feelsMax if h0.feelsMax is not None else final_current_feels, final_current_feels)
                h0.wmoCode = observed_wmo
                h0.condition = current_condition
                h0.cloudCover = cloud_cover_pct
                h0.humidity = curr_hum
                h0.dewPoint = dewpoint_c
                h0.windSpeed = round(wind_speed_kmh, 1)
                h0.windGusts = round(wind_gust_kmh, 1)
                h0.windDirection = wind_dir_deg
                h0.pressure = round(float(msl_pressure), 1)
                h0.visibilityKm = round(visibility_km, 2)
                h0.wetBulbTemperature = curr_wb
                h0.precipType = curr_pt

                if abs(temp_delta_now) >= 0.1:
                    for decay_idx, decay_factor in ((1, 0.55), (2, 0.25)):
                        if decay_idx < len(hourly_models):
                            hk = hourly_models[decay_idx]
                            adj_delta = round(temp_delta_now * decay_factor, 1)
                            hk.temperature = round(hk.temperature + adj_delta, 1)
                            if hk.feelsLike is not None:
                                hk.feelsLike = round(hk.feelsLike + adj_delta, 1)
                            if hk.tempMin is not None:
                                hk.tempMin = min(hk.tempMin, hk.temperature)
                            if hk.tempMax is not None:
                                hk.tempMax = max(hk.tempMax, hk.temperature)

            # 8. Compute Multi-Model Daily Blend (10 Days) Reconciled with RLS Hourly Slices
            daily_times = daily_raw.get("time", [])
            daily_models: List[DailyForecastModel] = []
            day_limit = min(len(daily_times), 10)
            d_uv_arr = daily_raw.get("uv_index_max") or daily_raw.get("uv_index_max_best_match") or []
            d_sr_arr = daily_raw.get("sunrise") or daily_raw.get("sunrise_best_match") or []
            d_ss_arr = daily_raw.get("sunset") or daily_raw.get("sunset_best_match") or []

            # Convert Open-Meteo UTC sunrise/sunset timestamps into location-local clock strings
            # ('YYYY-MM-DDTHH:MM') indexed by local calendar date so timezones never shift sunrise/sunset.
            local_sr_by_date: Dict[str, str] = {}
            local_ss_by_date: Dict[str, str] = {}
            tz_delta = timedelta(seconds=utc_offset_sec)
            for raw_sr in d_sr_arr:
                if raw_sr:
                    try:
                        dt_utc = datetime.strptime(str(raw_sr)[:16], "%Y-%m-%dT%H:%M")
                        dt_loc = dt_utc + tz_delta
                        local_sr_by_date[dt_loc.strftime("%Y-%m-%d")] = dt_loc.strftime("%Y-%m-%dT%H:%M")
                    except Exception:
                        pass
            for raw_ss in d_ss_arr:
                if raw_ss:
                    try:
                        dt_utc = datetime.strptime(str(raw_ss)[:16], "%Y-%m-%dT%H:%M")
                        dt_loc = dt_utc + tz_delta
                        local_ss_by_date[dt_loc.strftime("%Y-%m-%d")] = dt_loc.strftime("%Y-%m-%dT%H:%M")
                    except Exception:
                        pass

            local_now = now + tz_delta
            utc_offset_hrs = int(round(utc_offset_sec / 3600.0))

            hourly_by_local_date: Dict[str, List[HourlyForecastModel]] = {}
            for h_m in hourly_models:
                h_loc_key = (h_m.time + tz_delta).strftime("%Y-%m-%d")
                hourly_by_local_date.setdefault(h_loc_key, []).append(h_m)

            raw_hours_by_local_date: Dict[str, List[tuple[int, int]]] = {}
            for r_idx, t_str in enumerate(times_list):
                try:
                    dt_u = datetime.strptime(str(t_str)[:16], "%Y-%m-%dT%H:%M")
                    loc_k = (dt_u + tz_delta).strftime("%Y-%m-%d")
                    raw_hours_by_local_date.setdefault(loc_k, []).append((r_idx, dt_u.hour))
                except Exception:
                    pass

            daily_idx_by_date: Dict[str, int] = {
                str(t_d)[:10]: idx_d for idx_d, t_d in enumerate(daily_times)
            }

            for d_idx in range(day_limit):
                local_day_dt = local_now + timedelta(days=d_idx)
                local_date_key = local_day_dt.strftime("%Y-%m-%d")
                day_date_utc = datetime(
                    local_day_dt.year,
                    local_day_dt.month,
                    local_day_dt.day,
                    12,
                    0,
                    0,
                    tzinfo=timezone.utc,
                )
                raw_d_idx = daily_idx_by_date.get(
                    local_date_key,
                    min(d_idx, max(0, len(daily_times) - 1)),
                )
                loc_raw_hours = raw_hours_by_local_date.get(local_date_key, [])
                min_t_blend = 0.0
                max_t_blend = 0.0
                precip_blend = 0.0
                psum_blend = 0.0
                wmax_blend = 0.0
                w_total = 0.0
                wet_day_w = 0.0
                mod_wet_w = 0.0
                day_code = 0
                best_w = -1.0
                day_lead_h = float(d_idx * 24 + 12)

                afternoon_utc_h = (15 - utc_offset_hrs) % 24
                dawn_utc_h = (5 - utc_offset_hrs) % 24

                for m_name, suffix in model_suffixes.items():
                    if m_name not in weights_map:
                        continue
                    w_info = weights_map[m_name]
                    if d_idx == 0:
                        w = 0.6 * w_info["norm_short"] + 0.4 * w_info["norm_med"]
                    elif d_idx == 1:
                        w = 0.5 * w_info["norm_med"] + 0.5 * w_info["norm_ext"]
                    else:
                        w = w_info["norm_ext"]

                    b0 = float(w_info.get("bias_temp_c") or 0.0)
                    bsin = float(w_info.get("bias_sin_c") or 0.0)
                    bcos = float(w_info.get("bias_cos_c") or 0.0)

                    h_t_arr = _raw_model_series(hourly_raw, "temperature_2m", suffix)
                    h_p_arr = _raw_model_series(hourly_raw, "precipitation_probability", suffix)
                    h_pr_arr = _raw_model_series(hourly_raw, "precipitation", suffix)
                    h_w_arr = _raw_model_series(hourly_raw, "wind_speed_10m", suffix)
                    h_c_arr = _raw_model_series(hourly_raw, "weather_code", suffix)

                    loc_adj_temps = [
                        float(h_t_arr[ri]) - eval_decayed_diurnal_bias(b0, bsin, bcos, uh, max(0.0, float(ri - start_idx)))
                        for ri, uh in loc_raw_hours
                        if ri < len(h_t_arr) and h_t_arr[ri] is not None
                    ]
                    if len(loc_adj_temps) >= 12:
                        max_t_blend += max(loc_adj_temps) * w
                        min_t_blend += min(loc_adj_temps) * w
                        pop_vals = [
                            float(h_p_arr[ri])
                            for ri, _ in loc_raw_hours
                            if ri < len(h_p_arr) and h_p_arr[ri] is not None
                        ]
                        pr_vals = [
                            float(h_pr_arr[ri])
                            for ri, _ in loc_raw_hours
                            if ri < len(h_pr_arr) and h_pr_arr[ri] is not None
                        ]
                        w_vals = [
                            float(h_w_arr[ri])
                            for ri, _ in loc_raw_hours
                            if ri < len(h_w_arr) and h_w_arr[ri] is not None
                        ]
                        c_vals = [
                            int(h_c_arr[ri])
                            for ri, _ in loc_raw_hours
                            if ri < len(h_c_arr) and h_c_arr[ri] is not None
                        ]
                        m_psum = sum(pr_vals) if pr_vals else 0.0
                        precip_blend += (max(pop_vals) if pop_vals else 0.0) * w
                        psum_blend += m_psum * w
                        wmax_blend += (max(w_vals) if w_vals else 12.0) * w
                        w_total += w
                        if m_psum >= 1.0:
                            wet_day_w += w
                        if m_psum >= 2.5:
                            mod_wet_w += w
                        if w > best_w and c_vals:
                            best_w = w
                            wet_c = [cv for cv in c_vals if cv >= 51]
                            if wet_c and max(pop_vals or [0.0]) >= 30:
                                day_code = max(wet_c)
                            else:
                                dry_c = [cv for cv in c_vals if cv < 51]
                                day_code = max(dry_c) if dry_c else 2
                        continue

                    bias_max = eval_decayed_diurnal_bias(b0, bsin, bcos, afternoon_utc_h, day_lead_h)
                    bias_min = eval_decayed_diurnal_bias(b0, bsin, bcos, dawn_utc_h, max(0.0, day_lead_h - 6.0))

                    d_max_arr = _raw_model_series(daily_raw, "temperature_2m_max", suffix)
                    d_min_arr = _raw_model_series(daily_raw, "temperature_2m_min", suffix)
                    d_pop_arr = _raw_model_series(daily_raw, "precipitation_probability_max", suffix)
                    d_psum_arr = _raw_model_series(daily_raw, "precipitation_sum", suffix)
                    d_wmax_arr = _raw_model_series(daily_raw, "wind_speed_10m_max", suffix)
                    d_code_arr = _raw_model_series(daily_raw, "weather_code", suffix)

                    if (
                        raw_d_idx < len(d_max_arr)
                        and d_max_arr[raw_d_idx] is not None
                        and raw_d_idx < len(d_min_arr)
                        and d_min_arr[raw_d_idx] is not None
                    ):
                        max_t_blend += (float(d_max_arr[raw_d_idx]) - bias_max) * w
                        min_t_blend += (float(d_min_arr[raw_d_idx]) - bias_min) * w
                        pop_val = float(d_pop_arr[raw_d_idx]) if raw_d_idx < len(d_pop_arr) and d_pop_arr[raw_d_idx] is not None else 0.0
                        psum_val = float(d_psum_arr[raw_d_idx]) if raw_d_idx < len(d_psum_arr) and d_psum_arr[raw_d_idx] is not None else 0.0
                        wmax_val = float(d_wmax_arr[raw_d_idx]) if raw_d_idx < len(d_wmax_arr) and d_wmax_arr[raw_d_idx] is not None else 12.0
                        precip_blend += pop_val * w
                        psum_blend += psum_val * w
                        wmax_blend += wmax_val * w
                        w_total += w
                        if psum_val >= 1.0:
                            wet_day_w += w
                        if psum_val >= 2.5:
                            mod_wet_w += w
                        if w > best_w and raw_d_idx < len(d_code_arr) and d_code_arr[raw_d_idx] is not None:
                            best_w = w
                            raw_dc = int(d_code_arr[raw_d_idx])
                            day_code = 2 if (pop_val < 30 and raw_dc >= 51) else raw_dc

                if w_total > 0:
                    d_min = round(min_t_blend / w_total, 1)
                    d_max = round(max_t_blend / w_total, 1)
                    base_d_pop = precip_blend / w_total
                    agree_frac = wet_day_w / w_total
                    mod_agree_frac = mod_wet_w / w_total
                    if agree_frac >= 0.60 and base_d_pop >= 25.0:
                        d_pop = int(min(90, round(base_d_pop + 22.0 * (agree_frac - 0.45) + 12.0 * mod_agree_frac)))
                    else:
                        d_pop = int(round(base_d_pop))
                    d_psum = round(psum_blend / w_total, 2)
                    d_wmax = round(wmax_blend / w_total, 1)
                else:
                    d_min, d_max, d_pop, d_psum, d_wmax = 0.0, 0.0, 0, 0.0, 10.0

                d_wdir_arr = (
                    daily_raw.get("wind_direction_10m_dominant_best_match")
                    or daily_raw.get("wind_direction_10m_dominant")
                    or []
                )
                d_wdir = int(d_wdir_arr[raw_d_idx]) if raw_d_idx < len(d_wdir_arr) and d_wdir_arr[raw_d_idx] is not None else 180
                d_uv = float(d_uv_arr[raw_d_idx]) if raw_d_idx < len(d_uv_arr) and d_uv_arr[raw_d_idx] is not None else 5.0
                astro_sr, astro_ss = calc_local_sunrise_sunset_iso(
                    latitude, longitude, local_day_dt, utc_offset_sec
                )
                d_sr = local_sr_by_date.get(local_date_key) or astro_sr
                d_ss = local_ss_by_date.get(local_date_key) or astro_ss
                if d_ss <= d_sr:
                    d_sr, d_ss = astro_sr, astro_ss

                day_slice = hourly_by_local_date.get(local_date_key, [])
                if day_slice:
                    slice_temps = [h.temperature for h in day_slice]
                    if d_idx == 0:
                        d_min = round(min(d_min, min(slice_temps), final_current_temp), 1)
                        d_max = round(max(d_max, max(slice_temps), final_current_temp), 1)
                    else:
                        d_min = round(min(d_min, min(slice_temps)), 1)
                        d_max = round(max(d_max, max(slice_temps)), 1)

                    slice_pops = [int(h.precipitationChance) for h in day_slice if h.precipitationChance is not None]
                    if slice_pops:
                        d_pop = max(d_pop, max(slice_pops))

                    slice_uvs = [float(h.uvIndex) for h in day_slice if h.uvIndex is not None]
                    if slice_uvs:
                        d_uv = max(d_uv, max(slice_uvs))

                    sin_sum = sum(math.sin(math.radians(float(h.windDirection))) for h in day_slice if h.windDirection is not None)
                    cos_sum = sum(math.cos(math.radians(float(h.windDirection))) for h in day_slice if h.windDirection is not None)
                    if abs(sin_sum) > 1e-6 or abs(cos_sum) > 1e-6:
                        d_wdir = int(round(math.degrees(math.atan2(sin_sum, cos_sum)))) % 360

                    day_code, d_pt, d_snow_accum, d_slr = reconcile_daily_condition_and_snow(
                        day_slice=day_slice,
                        d_idx=d_idx,
                        d_pop=d_pop,
                        d_psum=d_psum,
                        d_min=d_min,
                        d_max=d_max,
                        initial_day_code=day_code,
                        tz_delta=tz_delta,
                    )
                elif d_idx == 0:
                    d_min = round(min(d_min, final_current_temp), 1)
                    d_max = round(max(d_max, final_current_temp), 1)
                    d_mid = (d_min + d_max) / 2.0
                    d_pt = classify_precip_type(temp_c=d_mid, wmo_code=day_code)
                    d_slr = calc_snow_liquid_ratio(d_mid, d_pt)
                    raw_snow_accum = round((d_psum / 10.0) * d_slr, 2) if (d_pt in ("snow", "sleet") and d_psum > 0) else 0.0
                    d_snow_accum = raw_snow_accum if (d_pop >= 20 or raw_snow_accum >= 0.25) else 0.0
                    if d_pop < 25 and day_code >= 51:
                        day_code = 2
                else:
                    d_mid = (d_min + d_max) / 2.0
                    d_pt = classify_precip_type(temp_c=d_mid, wmo_code=day_code)
                    d_slr = calc_snow_liquid_ratio(d_mid, d_pt)
                    d_snow_accum = round((d_psum / 10.0) * d_slr, 2) if (d_pt in ("snow", "sleet") and d_psum > 0 and d_pop >= 25) else 0.0
                    if d_pop < 30 and day_code >= 51:
                        day_code = 2

                daily_models.append(
                    DailyForecastModel(
                        date=day_date_utc,
                        condition=WeatherCondition.from_wmo_code(day_code),
                        wmoCode=day_code,
                        minTemperature=d_min,
                        maxTemperature=max(d_min, d_max),
                        precipitationChance=max(0, min(100, d_pop)),
                        precipSumMm=d_psum,
                        uvIndexMax=round(d_uv, 1),
                        windSpeedMax=d_wmax,
                        windDirectionDominant=d_wdir,
                        sunrise=d_sr,
                        sunset=d_ss,
                        precipType=d_pt,
                        snowAccumulationCm=d_snow_accum,
                        snowToLiquidRatio=d_slr,
                    )
                )

            temp_max_today = daily_models[0].maxTemperature if daily_models else final_current_temp
            temp_min_today = daily_models[0].minTemperature if daily_models else final_current_temp

            curr_sw = (
                hourly_models[0].shortwaveRadiation
                if (hourly_models and hourly_models[0].shortwaveRadiation is not None)
                else (
                    round(float(current_raw["shortwave_radiation"]), 1)
                    if current_raw.get("shortwave_radiation") is not None
                    else None
                )
            )
            curr_vpd = (
                hourly_models[0].vpdKpa
                if (hourly_models and hourly_models[0].vpdKpa is not None)
                else (
                    round(float(current_raw["vapour_pressure_deficit"]), 2)
                    if current_raw.get("vapour_pressure_deficit") is not None
                    else None
                )
            )
            curr_flh = (
                hourly_models[0].freezingLevelMeters
                if (hourly_models and hourly_models[0].freezingLevelMeters is not None)
                else None
            )
            curr_isday = (
                bool(int(current_raw["is_day"]))
                if current_raw.get("is_day") is not None
                else (
                    hourly_models[0].isDay
                    if (hourly_models and hourly_models[0].isDay is not None)
                    else None
                )
            )
            curr_cclow = (
                hourly_models[0].cloudCoverLow
                if (hourly_models and hourly_models[0].cloudCoverLow is not None)
                else (
                    int(round(float(current_raw["cloud_cover_low"])))
                    if current_raw.get("cloud_cover_low") is not None
                    else None
                )
            )
            curr_ccmid = (
                hourly_models[0].cloudCoverMid
                if (hourly_models and hourly_models[0].cloudCoverMid is not None)
                else (
                    int(round(float(current_raw["cloud_cover_mid"])))
                    if current_raw.get("cloud_cover_mid") is not None
                    else None
                )
            )
            curr_cchigh = (
                hourly_models[0].cloudCoverHigh
                if (hourly_models and hourly_models[0].cloudCoverHigh is not None)
                else (
                    int(round(float(current_raw["cloud_cover_high"])))
                    if current_raw.get("cloud_cover_high") is not None
                    else None
                )
            )

            current_model = WeatherDataModel(
                cityName=city_name,
                temperature=final_current_temp,
                feelsLike=final_current_feels,
                condition=current_condition,
                tempMin=min(temp_min_today, final_current_temp),
                tempMax=max(temp_max_today, final_current_temp),
                humidity=curr_hum,
                windSpeed=wind_speed_kmh,
                windDirection=wind_dir_deg,
                uvIndex=round(current_uv, 1),
                airPressure=round(float(msl_pressure), 1),
                visibility=visibility_km,
                timestamp=now,
                consensusScore=consensus_score,
                dominantModel=dominant_str,
                localStationName=local_station_name,
                localStationDistanceKm=local_station_dist,
                localStationWx=local_station_wx,
                elevationMeters=round(target_elevation, 1),
                usAqi=us_aqi,
                aqiCategory=classify_aqi(us_aqi),
                pm25=pm25_val,
                dewPoint=dewpoint_c,
                comfortLevel=comfort_label,
                nowcastSummary=nowcast_obj.summaryText,
                windGust=wind_gust_kmh,
                cloudCover=cloud_cover_pct,
                precipIntensityMmHr=round(curr_precip_mmhr, 2),
                observedWeatherCode=observed_wmo,
                capeJkg=round(curr_cape, 1),
                liftedIndex=round(curr_li, 2),
                wetBulbTemperature=curr_wb,
                precipType=curr_pt,
                ozone=ozone_val,
                isDay=curr_isday,
                shortwaveRadiation=curr_sw,
                vpdKpa=curr_vpd,
                cloudCoverLow=curr_cclow,
                cloudCoverMid=curr_ccmid,
                cloudCoverHigh=curr_cchigh,
                freezingLevelMeters=curr_flh,
                treePollen=tree_pollen_val,
                grassPollen=grass_pollen_val,
                weedPollen=weed_pollen_val,
            )

            result = WeatherResponse(
                current=current_model,
                hourly=hourly_models,
                daily=daily_models,
                modelWeights=model_weight_details,
                nearbyStations=nearby_stations,
                alerts=active_alerts,
                nowcast=nowcast_obj,
                dopplerTelemetry=doppler_telemetry,
                regionMode=region_mode,
                timezone=tz_name,
                utcOffsetSeconds=utc_offset_sec,
                source=f"ZüsAPI ({dominant_str})",
                cached=False,
                generatedAt=now,
            )

            self._cache[key] = {
                "cached_at": now,
                "data": result,
            }
            while len(self._cache) > self._max_cache_entries:
                self._cache.popitem(last=False)

            return result


weather_service = WeatherProviderService()
