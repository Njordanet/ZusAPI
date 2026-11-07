import asyncio
import math
import statistics
from collections import OrderedDict
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
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
        params: Dict[str, Any] = {
            "latitude": lat,
            "longitude": lon,
            "current": (
                "temperature_2m,relative_humidity_2m,apparent_temperature,dew_point_2m,"
                "weather_code,pressure_msl,surface_pressure,wind_speed_10m,wind_direction_10m,"
                "wind_gusts_10m,cloud_cover,precipitation,cape"
            ),
            "minutely_15": "precipitation,weather_code",
            "hourly": (
                "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation_probability,"
                "precipitation,weather_code,wind_speed_10m,wind_direction_10m,wind_gusts_10m,"
                "uv_index,dew_point_2m,cloud_cover,visibility,pressure_msl,cape,lifted_index"
            ),
            "daily": (
                "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,"
                "precipitation_sum,uv_index_max,wind_speed_10m_max,wind_direction_10m_dominant,"
                "sunrise,sunset"
            ),
            "models": "best_match,ecmwf_ifs025,gfs_seamless,icon_seamless,gem_seamless",
            "timezone": "UTC",
            "forecast_days": 10,
            "forecast_minutely_15": 8,  # Next 2 hours in 15-min intervals
        }
        if include_past_24h:
            params["past_hours"] = 24
        resp = await client.get(settings.open_meteo_forecast_url, params=params)
        resp.raise_for_status()
        return resp.json()

    async def _fetch_air_quality_and_tz(
        self, client: httpx.AsyncClient, lat: float, lon: float
    ) -> Tuple[Optional[int], Optional[float], str, int]:
        default_offset = int(round(lon / 15.0)) * 3600
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "current": "us_aqi,pm2_5",
                "timezone": "auto",
            }
            resp = await client.get(settings.open_meteo_air_quality_url, params=params, timeout=5.0)
            if resp.status_code == 200:
                body = resp.json()
                curr = body.get("current", {})
                aqi = curr.get("us_aqi")
                pm25 = curr.get("pm2_5")
                tz_name = str(body.get("timezone") or "UTC")
                utc_off = int(body.get("utc_offset_seconds") or default_offset)
                return (
                    int(aqi) if aqi is not None else None,
                     round(float(pm25), 2) if pm25 is not None else None,
                    tz_name,
                    utc_off,
                )
        except Exception:
            pass
        return None, None, "UTC", default_offset

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
                # Radar dominates minutes 0..35 (88%), gradually giving NWP more weight by minute 60 (72% radar / 28% NWP)
                w_rad = 0.88 - 0.16 * (t_min / 60.0)
                blended = max(rad_mmhr, rad_mmhr * w_rad + nwp_mmhr * (1.0 - w_rad))
            else:
                blended = max(rad_mmhr, nwp_mmhr)
            fused_61.append(round(blended, 2) if blended >= 0.04 else 0.0)

        if doppler and doppler.inConusGrid:
            doppler.minuteByMinuteRates = fused_61
            doppler.minuteRates60m = [fused_61[t] for t in range(0, 61, 5)]
            if station_mmhr > 0.0 and doppler.currentDbz < 18.0:
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
            region_mode = "US_HYPERLOCAL" if is_us else "GLOBAL_ENSEMBLE"
            cell_id = db_manager.make_cell_id(latitude, longitude)

            # Fast O(1) check if this cell has expired snapshots needing past_hours=24 backfill
            need_past_24h = await db_manager.has_expired_unverified_snapshots(cell_id)

            # 1. Execute ALL upstream queries in parallel (Ensemble + AQI/TZ + NWS Alerts + Nearby METAR/PWS Stations)
            client = http_pool.get_client()
            ensemble_task = self._fetch_open_meteo_ensemble(
                client, latitude, longitude, include_past_24h=need_past_24h
            )
            aqi_task = self._fetch_air_quality_and_tz(client, latitude, longitude)
            alerts_task = self._fetch_nws_polygon_alerts(client, latitude, longitude)
            stations_task = station_service.fetch_nearby_stations(
                lat=latitude, lon=longitude, target_elev_m=0.0
            )
            cell_reg_task = db_manager.register_active_cell(
                lat=latitude, lon=longitude, city_name=city_name, elevation_m=0.0
            )

            (
                raw_data,
                (us_aqi, pm25_val, tz_name, utc_offset_sec),
                active_alerts,
                (nearby_stations, station_composite),
                _,
            ) = await asyncio.gather(
                ensemble_task, aqi_task, alerts_task, stations_task, cell_reg_task
            )

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

            model_hourly_streams: Dict[str, List[Dict[str, Any]]] = {}
            for m_name, suffix in model_suffixes.items():
                t_arr = hourly_raw.get(f"temperature_2m_{suffix}") or hourly_raw.get("temperature_2m") or []
                fl_arr = hourly_raw.get(f"apparent_temperature_{suffix}") or hourly_raw.get("apparent_temperature") or []
                p_arr = hourly_raw.get(f"precipitation_probability_{suffix}") or hourly_raw.get("precipitation_probability") or []
                pr_arr = hourly_raw.get(f"precipitation_{suffix}") or hourly_raw.get("precipitation") or []
                w_arr = hourly_raw.get(f"wind_speed_10m_{suffix}") or hourly_raw.get("wind_speed_10m") or []
                wg_arr = hourly_raw.get(f"wind_gusts_10m_{suffix}") or hourly_raw.get("wind_gusts_10m") or []
                uv_arr = hourly_raw.get(f"uv_index_{suffix}") or hourly_raw.get("uv_index") or []
                dp_arr = hourly_raw.get(f"dew_point_2m_{suffix}") or hourly_raw.get("dew_point_2m") or []
                cc_arr = hourly_raw.get(f"cloud_cover_{suffix}") or hourly_raw.get("cloud_cover") or []
                vis_arr = hourly_raw.get(f"visibility_{suffix}") or hourly_raw.get("visibility") or []
                pr_msl_arr = hourly_raw.get(f"pressure_msl_{suffix}") or hourly_raw.get("pressure_msl") or []
                c_arr = hourly_raw.get(f"weather_code_{suffix}") or hourly_raw.get("weather_code") or []

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
                            "gust_kmh": float(wg_arr[idx]) if idx < len(wg_arr) and wg_arr[idx] is not None else round(w_val * 1.3, 1),
                            "uv_index": float(uv_arr[idx]) if idx < len(uv_arr) and uv_arr[idx] is not None else None,
                            "dew_c": float(dp_arr[idx]) if idx < len(dp_arr) and dp_arr[idx] is not None else float(temp_val) - 4.0,
                            "cloud_pct": int(cc_arr[idx]) if idx < len(cc_arr) and cc_arr[idx] is not None else 0,
                            "vis_km": round(float(vis_arr[idx]) / 1000.0, 2) if idx < len(vis_arr) and vis_arr[idx] is not None else None,
                            "press_hpa": float(pr_msl_arr[idx]) if idx < len(pr_msl_arr) and pr_msl_arr[idx] is not None else 1013.25,
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
            st_wet_now = bool(station_composite and station_composite.get("wx_condition") is not None)
            doppler_telemetry = await doppler_service.get_point_telemetry(
                latitude,
                longitude,
                temp_c=est_temp_now,
                dewpoint_c=est_dewp_now,
                station_wet=st_wet_now,
                radar_zr_factor=float(rls_meta.get("radar_zr_factor") or 1.0),
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
                or (doppler_telemetry and doppler_telemetry.currentDbz >= 15.0)
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
                        )
                    )

            if not weights_map:
                equal_w = round(1.0 / max(1, len(model_hourly_streams)), 3)
                for m in model_hourly_streams:
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
                blended_uv = 0.0
                uv_w_total = 0.0
                blended_dew = 0.0
                blended_cloud = 0.0
                blended_vis = 0.0
                vis_w_total = 0.0
                blended_press = 0.0
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
                top_code = 0
                top_w = -1.0
                top_wet_code = 0
                top_wet_w = -1.0
                dt = now + timedelta(hours=h_offset)
                target_utc_hour = dt.hour

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

                        bias = eval_diurnal_bias(
                            float(w_info.get("bias_temp_c") or 0.0),
                            float(w_info.get("bias_sin_c") or 0.0),
                            float(w_info.get("bias_cos_c") or 0.0),
                            target_utc_hour,
                        )
                        wind_bias = float(w_info.get("bias_wind_kmh") or 0.0) * 0.4
                        bias_corrected_t = pt["temp_c"] - bias
                        bias_corrected_fl = pt.get("feels_c", pt["temp_c"]) - bias
                        bias_corrected_dew = pt.get("dew_c", pt["temp_c"] - 4.0) - bias
                        pop_val = int(pt["precip_prob"])
                        pr_mm_val = float(pt.get("precip_mm", 0.0))
                        w_val = max(0.0, float(pt.get("wind_kmh", 0.0)) - wind_bias)
                        g_val = max(w_val, float(pt.get("gust_kmh", round(w_val * 1.3, 1))) - wind_bias)
                        c_val = int(pt.get("cloud_pct", 0))
                        p_val = float(pt.get("press_hpa", 1013.25))

                        blended_t += bias_corrected_t * w
                        blended_fl += bias_corrected_fl * w
                        blended_p += pop_val * w
                        blended_pr_mm += pr_mm_val * w
                        blended_wind += w_val * w
                        blended_gust += g_val * w
                        blended_dew += bias_corrected_dew * w
                        blended_cloud += c_val * w
                        blended_press += p_val * w
                        total_w += w

                        hour_temps.append(bias_corrected_t)
                        hour_feels.append(bias_corrected_fl)
                        hour_pops.append(pop_val)
                        hour_pr_mms.append(pr_mm_val)
                        hour_winds.append(w_val)
                        hour_gusts.append(g_val)
                        hour_dews.append(bias_corrected_dew)
                        hour_clouds.append(c_val)
                        hour_press.append(p_val)

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

                        code = int(pt["wmo_code"])
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
                    final_h_press = round(blended_press / total_w, 1)
                else:
                    final_h_temp, final_h_feels, final_h_precip, final_h_pr_mm, final_h_wind = 0.0, 0.0, 0, 0.0, 0.0
                    final_h_gust, final_h_dew, final_h_cloud, final_h_press = 0.0, 0.0, 0, 1013.2

                final_h_uv = round(blended_uv / uv_w_total, 2) if uv_w_total > 0 else 0.0
                final_h_vis = round(blended_vis / vis_w_total, 2) if vis_w_total > 0 else 16.0

                raw_idx = start_idx + h_offset
                h_wdir = (
                    int(wdir_hourly_arr[raw_idx])
                    if (raw_idx < len(wdir_hourly_arr) and wdir_hourly_arr[raw_idx] is not None)
                    else 180
                )
                h_rh = (
                    int(rh_hourly_arr[raw_idx])
                    if (raw_idx < len(rh_hourly_arr) and rh_hourly_arr[raw_idx] is not None)
                    else calc_relative_humidity(final_h_temp, final_h_dew)
                )
                h_cape = (
                    round(float(cape_hourly_arr[raw_idx]), 1)
                    if (raw_idx < len(cape_hourly_arr) and cape_hourly_arr[raw_idx] is not None)
                    else 0.0
                )

                if top_wet_code >= 51 and (wet_w >= 0.32 or final_h_precip >= 40 or final_h_pr_mm >= 0.1):
                    top_code = top_wet_code

                # Assimilate live ground-truth METAR / 1km Doppler radar into hour 0
                if h_offset == 0:
                    if station_composite and station_composite.get("wx_wmo"):
                        top_code = int(station_composite["wx_wmo"])
                        final_h_precip = max(final_h_precip, 85)
                        final_h_pr_mm = max(final_h_pr_mm, float(station_composite.get("precip_mmhr") or 0.5))
                    elif doppler_telemetry and doppler_telemetry.currentDbz >= 15.0:
                        top_code = 65 if doppler_telemetry.currentDbz >= 35.0 else (61 if doppler_telemetry.currentDbz >= 20.0 else 51)
                        if final_h_temp <= 0.5:
                            top_code = 71
                        final_h_precip = max(final_h_precip, 75)
                        final_h_pr_mm = max(final_h_pr_mm, doppler_telemetry.precipRateMmHr)

                if len(hour_temps) >= 2:
                    hourly_spreads.append(statistics.pstdev(hour_temps))

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
                    )
                )

            avg_spread = statistics.mean(hourly_spreads) if hourly_spreads else 0.8
            consensus_score = int(round(max(62.0, min(99.0, 100.0 - (avg_spread * 8.5)))))

            # 7. Compute Multi-Model Daily Blend (10 Days)
            daily_times = daily_raw.get("time", [])
            daily_models: List[DailyForecastModel] = []
            day_limit = min(len(daily_times), 10)
            d_uv_arr = daily_raw.get("uv_index_max") or daily_raw.get("uv_index_max_best_match") or []
            d_sr_arr = daily_raw.get("sunrise") or daily_raw.get("sunrise_best_match") or []
            d_ss_arr = daily_raw.get("sunset") or daily_raw.get("sunset_best_match") or []

            for d_idx in range(day_limit):
                dt = now + timedelta(days=d_idx)
                min_t_blend = 0.0
                max_t_blend = 0.0
                precip_blend = 0.0
                psum_blend = 0.0
                wmax_blend = 0.0
                w_total = 0.0
                day_code = 0
                best_w = -1.0

                utc_offset_hrs = int(round(longitude / 15.0))
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
                    bias_max = eval_diurnal_bias(b0, bsin, bcos, afternoon_utc_h)
                    bias_min = eval_diurnal_bias(b0, bsin, bcos, dawn_utc_h)

                    d_max_arr = daily_raw.get(f"temperature_2m_max_{suffix}") or daily_raw.get("temperature_2m_max") or []
                    d_min_arr = daily_raw.get(f"temperature_2m_min_{suffix}") or daily_raw.get("temperature_2m_min") or []
                    d_pop_arr = (
                        daily_raw.get(f"precipitation_probability_max_{suffix}")
                        or daily_raw.get("precipitation_probability_max")
                        or []
                    )
                    d_psum_arr = (
                        daily_raw.get(f"precipitation_sum_{suffix}")
                        or daily_raw.get("precipitation_sum")
                        or []
                    )
                    d_wmax_arr = (
                        daily_raw.get(f"wind_speed_10m_max_{suffix}")
                        or daily_raw.get("wind_speed_10m_max")
                        or []
                    )
                    d_code_arr = daily_raw.get(f"weather_code_{suffix}") or daily_raw.get("weather_code") or []

                    if d_idx < len(d_max_arr) and d_max_arr[d_idx] is not None and d_idx < len(d_min_arr) and d_min_arr[d_idx] is not None:
                        max_t_blend += (float(d_max_arr[d_idx]) - bias_max) * w
                        min_t_blend += (float(d_min_arr[d_idx]) - bias_min) * w
                        pop_val = float(d_pop_arr[d_idx]) if d_idx < len(d_pop_arr) and d_pop_arr[d_idx] is not None else 0.0
                        psum_val = float(d_psum_arr[d_idx]) if d_idx < len(d_psum_arr) and d_psum_arr[d_idx] is not None else 0.0
                        wmax_val = float(d_wmax_arr[d_idx]) if d_idx < len(d_wmax_arr) and d_wmax_arr[d_idx] is not None else 12.0
                        precip_blend += pop_val * w
                        psum_blend += psum_val * w
                        wmax_blend += wmax_val * w
                        w_total += w
                        if w > best_w and d_idx < len(d_code_arr) and d_code_arr[d_idx] is not None:
                            best_w = w
                            day_code = int(d_code_arr[d_idx])

                if w_total > 0:
                    d_min = round(min_t_blend / w_total, 1)
                    d_max = round(max_t_blend / w_total, 1)
                    d_pop = int(round(precip_blend / w_total))
                    d_psum = round(psum_blend / w_total, 2)
                    d_wmax = round(wmax_blend / w_total, 1)
                else:
                    d_min, d_max, d_pop, d_psum, d_wmax = 0.0, 0.0, 0, 0.0, 10.0

                d_wdir_arr = (
                    daily_raw.get("wind_direction_10m_dominant_best_match")
                    or daily_raw.get("wind_direction_10m_dominant")
                    or []
                )
                d_wdir = int(d_wdir_arr[d_idx]) if d_idx < len(d_wdir_arr) and d_wdir_arr[d_idx] is not None else 180
                d_uv = float(d_uv_arr[d_idx]) if d_idx < len(d_uv_arr) and d_uv_arr[d_idx] is not None else 5.0
                d_sr = str(d_sr_arr[d_idx]) if d_idx < len(d_sr_arr) and d_sr_arr[d_idx] is not None else None
                d_ss = str(d_ss_arr[d_idx]) if d_idx < len(d_ss_arr) and d_ss_arr[d_idx] is not None else None

                daily_models.append(
                    DailyForecastModel(
                        date=dt,
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
                    )
                )

            # 8. Assimilate Hyperlocal Station Ground-Truth with Bias-Corrected Model Ensemble for Current Weather
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

            raw_dewp = (
                station_composite["dewpoint_c"]
                if (station_composite and station_composite.get("dewpoint_c") is not None)
                else (
                    current_raw["dew_point_2m"]
                    if current_raw.get("dew_point_2m") is not None
                    else (
                        current_raw["dew_point_2m_best_match"]
                        if current_raw.get("dew_point_2m_best_match") is not None
                        else (hourly_models[0].dewPoint if hourly_models else (final_current_temp - 5.0))
                    )
                )
            )
            dewpoint_c = round(float(raw_dewp), 1)
            comfort_label = classify_dewpoint_comfort(dewpoint_c)

            wmo_now = int(
                current_raw.get("weather_code")
                if current_raw.get("weather_code") is not None
                else (
                    current_raw.get("weather_code_best_match")
                    if current_raw.get("weather_code_best_match") is not None
                    else (hourly_models[0].wmoCode if hourly_models else 0)
                )
            )

            if station_composite and station_composite.get("wx_condition") is not None:
                current_condition = station_composite["wx_condition"]
                observed_wmo = int(station_composite.get("wx_wmo") or 61)
            elif doppler_telemetry and doppler_telemetry.currentDbz >= 15.0:
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
            elif hourly_models and hourly_models[0].wmoCode >= 51:
                current_condition = hourly_models[0].condition
                observed_wmo = hourly_models[0].wmoCode
            else:
                current_condition = WeatherCondition.from_wmo_code(wmo_now)
                observed_wmo = wmo_now

            current_uv = float(hourly_models[0].uvIndex) if hourly_models else 0.0
            temp_max_today = daily_models[0].maxTemperature if daily_models else final_current_temp
            temp_min_today = daily_models[0].minTemperature if daily_models else final_current_temp

            top_models = [m.modelName for m in model_weight_details[:2]]
            dominant_str = " + ".join(top_models) + " Blend" if top_models else "Multi-Model Blend"

            msl_pressure = (
                (station_composite and station_composite.get("pressure_hpa"))
                or current_raw.get("pressure_msl")
                or current_raw.get("pressure_msl_best_match")
                or current_raw.get("surface_pressure")
                or current_raw.get("surface_pressure_best_match")
                or 1013.25
            )

            wind_speed_kmh = float(
                (station_composite and station_composite.get("wind_kmh"))
                or current_raw.get("wind_speed_10m")
                or current_raw.get("wind_speed_10m_best_match")
                or 0.0
            )
            wind_gust_kmh = float(
                (station_composite and station_composite.get("gust_kmh"))
                or current_raw.get("wind_gusts_10m")
                or current_raw.get("wind_gusts_10m_best_match")
                or round(wind_speed_kmh * 1.25, 1)
            )
            wind_dir_deg = int(
                (station_composite and station_composite.get("wind_dir"))
                or current_raw.get("wind_direction_10m")
                or current_raw.get("wind_direction_10m_best_match")
                or 0
            )

            vis_arr = hourly_raw.get("visibility") or hourly_raw.get("visibility_best_match") or []
            om_vis_km = round(float(vis_arr[start_idx]) / 1000.0, 1) if (start_idx < len(vis_arr) and vis_arr[start_idx] is not None) else 16.0
            visibility_km = float(
                (station_composite and station_composite.get("visibility_km"))
                or om_vis_km
            )

            cloud_cover_pct = int(
                current_raw["cloud_cover"]
                if current_raw.get("cloud_cover") is not None
                else (
                    current_raw["cloud_cover_best_match"]
                    if current_raw.get("cloud_cover_best_match") is not None
                    else (hourly_models[0].cloudCover if hourly_models else (95 if observed_wmo >= 3 else 25))
                )
            )
            curr_precip_mmhr = float(
                max(
                    nowcast_obj.points[0].precipIntensityMmHr if nowcast_obj.points else 0.0,
                    doppler_telemetry.precipRateMmHr if doppler_telemetry else 0.0,
                    float(station_composite.get("precip_mmhr") or 0.0) if station_composite else 0.0,
                )
            )

            raw_feels = (
                current_raw["apparent_temperature"]
                if current_raw.get("apparent_temperature") is not None
                else (
                    current_raw["apparent_temperature_best_match"]
                    if current_raw.get("apparent_temperature_best_match") is not None
                    else (hourly_models[0].feelsLike if hourly_models and hourly_models[0].feelsLike is not None else final_current_temp)
                )
            )
            curr_cape = float(
                current_raw.get("cape")
                or current_raw.get("cape_best_match")
                or (hourly_models[0].capeJkg if hourly_models and hourly_models[0].capeJkg is not None else 0.0)
            )
            curr_li = float(
                li_hourly_arr[start_idx]
                if (start_idx < len(li_hourly_arr) and li_hourly_arr[start_idx] is not None)
                else 2.0
            )

            current_model = WeatherDataModel(
                cityName=city_name,
                temperature=final_current_temp,
                feelsLike=round(float(raw_feels), 1),
                condition=current_condition,
                tempMin=min(temp_min_today, final_current_temp),
                tempMax=max(temp_max_today, final_current_temp),
                humidity=int(
                    (station_composite and station_composite.get("humidity"))
                    or current_raw.get("relative_humidity_2m")
                    or current_raw.get("relative_humidity_2m_best_match")
                    or 50
                ),
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
