import asyncio
import httpx
import statistics
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
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
from app.services.db import db_manager
from app.services.station_service import station_service
from app.services.doppler_service import doppler_service


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
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _cache_key(self, lat: float, lon: float) -> str:
        return f"{lat:.3f}:{lon:.3f}"

    async def _fetch_open_meteo_ensemble(
        self, client: httpx.AsyncClient, lat: float, lon: float
    ) -> Dict[str, Any]:
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": (
                "temperature_2m,relative_humidity_2m,apparent_temperature,dew_point_2m,"
                "weather_code,pressure_msl,surface_pressure,wind_speed_10m,wind_direction_10m,"
                "wind_gusts_10m,cloud_cover,precipitation"
            ),
            "minutely_15": "precipitation,weather_code",
            "hourly": (
                "temperature_2m,apparent_temperature,precipitation_probability,precipitation,"
                "weather_code,wind_speed_10m,wind_gusts_10m,uv_index,dew_point_2m,"
                "cloud_cover,visibility,pressure_msl"
            ),
            "daily": (
                "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,"
                "precipitation_sum,uv_index_max,wind_speed_10m_max,wind_direction_10m_dominant,"
                "sunrise,sunset"
            ),
            "models": "best_match,ecmwf_ifs025,gfs_seamless,icon_seamless,gem_seamless",
            "timezone": "UTC",
            "past_hours": 24,
            "forecast_days": 10,
            "forecast_minutely_15": 8,  # Next 2 hours in 15-min intervals
        }
        resp = await client.get(settings.open_meteo_forecast_url, params=params)
        resp.raise_for_status()
        return resp.json()

    async def _fetch_air_quality(
        self, client: httpx.AsyncClient, lat: float, lon: float
    ) -> Tuple[Optional[int], Optional[float]]:
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "current": "us_aqi,pm2_5",
            }
            resp = await client.get(settings.open_meteo_air_quality_url, params=params)
            if resp.status_code == 200:
                curr = resp.json().get("current", {})
                aqi = curr.get("us_aqi")
                pm25 = curr.get("pm2_5")
                return (int(aqi) if aqi is not None else None, float(pm25) if pm25 is not None else None)
        except Exception:
            pass
        return None, None

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
                headers={"User-Agent": "ZusWeatherEngine/2.0 (panda-telemetry)"},
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

    def _build_nowcast(
        self,
        raw_data: Dict[str, Any],
        doppler: Optional[RadarLocalTelemetry] = None,
        station_composite: Optional[Dict[str, Any]] = None,
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

        points: List[NowcastPoint] = []
        max_precip = 0.0

        # Ground-truth METAR precipitation rate (if a local station within 25km reports active rain/drizzle/snow)
        station_mmhr = (
            float(station_composite.get("precip_mmhr") or 0.0)
            if (station_composite and (station_composite.get("wx_station_dist_km") or 99.0) <= 25.0)
            else 0.0
        )

        # Blend Open-Meteo 15-min model precipitation with Züs 1km CONUS Doppler radar trajectory
        doppler_rates = list(doppler.minuteRates60m) if (doppler and doppler.inConusGrid) else []
        if station_mmhr > 0.0 and doppler_rates:
            # Ensure sub-beam drizzle/light rain observed by surface ASOS/METAR is reflected in initial minutes
            for s_idx in range(min(4, len(doppler_rates))):
                decay = max(0.35, 1.0 - s_idx * 0.2)
                doppler_rates[s_idx] = round(max(doppler_rates[s_idx], station_mmhr * decay), 2)
            doppler.minuteRates60m = doppler_rates
            if doppler.currentDbz < 18.0:
                doppler.currentDbz = 19.5 if station_mmhr >= 1.0 else 15.0
                doppler.precipRateMmHr = max(doppler.precipRateMmHr, station_mmhr)
                doppler.intensityCategory = "Light Rain" if station_mmhr >= 0.8 else "Light Drizzle"

        for idx in range(5):
            p_idx = m15_start_idx + idx
            model_val = float(precips[p_idx]) * 4.0 if (p_idx < len(precips) and precips[p_idx] is not None) else 0.0
            # Map idx (0, 15, 30, 45, 60m) to doppler_rates index (0, 3, 6, 9, 12)
            dop_idx = idx * 3
            radar_val = float(doppler_rates[dop_idx]) if dop_idx < len(doppler_rates) else 0.0
            if idx == 0 and station_mmhr > 0.0:
                radar_val = max(radar_val, station_mmhr)
            elif idx == 1 and station_mmhr > 0.0:
                radar_val = max(radar_val, round(station_mmhr * 0.55, 2))
            blended_mmhr = round(max(model_val, radar_val * 0.85 + model_val * 0.15), 2)
            max_precip = max(max_precip, blended_mmhr)
            prob = min(100, int(round(blended_mmhr * 35))) if blended_mmhr > 0.08 else 0
            points.append(
                NowcastPoint(
                    timeOffsetMinutes=idx * 15,
                    precipProbability=prob,
                    precipIntensityMmHr=blended_mmhr,
                )
            )

        if max_precip < 0.08:
            if doppler and doppler.nearestEchoKm is not None and doppler.nearestEchoKm <= 35.0:
                summary_text = (
                    f"Dry overhead — nearest radar echo ({doppler.maxDbzNearby:.0f} dBZ) "
                    f"is {doppler.nearestEchoKm:.0f} km {doppler.nearestEchoCompass}, tracking {doppler.stormMotionCompass}."
                )
            else:
                summary_text = "Clear skies for the next 60 minutes — no precipitation echoes on Doppler radar."
        elif points and points[0].precipIntensityMmHr >= 0.15:
            dbz_tag = f" ({doppler.currentDbz:.0f} dBZ)" if (doppler and doppler.currentDbz >= 12) else ""
            # Check if rain stops within the hour in doppler_rates
            stop_min = None
            if doppler_rates:
                for step_i in range(1, len(doppler_rates)):
                    if doppler_rates[step_i] < 0.08:
                        stop_min = step_i * 5
                        break
            if stop_min is not None:
                summary_text = (
                    f"Active precipitation{dbz_tag} ({points[0].precipIntensityMmHr:.1f} mm/hr) "
                    f"stopping in ~{stop_min} min."
                )
            else:
                summary_text = (
                    f"Active precipitation{dbz_tag} ({points[0].precipIntensityMmHr:.1f} mm/hr) "
                    f"continuing over the next hour."
                )
        else:
            first_rain = next((pt for pt in points if pt.precipIntensityMmHr >= 0.15), None)
            if first_rain:
                summary_text = f"Precipitation echoes approaching — starting in ~{first_rain.timeOffsetMinutes} minutes."
            else:
                summary_text = "Light mist or drizzle in the vicinity over the next hour."

        source_label = (
            "Züs 1km CONUS Doppler Master Grid + ASOS/HRRR Nowcast"
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
    ) -> WeatherResponse:
        key = self._cache_key(latitude, longitude)
        now = datetime.now(timezone.utc)

        if key in self._cache:
            entry = self._cache[key]
            if (now - entry["cached_at"]).total_seconds() < settings.cache_ttl_seconds:
                response: WeatherResponse = entry["data"]
                return response.model_copy(update={"cached": True})

        is_us = db_manager.is_us_region(latitude, longitude)
        region_mode = "US_HYPERLOCAL" if is_us else "GLOBAL_ENSEMBLE"

        # 1. Execute parallel upstream queries (Multi-Model Ensemble + AQI + NWS Polygon Alerts)
        async with httpx.AsyncClient(timeout=10.0) as client:
            ensemble_task = self._fetch_open_meteo_ensemble(client, latitude, longitude)
            aqi_task = self._fetch_air_quality(client, latitude, longitude)
            alerts_task = self._fetch_nws_polygon_alerts(client, latitude, longitude)
            raw_data, (us_aqi, _), active_alerts = await asyncio.gather(
                ensemble_task, aqi_task, alerts_task
            )

        target_elevation = float(raw_data.get("elevation") or 0.0)

        # 2. Register active cell in PostGIS and fetch nearby PWS/METAR stations with elevation normalization & QC
        cell_id = await db_manager.register_active_cell(
            lat=latitude,
            lon=longitude,
            city_name=city_name,
            elevation_m=target_elevation,
        )
        nearby_stations, station_composite = await station_service.fetch_nearby_stations(
            lat=latitude,
            lon=longitude,
            target_elev_m=target_elevation,
        )

        # 3. Extract individual model hourly streams from Open-Meteo Multi-Model payload
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
            # Outside North America, best_match blends regional high-res (DWD/UKMO/MeteoFrance/JMA)
            model_suffixes.pop("HRRR", None)

        times_list = hourly_raw.get("time", [])
        current_hour_str = now.strftime("%Y-%m-%dT%H:00")
        start_idx = 0
        for idx, t_str in enumerate(times_list):
            if t_str >= current_hour_str:
                start_idx = idx
                break

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

        # Query 1km Doppler telemetry early so radar + station ground truth can grade models & condition codes
        doppler_telemetry = await doppler_service.get_point_telemetry(latitude, longitude)
        nowcast_obj = self._build_nowcast(raw_data, doppler_telemetry, station_composite)

        # 4. Run Accuracy Verification, Offline Snapshot Backfill & MOS Bias Update
        past_hourly_temps: Dict[datetime, float] = {}
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
        if past_hourly_temps:
            await db_manager.backfill_expired_snapshots(cell_id=cell_id, past_hourly_temps=past_hourly_temps)

        ground_truth_temp = (
            station_composite["temperature_c"]
            if station_composite and station_composite["closest_distance_km"] <= 45.0
            else None
        )
        ground_truth_precip_active = bool(
            (station_composite and station_composite.get("wx_condition") is not None)
            or (doppler_telemetry and doppler_telemetry.currentDbz >= 15.0)
        )
        await db_manager.save_forecast_snapshots_and_verify(
            cell_id=cell_id,
            model_forecasts=model_hourly_streams,
            ground_truth_temp_c=ground_truth_temp,
            ground_truth_precip_active=ground_truth_precip_active,
        )

        # 5. Fetch updated dynamic weights & learned MOS bias offsets from PostGIS
        db_weights = await db_manager.get_model_weights(cell_id)
        weights_map: Dict[str, Dict[str, Any]] = {}
        model_weight_details: List[ModelWeightDetail] = []

        for w_row in db_weights:
            m = w_row["model_name"]
            if m in model_hourly_streams:
                weights_map[m] = w_row
                model_weight_details.append(
                    ModelWeightDetail(
                        modelName=m,
                        weight=round(float(w_row["dynamic_weight"]), 3),
                        accuracyPercent=round(float(w_row["accuracy_pct"]), 1),
                        maeCelsius=round(float(w_row["mae_temp_c"]), 2),
                        biasCelsius=round(float(w_row["bias_temp_c"]), 2),
                        samplesVerified=int(w_row["samples_verified"]),
                    )
                )

        # Fallback if DB isn't reachable
        if not weights_map:
            equal_w = round(1.0 / max(1, len(model_hourly_streams)), 3)
            for m in model_hourly_streams:
                weights_map[m] = {"dynamic_weight": equal_w, "bias_temp_c": 0.0, "mae_temp_c": 1.0, "accuracy_pct": 94.0}
                model_weight_details.append(
                    ModelWeightDetail(
                        modelName=m,
                        weight=equal_w,
                        accuracyPercent=94.0,
                        maeCelsius=1.0,
                        biasCelsius=0.0,
                        samplesVerified=1,
                    )
                )

        # Normalize active weights to sum to 1.0
        w_sum = sum(weights_map[m]["dynamic_weight"] for m in weights_map) or 1.0
        for m in weights_map:
            weights_map[m]["norm_w"] = weights_map[m]["dynamic_weight"] / w_sum

        # 6. Compute Bias-Corrected Multi-Model Hourly Blend + Oracle Consensus Score
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

            for m_name, stream in model_hourly_streams.items():
                if h_offset < len(stream) and m_name in weights_map:
                    pt = stream[h_offset]
                    w_info = weights_map[m_name]
                    w = w_info["norm_w"]
                    # Subtract learned MOS bias (if model runs +0.5°C hot, subtract 0.5°C)
                    bias = float(w_info["bias_temp_c"])
                    bias_corrected_t = pt["temp_c"] - bias
                    bias_corrected_fl = pt.get("feels_c", pt["temp_c"]) - bias
                    bias_corrected_dew = pt.get("dew_c", pt["temp_c"] - 4.0) - bias
                    pop_val = int(pt["precip_prob"])
                    pr_mm_val = float(pt.get("precip_mm", 0.0))
                    w_val = float(pt.get("wind_kmh", 0.0))
                    g_val = float(pt.get("gust_kmh", round(w_val * 1.3, 1)))
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
                        # Give high-res HRRR extra weight for mesoscale precipitation detection
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

            # If high-res models indicate precipitation (wet_w >= 0.32 or precip prob >= 40%), use the wet WMO code
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
                )
            )

        # Calculate Oracle Consensus Score (100% = zero spread between models; lower when models diverge)
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

            for m_name, suffix in model_suffixes.items():
                if m_name not in weights_map:
                    continue
                w = weights_map[m_name]["norm_w"]
                bias = float(weights_map[m_name]["bias_temp_c"])

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
                    max_t_blend += (float(d_max_arr[d_idx]) - bias) * w
                    min_t_blend += (float(d_min_arr[d_idx]) - bias) * w
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
            # Blend hyperlocal station reading (65% weight if very close) with model ensemble (35%)
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
            else float(current_raw.get("dew_point_2m") or current_raw.get("dew_point_2m_best_match") or (final_current_temp - 5.0))
        )
        dewpoint_c = round(float(raw_dewp), 1)
        comfort_label = classify_dewpoint_comfort(dewpoint_c)

        wmo_now = int(
            current_raw.get("weather_code")
            or current_raw.get("weather_code_best_match")
            or (hourly_models[0].wmoCode if hourly_models else 0)
        )

        # Ground-truth hierarchy for current sky condition:
        # 1. Local METAR/ASOS present weather within 25km (e.g. SPECI KARB -RA)
        # 2. 1km CONUS Doppler overhead reflectivity (>= 15 dBZ)
        # 3. High-res hourly/current ensemble consensus
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

        uv_arr = daily_raw.get("uv_index_max") or daily_raw.get("uv_index_max_best_match") or [5.0]
        uv_max = float(uv_arr[0]) if uv_arr and uv_arr[0] is not None else 5.0

        temp_max_today = daily_models[0].maxTemperature if daily_models else final_current_temp
        temp_min_today = daily_models[0].minTemperature if daily_models else final_current_temp

        top_models = [m.modelName for m in model_weight_details[:2]]
        dominant_str = " + ".join(top_models) + " Blend" if top_models else "Multi-Model Blend"

        # Sea-level barometric pressure (MSL hPa): prioritize METAR altimeter/SLP, then Open-Meteo pressure_msl
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
            current_raw.get("cloud_cover")
            or current_raw.get("cloud_cover_best_match")
            or (95 if observed_wmo >= 3 else 25)
        )
        curr_precip_mmhr = float(
            max(
                nowcast_obj.points[0].precipIntensityMmHr if nowcast_obj.points else 0.0,
                doppler_telemetry.precipRateMmHr if doppler_telemetry else 0.0,
                float(station_composite.get("precip_mmhr") or 0.0) if station_composite else 0.0,
            )
        )

        current_model = WeatherDataModel(
            cityName=city_name,
            temperature=final_current_temp,
            feelsLike=float(
                current_raw.get("apparent_temperature")
                or current_raw.get("apparent_temperature_best_match")
                or (hourly_models[0].feelsLike if hourly_models and hourly_models[0].feelsLike is not None else final_current_temp)
            ),
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
            uvIndex=round(uv_max, 1),
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
            dewPoint=dewpoint_c,
            comfortLevel=comfort_label,
            nowcastSummary=nowcast_obj.summaryText,
            windGust=wind_gust_kmh,
            cloudCover=cloud_cover_pct,
            precipIntensityMmHr=round(curr_precip_mmhr, 2),
            observedWeatherCode=observed_wmo,
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
            source=f"ZüsAPI ({dominant_str})",
            cached=False,
            generatedAt=now,
        )

        self._cache[key] = {
            "cached_at": now,
            "data": result,
        }

        return result


weather_service = WeatherProviderService()
