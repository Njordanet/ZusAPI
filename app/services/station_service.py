import asyncio
import math
import statistics
from collections import OrderedDict
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Dict, Any
from app.config import settings
from app.models.weather import HyperlocalStationReading, WeatherCondition
from app.services.db import db_manager
from app.services.http_client import http_pool


def _parse_obs_timestamp(val: Any) -> Optional[datetime]:
    """Parses epoch timestamps (s or ms) or ISO-8601 strings into UTC datetimes."""
    if val is None:
        return None
    try:
        if isinstance(val, (int, float)):
            ts = float(val)
            if ts > 1e12:
                ts /= 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        s = str(val).strip()
        if not s:
            return None
        if s.isdigit():
            ts = float(s)
            if ts > 1e12:
                ts /= 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _parse_wind_dir(val: Any) -> Optional[int]:
    """Parses wind bearing degrees, returning None for missing or variable ('VRB') winds."""
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip().upper()
        if not s or s in ("VRB", "M", "CALM"):
            return None
        val = s
    try:
        deg = int(round(float(val))) % 360
        return deg
    except Exception:
        return None


def parse_metar_visib_km(visib_val: Any) -> Optional[float]:
    if visib_val is None:
        return None
    try:
        s = str(visib_val).strip().replace("+", "")
        if "/" in s:
            parts = s.split("/")
            miles = float(parts[0]) / float(parts[1])
        else:
            miles = float(s)
        return round(min(16.0, max(0.1, miles * 1.60934)), 1)
    except Exception:
        return None


def parse_metar_wx(
    wx_str: Optional[str], precip_in: Optional[float] = None
) -> Tuple[Optional[WeatherCondition], Optional[int], float]:
    """
    Translates ICAO METAR present weather string (e.g. '-RA', 'RA', '+TSRA', '-DZ', '-SN', 'FG')
    into (WeatherCondition, wmo_code, estimated_mm_hr).
    """
    base_mmhr = round(float(precip_in) * 25.4, 2) if (precip_in and precip_in > 0) else 0.0
    if not wx_str:
        return (None, None, base_mmhr)

    wx = wx_str.upper().strip()
    # Ignore vicinity-only ('VC') codes unless accompanied by overhead precip
    tokens = [t for t in wx.split() if not t.startswith("VC")]
    if not tokens:
        return (None, None, base_mmhr)

    joined = " ".join(tokens)
    if "TS" in joined:
        if "+" in joined or "GR" in joined or "SQ" in joined:
            return (WeatherCondition.SEVERE_THUNDERSTORM, 96, max(base_mmhr, 9.5))
        return (WeatherCondition.THUNDERSTORM, 95, max(base_mmhr, 4.5))

    if any(k in joined for k in ("FZRA", "FZDZ", "PL", "IC")):
        if "FZRA" in joined:
            return (WeatherCondition.SNOW, 67 if "+" in joined else 66, max(base_mmhr, 2.5))
        if "FZDZ" in joined:
            return (WeatherCondition.SNOW, 57 if "+" in joined else 56, max(base_mmhr, 0.7))
        return (WeatherCondition.SNOW, 77, max(base_mmhr, 1.8))

    if any(k in joined for k in ("SN", "SG", "GS")):
        if "+" in joined:
            return (WeatherCondition.SNOW, 75, max(base_mmhr, 3.0))
        if "-" in joined:
            return (WeatherCondition.SNOW, 71, max(base_mmhr, 0.8))
        return (WeatherCondition.SNOW, 73, max(base_mmhr, 1.8))

    if any(k in joined for k in ("RA", "SH", "UP")):
        if "+" in joined:
            return (WeatherCondition.HEAVY_RAIN, 65, max(base_mmhr, 7.5))
        if "-" in joined:
            return (WeatherCondition.RAIN, 61, max(base_mmhr, 1.1))
        return (WeatherCondition.RAIN, 63, max(base_mmhr, 2.8))

    if "DZ" in joined:
        if "+" in joined:
            return (WeatherCondition.DRIZZLE, 55, max(base_mmhr, 1.0))
        if "-" in joined:
            return (WeatherCondition.DRIZZLE, 51, max(base_mmhr, 0.45))
        return (WeatherCondition.DRIZZLE, 53, max(base_mmhr, 0.7))

    if "FG" in joined:
        return (WeatherCondition.FOGGY, 45, 0.0)

    return (None, None, base_mmhr)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def elevation_adjust_temp(raw_temp_c: float, station_elev_m: float, target_elev_m: float) -> float:
    """
    Adjusts a station's temperature to the user's target elevation using the
    standard environmental lapse rate (6.5°C per 1000m).
    If a station is higher than the target, its air is cooler, so we add +6.5°C/km.
    """
    delta_km = (station_elev_m - target_elev_m) / 1000.0
    # Cap correction at +/- 6°C to avoid extreme distortion in inversions
    correction = max(-6.0, min(6.0, delta_km * settings.standard_lapse_rate_c_per_km))
    return round(raw_temp_c + correction, 2)


def elevation_adjust_dewpoint(
    raw_dew_c: float,
    station_elev_m: float,
    target_elev_m: float,
    max_temp_c: Optional[float] = None,
) -> float:
    """
    Adjusts a station's dewpoint temperature to target elevation (~2.0°C/km moist lapse)
    and ensures dewpoint never exceeds air temperature.
    """
    delta_km = (station_elev_m - target_elev_m) / 1000.0
    correction = max(-2.0, min(2.0, delta_km * 2.0))
    adj = float(raw_dew_c) + correction
    if max_temp_c is not None:
        adj = min(float(max_temp_c), adj)
    return round(adj, 2)


def run_spatial_buddy_qc(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Leave-one-out Median + MAD spatial buddy QC check across nearby stations.
    Avoids small-sample masking (N <= 5) where inclusive pstdev caps z <= sqrt(N-1).
    """
    n = len(candidates)
    if n == 0:
        return candidates
    adj_temps = [float(c["adj_temp_c"]) for c in candidates]
    mean_all = statistics.mean(adj_temps)
    stdev_all = statistics.pstdev(adj_temps) if n >= 3 else 0.0

    for i, c in enumerate(candidates):
        if n >= 3 and settings.zus_accuracy_v2:
            peers = [adj_temps[j] for j in range(n) if j != i]
            med = statistics.median(peers)
            mad = statistics.median([abs(x - med) for x in peers])
            scale = max(1.0, 1.4826 * mad)
            z_score = round(abs(adj_temps[i] - med) / scale, 2)
            qc_passed = z_score <= 3.0
            spatial_res_c = round(adj_temps[i] - med, 2)
        elif n >= 3:
            if stdev_all > 1.2:
                z_score = round(abs(adj_temps[i] - mean_all) / stdev_all, 2)
                qc_passed = z_score <= 2.0
            else:
                z_score = 0.0
                qc_passed = True
            peer_mean = (sum(adj_temps) - adj_temps[i]) / (n - 1)
            spatial_res_c = round(adj_temps[i] - peer_mean, 2)
        else:
            z_score = 0.0
            qc_passed = True
            spatial_res_c = (
                round(adj_temps[i] - ((sum(adj_temps) - adj_temps[i]) / (n - 1)), 2)
                if n == 2
                else 0.0
            )
        c["qc_passed"] = qc_passed
        c["qc_zscore"] = z_score
        c["spatial_residual_c"] = spatial_res_c
    return candidates


def calc_relative_humidity(temp_c: float, dewpoint_c: float) -> int:
    try:
        # Magnus formula
        a, b = 17.625, 243.04
        alpha_td = (a * dewpoint_c) / (b + dewpoint_c)
        alpha_t = (a * temp_c) / (b + temp_c)
        rh = 100.0 * math.exp(alpha_td - alpha_t)
        return int(max(5, min(100, round(rh))))
    except Exception:
        return 50


class HyperlocalStationService:
    """
    Ingests nearby official METARs, NWS Mesonet stations, and CWOP/APRS citizen
    weather stations, runs elevation lapse-rate normalization and Spatial Buddy Check QC,
    persists to PostGIS in a single batch transaction, and produces a hyperlocal ground-truth reading.
    """

    def __init__(self):
        self._station_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._max_cache_entries = 256
        self._station_cache_ttl = 300.0  # 5 minutes
        self._nws_points_cache: OrderedDict[str, str] = OrderedDict()
        self._inflight_locks: Dict[str, asyncio.Lock] = {}

    @staticmethod
    def _sector_key(lat: float, lon: float) -> str:
        # 0.05 deg (~5.5 km) spatial sector key for station observation reuse
        s_lat = round(lat * 20.0) / 20.0
        s_lon = round(lon * 20.0) / 20.0
        return f"{s_lat:.2f}:{s_lon:.2f}"

    @staticmethod
    def _compute_composite_and_readings(
        candidates: List[Dict[str, Any]],
        lat: float,
        lon: float,
        target_elev_m: float,
    ) -> Tuple[List[HyperlocalStationReading], Optional[Dict[str, Any]]]:
        if not candidates:
            return [], None

        # Recompute exact distance & target elevation adjustment for the caller's coordinates
        recomputed: List[Dict[str, Any]] = []
        for orig in candidates:
            c = dict(orig)
            c["dist_km"] = round(haversine_km(lat, lon, c["lat"], c["lon"]), 2)
            c["adj_temp_c"] = elevation_adjust_temp(c["raw_temp_c"], c["elev_m"], target_elev_m)
            recomputed.append(c)

        recomputed.sort(key=lambda x: x["dist_km"])
        recomputed = recomputed[:8]

        verified_readings: List[HyperlocalStationReading] = []
        valid_for_blend: List[Dict[str, Any]] = []

        for c in recomputed:
            qc_passed = bool(c.get("qc_passed", True))
            reading = HyperlocalStationReading(
                stationId=c["station_id"],
                stationName=c["name"],
                network=c["network"],
                distanceKm=c["dist_km"],
                elevationMeters=round(c["elev_m"], 1),
                rawTempCelsius=c["raw_temp_c"],
                elevationAdjustedTempCelsius=c["adj_temp_c"],
                humidity=c["humidity"],
                windSpeedKmh=c["wind_kmh"],
                windGustKmh=c.get("gust_kmh"),
                pressureHpa=c.get("pressure_hpa"),
                visibilityKm=c.get("visibility_km"),
                presentWeather=c.get("wx_string"),
                qcPassed=qc_passed,
                zScore=float(c.get("qc_zscore") or 0.0),
                reliabilityScore=c.get("reliability_score"),
                residualBiasCelsius=c.get("residual_bias_c"),
            )
            verified_readings.append(reading)
            if qc_passed:
                valid_for_blend.append(c)

        if not valid_for_blend:
            valid_for_blend = recomputed[:1]

        # Inverse Distance, Elevation, Age & Recursive Reliability Weighting (IDERW)
        total_w = 0.0
        blended_temp = 0.0
        blended_dewp = 0.0
        dewp_w = 0.0
        blended_press = 0.0
        press_w = 0.0
        blended_wind = 0.0
        wind_w = 0.0
        blended_gust = 0.0
        gust_w = 0.0
        blended_wdir_sin = 0.0
        blended_wdir_cos = 0.0
        wdir_w = 0.0
        blended_vis = 0.0
        vis_w = 0.0
        blended_hum = 0.0
        hum_w = 0.0

        for c in valid_for_blend:
            elev_diff_km = abs(c["elev_m"] - target_elev_m) / 1000.0
            rel_w = max(0.2, float(c.get("reliability_score") or 1.0))
            net_prior = 1.0 if "METAR" in str(c.get("network", "")) else 0.78
            age_min = float(c.get("obs_age_min") or 0.0)
            age_w = max(0.35, 1.0 - max(0.0, age_min - 15.0) / 90.0) if settings.zus_accuracy_v2 else 1.0
            w = (rel_w * net_prior * age_w) / (((c["dist_km"] + 1.5) ** 2) * (1.0 + elev_diff_km * 2.0))
            bias_adj = float(c.get("residual_bias_c") or 0.0) * (0.25 if "METAR" in str(c.get("network", "")) else 0.55)
            station_adj_t = c["adj_temp_c"] - bias_adj
            blended_temp += station_adj_t * w
            total_w += w
            if c.get("dewpoint_c") is not None:
                adj_dp = (
                    elevation_adjust_dewpoint(float(c["dewpoint_c"]), float(c["elev_m"]), target_elev_m, max_temp_c=station_adj_t)
                    if settings.zus_accuracy_v2
                    else float(c["dewpoint_c"])
                )
                blended_dewp += adj_dp * w
                dewp_w += w
            if c.get("pressure_hpa") is not None:
                blended_press += float(c["pressure_hpa"]) * w
                press_w += w
            if c.get("wind_kmh") is not None:
                blended_wind += float(c["wind_kmh"]) * w
                wind_w += w
            if c.get("gust_kmh") is not None:
                blended_gust += float(c["gust_kmh"]) * w
                gust_w += w
            if c.get("wind_dir") is not None:
                wd_rad = math.radians(float(c["wind_dir"]))
                blended_wdir_sin += math.sin(wd_rad) * w
                blended_wdir_cos += math.cos(wd_rad) * w
                wdir_w += w
            if c.get("visibility_km") is not None:
                blended_vis += float(c["visibility_km"]) * w
                vis_w += w
            if c.get("humidity") is not None:
                blended_hum += float(c["humidity"]) * w
                hum_w += w

        closest = valid_for_blend[0]
        active_wx_station = next(
            (
                c
                for c in valid_for_blend
                if c.get("wx_condition") is not None and c["dist_km"] <= 25.0
            ),
            None,
        )

        final_t = round(blended_temp / total_w, 2) if total_w > 0 else closest["adj_temp_c"]
        if dewp_w > 0:
            raw_dp = blended_dewp / dewp_w
            final_dp: Optional[float] = round(min(final_t, raw_dp) if settings.zus_accuracy_v2 else raw_dp, 2)
        else:
            final_dp = closest["dewpoint_c"]

        if settings.zus_accuracy_v2:
            final_wind = round(blended_wind / wind_w, 1) if wind_w > 0 else closest["wind_kmh"]
            final_gust = round(blended_gust / gust_w, 1) if gust_w > 0 else closest.get("gust_kmh")
            final_wdir = (
                int(round(math.degrees(math.atan2(blended_wdir_sin, blended_wdir_cos)))) % 360
                if wdir_w > 0
                else closest.get("wind_dir")
            )
            final_vis = round(blended_vis / vis_w, 1) if vis_w > 0 else closest.get("visibility_km")
            if final_dp is not None and hum_w > 0:
                thermo_rh = calc_relative_humidity(final_t, final_dp)
                final_hum: Optional[int] = int(max(5, min(100, round(0.60 * (blended_hum / hum_w) + 0.40 * thermo_rh))))
            elif final_dp is not None:
                final_hum = calc_relative_humidity(final_t, final_dp)
            elif hum_w > 0:
                final_hum = int(max(5, min(100, round(blended_hum / hum_w))))
            else:
                final_hum = closest["humidity"]
        else:
            final_wind = closest["wind_kmh"]
            final_gust = closest.get("gust_kmh")
            final_wdir = closest.get("wind_dir")
            final_vis = closest.get("visibility_km")
            final_hum = closest["humidity"]

        composite = {
            "temperature_c": final_t,
            "dewpoint_c": final_dp,
            "pressure_hpa": round(blended_press / press_w, 1) if press_w > 0 else closest.get("pressure_hpa"),
            "humidity": final_hum,
            "wind_kmh": final_wind,
            "gust_kmh": final_gust,
            "wind_dir": final_wdir,
            "visibility_km": final_vis,
            "closest_station_id": closest["station_id"],
            "closest_station_name": f"{closest['station_id']} ({closest['name']})",
            "closest_distance_km": closest["dist_km"],
            "wx_string": active_wx_station["wx_string"] if active_wx_station else closest.get("wx_string"),
            "wx_condition": active_wx_station["wx_condition"] if active_wx_station else None,
            "wx_wmo": active_wx_station["wx_wmo"] if active_wx_station else None,
            "precip_mmhr": active_wx_station["precip_mmhr"] if active_wx_station else 0.0,
            "wx_station_dist_km": active_wx_station["dist_km"] if active_wx_station else None,
        }
        return verified_readings, composite

    async def _fetch_single_nws_station_obs(
        self,
        feat: Dict[str, Any],
        lat: float,
        lon: float,
        target_elev_m: float,
    ) -> Optional[Dict[str, Any]]:
        props = feat.get("properties", {})
        st_id = props.get("stationIdentifier")
        if not st_id:
            return None
        client = http_pool.get_client()
        try:
            obs_resp = await client.get(
                f"{settings.nws_base_url}/stations/{st_id}/observations/latest",
                timeout=4.5,
            )
            if obs_resp.status_code != 200:
                return None
            obs_props = obs_resp.json().get("properties", {})
            now_utc = datetime.now(timezone.utc)
            obs_dt = _parse_obs_timestamp(obs_props.get("timestamp"))
            age_min = ((now_utc - obs_dt).total_seconds() / 60.0) if obs_dt is not None else 0.0
            if settings.zus_accuracy_v2 and obs_dt is not None and age_min > settings.station_max_age_minutes:
                return None

            t_val = obs_props.get("temperature", {}).get("value")
            coords = feat.get("geometry", {}).get("coordinates", [lon, lat])
            if t_val is None or len(coords) < 2:
                return None

            s_lon, s_lat = float(coords[0]), float(coords[1])
            elev_raw = props.get("elevation", {}).get("value")
            elev_m = float(elev_raw) if elev_raw is not None else target_elev_m
            dist_km = haversine_km(lat, lon, s_lat, s_lon)
            raw_temp_c = round(float(t_val), 2)
            adj_temp_c = elevation_adjust_temp(raw_temp_c, elev_m, target_elev_m)
            dewp_val = obs_props.get("dewpoint", {}).get("value")
            rh_val = obs_props.get("relativeHumidity", {}).get("value")

            wspd_obj = obs_props.get("windSpeed", {})
            wspd_val = wspd_obj.get("value")
            if wspd_val is not None and "m_s" in str(wspd_obj.get("unitCode") or ""):
                wspd_val = float(wspd_val) * 3.6
            wgst_obj = obs_props.get("windGust", {})
            wgst_val = wgst_obj.get("value")
            if wgst_val is not None and "m_s" in str(wgst_obj.get("unitCode") or ""):
                wgst_val = float(wgst_val) * 3.6

            wdir_val = obs_props.get("windDirection", {}).get("value")
            slp_pa = obs_props.get("seaLevelPressure", {}).get("value")
            vis_m = obs_props.get("visibility", {}).get("value")
            precip_obj = obs_props.get("precipitationLastHour", {})
            precip_val = precip_obj.get("value")
            precip_in_nws: Optional[float] = None
            if precip_val is not None and float(precip_val) > 0:
                u_code = str(precip_obj.get("unitCode") or "")
                precip_mm = float(precip_val) if "mm" in u_code else (float(precip_val) * 1000.0)
                precip_in_nws = precip_mm / 25.4

            raw_metar = obs_props.get("rawMessage") or ""
            text_desc = (obs_props.get("textDescription") or "").lower()
            wx_str_nws = None
            if "thunder" in text_desc or "tsra" in raw_metar.lower():
                wx_str_nws = "+TSRA" if "heavy" in text_desc else "TSRA"
            elif "freezing rain" in text_desc or "fzra" in raw_metar.lower():
                wx_str_nws = "+FZRA" if "heavy" in text_desc else "-FZRA"
            elif "freezing drizzle" in text_desc or "fzdz" in raw_metar.lower():
                wx_str_nws = "-FZDZ"
            elif "ice pellet" in text_desc or "sleet" in text_desc or " pl " in f" {raw_metar.lower()} ":
                wx_str_nws = "PL"
            elif "snow" in text_desc or " sn " in f" {raw_metar.lower()} ":
                wx_str_nws = "+SN" if "heavy" in text_desc else ("-SN" if "light" in text_desc else "SN")
            elif "rain" in text_desc or " -ra" in raw_metar.lower() or " +ra" in raw_metar.lower():
                wx_str_nws = "+RA" if ("heavy" in text_desc or "+ra" in raw_metar.lower()) else ("-RA" if "light" in text_desc else "RA")
            elif "drizzle" in text_desc:
                wx_str_nws = "-DZ"
            elif "fog" in text_desc:
                wx_str_nws = "FG"
            wx_cond, wx_wmo, wx_mmhr = parse_metar_wx(wx_str_nws, precip_in_nws)

            return {
                "station_id": st_id,
                "name": props.get("name", st_id),
                "network": "CWOP_MESONET" if len(st_id) > 4 else "NWS_METAR",
                "lat": s_lat,
                "lon": s_lon,
                "dist_km": round(dist_km, 2),
                "elev_m": elev_m,
                "raw_temp_c": raw_temp_c,
                "adj_temp_c": adj_temp_c,
                "dewpoint_c": round(float(dewp_val), 2) if dewp_val is not None else None,
                "humidity": int(round(float(rh_val))) if rh_val is not None else None,
                "wind_kmh": round(float(wspd_val), 1) if wspd_val is not None else None,
                "gust_kmh": round(float(wgst_val), 1) if wgst_val is not None else None,
                "wind_dir": _parse_wind_dir(wdir_val) if settings.zus_accuracy_v2 else (int(wdir_val) if wdir_val is not None else 0),
                "pressure_hpa": round(float(slp_pa) / 100.0, 1) if slp_pa is not None else None,
                "visibility_km": round(float(vis_m) / 1000.0, 1) if vis_m is not None else None,
                "wx_string": wx_str_nws,
                "wx_condition": wx_cond,
                "wx_wmo": wx_wmo,
                "precip_mmhr": wx_mmhr,
                "obs_age_min": round(max(0.0, age_min), 1),
            }
        except Exception:
            return None

    async def fetch_nearby_stations(
        self,
        lat: float,
        lon: float,
        target_elev_m: float = 0.0,
    ) -> Tuple[List[HyperlocalStationReading], Optional[Dict[str, Any]]]:
        sector = self._sector_key(lat, lon)
        now = datetime.now(timezone.utc)

        if sector in self._station_cache:
            entry = self._station_cache[sector]
            if (now - entry["cached_at"]).total_seconds() < self._station_cache_ttl:
                self._station_cache.move_to_end(sector)
                return self._compute_composite_and_readings(
                    entry["candidates"], lat, lon, target_elev_m
                )

        lock = self._inflight_locks.setdefault(sector, asyncio.Lock())
        async with lock:
            if sector in self._station_cache:
                entry = self._station_cache[sector]
                if (now - entry["cached_at"]).total_seconds() < self._station_cache_ttl:
                    self._station_cache.move_to_end(sector)
                    return self._compute_composite_and_readings(
                        entry["candidates"], lat, lon, target_elev_m
                    )

            raw_candidates: List[Dict[str, Any]] = []
            delta_deg = 0.65
            bbox = f"{lat - delta_deg:.3f},{lon - delta_deg:.3f},{lat + delta_deg:.3f},{lon + delta_deg:.3f}"
            client = http_pool.get_client()

            # 1. Query NOAA / AviationWeather Global METAR & Mesonet Bounding Box (~70km radius)
            try:
                resp = await client.get(
                    settings.noaa_metar_url,
                    params={"bbox": bbox, "format": "json"},
                    timeout=5.5,
                )
                if resp.status_code == 200:
                    metars = resp.json()
                    if isinstance(metars, list):
                        for m in metars:
                            obs_dt = _parse_obs_timestamp(m.get("obsTime") or m.get("reportTime"))
                            age_min = ((now - obs_dt).total_seconds() / 60.0) if obs_dt is not None else 0.0
                            if settings.zus_accuracy_v2 and obs_dt is not None and age_min > settings.station_max_age_minutes:
                                continue

                            temp = m.get("temp")
                            s_lat = m.get("lat")
                            s_lon = m.get("lon")
                            if temp is None or s_lat is None or s_lon is None:
                                continue
                            station_id = str(m.get("icaoId") or m.get("stationId") or "UNKNOWN")
                            name = str(m.get("name") or station_id)
                            raw_elev = m.get("elev")
                            elev_m = float(raw_elev) if raw_elev is not None else target_elev_m
                            dewp = m.get("dewp")
                            wspd_kt = m.get("wspd")
                            wgst_kt = m.get("wgst")
                            wdir = m.get("wdir")
                            slp = m.get("slp")
                            altim = m.get("altim")
                            visib_km = parse_metar_visib_km(m.get("visib"))
                            wx_str = m.get("wxString")
                            precip_in = m.get("precip")
                            wx_cond, wx_wmo, wx_mmhr = parse_metar_wx(
                                str(wx_str) if wx_str else None,
                                float(precip_in) if precip_in is not None else None,
                            )

                            dist_km = haversine_km(lat, lon, float(s_lat), float(s_lon))
                            raw_temp_c = float(temp)
                            adj_temp_c = elevation_adjust_temp(raw_temp_c, elev_m, target_elev_m)
                            rh = calc_relative_humidity(raw_temp_c, float(dewp)) if dewp is not None else None
                            wind_kmh = round(float(wspd_kt) * 1.852, 1) if wspd_kt is not None else None
                            gust_kmh = round(float(wgst_kt) * 1.852, 1) if wgst_kt is not None else None
                            wind_deg = _parse_wind_dir(wdir) if settings.zus_accuracy_v2 else (int(wdir) if isinstance(wdir, (int, float)) else 0)
                            msl_hpa = float(slp or altim) if (slp is not None or altim is not None) else None

                            network = "NWS_METAR" if station_id.startswith("K") else "GLOBAL_METAR"
                            raw_candidates.append(
                                {
                                    "station_id": station_id,
                                    "name": name,
                                    "network": network,
                                    "lat": float(s_lat),
                                    "lon": float(s_lon),
                                    "dist_km": round(dist_km, 2),
                                    "elev_m": elev_m,
                                    "raw_temp_c": raw_temp_c,
                                    "adj_temp_c": adj_temp_c,
                                    "dewpoint_c": float(dewp) if dewp is not None else None,
                                    "humidity": rh,
                                    "wind_kmh": wind_kmh,
                                    "gust_kmh": gust_kmh,
                                    "wind_dir": wind_deg,
                                    "pressure_hpa": round(msl_hpa, 1) if msl_hpa is not None else None,
                                    "visibility_km": visib_km,
                                    "wx_string": str(wx_str).strip() if wx_str else None,
                                    "wx_condition": wx_cond,
                                    "wx_wmo": wx_wmo,
                                    "precip_mmhr": wx_mmhr,
                                    "obs_age_min": round(max(0.0, age_min), 1),
                                }
                            )
            except Exception:
                pass

            # 2. If in the US and (<3 METAR candidates OR nearest METAR >20km), fetch NWS Mesonet stations in parallel
            nearest_dist_km = min((c["dist_km"] for c in raw_candidates), default=999.0)
            need_nws_fallback = len(raw_candidates) < 3 or (settings.zus_accuracy_v2 and nearest_dist_km > 20.0)
            if db_manager.is_us_region(lat, lon) and need_nws_fallback:
                try:
                    pt_key = f"{lat:.2f},{lon:.2f}"
                    stations_url = self._nws_points_cache.get(pt_key)
                    if not stations_url:
                        pt_resp = await client.get(
                            f"{settings.nws_base_url}/points/{lat:.4f},{lon:.4f}",
                            timeout=4.5,
                        )
                        if pt_resp.status_code == 200:
                            stations_url = pt_resp.json().get("properties", {}).get("observationStations")
                            if stations_url:
                                self._nws_points_cache[pt_key] = str(stations_url)
                                if len(self._nws_points_cache) > 512:
                                    self._nws_points_cache.popitem(last=False)

                    if stations_url:
                        st_resp = await client.get(stations_url, params={"limit": 5}, timeout=4.5)
                        if st_resp.status_code == 200:
                            features = st_resp.json().get("features", [])
                            existing_ids = {c["station_id"] for c in raw_candidates}
                            max_extra = 2 if len(raw_candidates) >= 3 else 3
                            to_fetch = [
                                feat
                                for feat in features[:4]
                                if feat.get("properties", {}).get("stationIdentifier") not in existing_ids
                            ][:max_extra]
                            if to_fetch:
                                nws_results = await asyncio.gather(
                                    *(
                                        self._fetch_single_nws_station_obs(feat, lat, lon, target_elev_m)
                                        for feat in to_fetch
                                    ),
                                    return_exceptions=True,
                                )
                                for r in nws_results:
                                    if isinstance(r, dict):
                                        raw_candidates.append(r)
                except Exception:
                    pass

            if not raw_candidates:
                return [], None

            raw_candidates.sort(key=lambda x: x["dist_km"])
            raw_candidates = raw_candidates[:8]

            # 3. Leave-One-Out MAD Spatial Buddy Check QC
            run_spatial_buddy_qc(raw_candidates)

            # 4. Batch-persist all stations & observations in 1 transaction
            rel_map = await db_manager.batch_upsert_stations_and_observations(
                records=raw_candidates,
                observed_at=now.replace(second=0, microsecond=0),
            )
            for c in raw_candidates:
                rel_score, res_bias_c = rel_map.get(c["station_id"], (1.0, 0.0))
                c["reliability_score"] = rel_score
                c["residual_bias_c"] = res_bias_c

            self._station_cache[sector] = {
                "cached_at": now,
                "candidates": raw_candidates,
            }
            while len(self._station_cache) > self._max_cache_entries:
                self._station_cache.popitem(last=False)

            return self._compute_composite_and_readings(
                raw_candidates, lat, lon, target_elev_m
            )


station_service = HyperlocalStationService()
