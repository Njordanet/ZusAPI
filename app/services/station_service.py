import math
import httpx
import statistics
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Dict, Any
from app.config import settings
from app.models.weather import HyperlocalStationReading, WeatherCondition
from app.services.db import db_manager


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

    if any(k in joined for k in ("SN", "SG", "PL", "IC", "GS")):
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
    persists to PostGIS, and produces a hyperlocal ground-truth reading.
    """

    async def fetch_nearby_stations(
        self,
        lat: float,
        lon: float,
        target_elev_m: float = 0.0,
    ) -> Tuple[List[HyperlocalStationReading], Optional[Dict[str, Any]]]:
        raw_candidates: List[Dict[str, Any]] = []

        # 1. Query NOAA / AviationWeather Global METAR & Mesonet Bounding Box (~70km radius)
        # Note: AviationWeather bbox parameter format is minLat,minLon,maxLat,maxLon
        delta_deg = 0.65
        bbox = f"{lat - delta_deg:.3f},{lon - delta_deg:.3f},{lat + delta_deg:.3f},{lon + delta_deg:.3f}"

        headers = {"User-Agent": "ZusWeatherEngine/2.0 (panda-telemetry)"}
        async with httpx.AsyncClient(timeout=6.0, headers=headers) as client:
            try:
                resp = await client.get(
                    settings.noaa_metar_url,
                    params={"bbox": bbox, "format": "json"},
                )
                if resp.status_code == 200:
                    metars = resp.json()
                    if isinstance(metars, list):
                        for m in metars:
                            temp = m.get("temp")
                            s_lat = m.get("lat")
                            s_lon = m.get("lon")
                            if temp is None or s_lat is None or s_lon is None:
                                continue
                            station_id = str(m.get("icaoId") or m.get("stationId") or "UNKNOWN")
                            name = str(m.get("name") or station_id)
                            elev_m = float(m.get("elev") or target_elev_m)
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
                            wind_deg = int(wdir) if isinstance(wdir, (int, float)) else 0
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
                                }
                            )
            except Exception:
                pass

            # 2. If in the US and we want hyperlocal density, also check NWS Mesonet/CWOP observation stations
            if db_manager.is_us_region(lat, lon) and len(raw_candidates) < 3:
                try:
                    pt_resp = await client.get(f"{settings.nws_base_url}/points/{lat:.4f},{lon:.4f}")
                    if pt_resp.status_code == 200:
                        stations_url = pt_resp.json().get("properties", {}).get("observationStations")
                        if stations_url:
                            st_resp = await client.get(stations_url, params={"limit": 5})
                            if st_resp.status_code == 200:
                                features = st_resp.json().get("features", [])
                                for feat in features[:3]:
                                    props = feat.get("properties", {})
                                    st_id = props.get("stationIdentifier")
                                    if not st_id or any(c["station_id"] == st_id for c in raw_candidates):
                                        continue
                                    obs_resp = await client.get(f"{settings.nws_base_url}/stations/{st_id}/observations/latest")
                                    if obs_resp.status_code == 200:
                                        obs_props = obs_resp.json().get("properties", {})
                                        t_val = obs_props.get("temperature", {}).get("value")
                                        coords = feat.get("geometry", {}).get("coordinates", [lon, lat])
                                        if t_val is not None and len(coords) >= 2:
                                            s_lon, s_lat = float(coords[0]), float(coords[1])
                                            elev_m = float(props.get("elevation", {}).get("value") or target_elev_m)
                                            dist_km = haversine_km(lat, lon, s_lat, s_lon)
                                            raw_temp_c = round(float(t_val), 2)
                                            adj_temp_c = elevation_adjust_temp(raw_temp_c, elev_m, target_elev_m)
                                            dewp_val = obs_props.get("dewpoint", {}).get("value")
                                            rh_val = obs_props.get("relativeHumidity", {}).get("value")
                                            wspd_val = obs_props.get("windSpeed", {}).get("value")
                                            wgst_val = obs_props.get("windGust", {}).get("value")
                                            wdir_val = obs_props.get("windDirection", {}).get("value")
                                            slp_pa = obs_props.get("seaLevelPressure", {}).get("value")
                                            vis_m = obs_props.get("visibility", {}).get("value")
                                            raw_metar = obs_props.get("rawMessage") or ""
                                            text_desc = (obs_props.get("textDescription") or "").lower()
                                            wx_str_nws = None
                                            if "rain" in text_desc or " -ra" in raw_metar.lower():
                                                wx_str_nws = "-RA" if "light" in text_desc else "RA"
                                            elif "drizzle" in text_desc:
                                                wx_str_nws = "-DZ"
                                            elif "snow" in text_desc:
                                                wx_str_nws = "-SN"
                                            elif "thunder" in text_desc:
                                                wx_str_nws = "TSRA"
                                            wx_cond, wx_wmo, wx_mmhr = parse_metar_wx(wx_str_nws, None)

                                            raw_candidates.append(
                                                {
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
                                                    "wind_dir": int(wdir_val) if wdir_val is not None else 0,
                                                    "pressure_hpa": round(float(slp_pa) / 100.0, 1) if slp_pa is not None else None,
                                                    "visibility_km": round(float(vis_m) / 1000.0, 1) if vis_m is not None else None,
                                                    "wx_string": wx_str_nws,
                                                    "wx_condition": wx_cond,
                                                    "wx_wmo": wx_wmo,
                                                    "precip_mmhr": wx_mmhr,
                                                }
                                            )
                except Exception:
                    pass

        if not raw_candidates:
            return [], None

        # Sort by distance ascending and take closest 8 stations
        raw_candidates.sort(key=lambda x: x["dist_km"])
        raw_candidates = raw_candidates[:8]

        # 3. Run Spatial Buddy Check QC (+/- 2.0 sigma outlier rejection on elevation-adjusted temps)
        adj_temps = [c["adj_temp_c"] for c in raw_candidates]
        mean_t = statistics.mean(adj_temps)
        stdev_t = statistics.pstdev(adj_temps) if len(adj_temps) >= 3 else 0.0

        verified_readings: List[HyperlocalStationReading] = []
        valid_for_blend: List[Dict[str, Any]] = []
        now = datetime.now(timezone.utc)

        for c in raw_candidates:
            if stdev_t > 1.2:
                z_score = round(abs(c["adj_temp_c"] - mean_t) / stdev_t, 2)
                qc_passed = z_score <= 2.0
            else:
                z_score = 0.0
                qc_passed = True

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
                zScore=z_score,
            )
            verified_readings.append(reading)

            if qc_passed:
                valid_for_blend.append(c)

            # Persist station & observation to PostGIS
            await db_manager.upsert_station_and_observation(
                station_id=c["station_id"],
                name=c["name"],
                network=c["network"],
                lat=c["lat"],
                lon=c["lon"],
                elevation_m=c["elev_m"],
                observed_at=now.replace(second=0, microsecond=0),
                temp_c=c["raw_temp_c"],
                dewpoint_c=c["dewpoint_c"],
                humidity=c["humidity"],
                wind_kmh=c["wind_kmh"],
                wind_dir=c["wind_dir"],
                pressure_hpa=c["pressure_hpa"],
                precip_rate_mm=c.get("precip_mmhr", 0.0),
                qc_passed=qc_passed,
                qc_zscore=z_score,
            )

        if not valid_for_blend:
            valid_for_blend = raw_candidates[:1]

        # 4. Inverse Distance & Elevation Weighting (IDEW)
        total_w = 0.0
        blended_temp = 0.0
        blended_dewp = 0.0
        dewp_w = 0.0
        blended_press = 0.0
        press_w = 0.0

        for c in valid_for_blend:
            elev_diff_km = abs(c["elev_m"] - target_elev_m) / 1000.0
            # Weight decays with distance^2 and elevation difference
            w = 1.0 / (((c["dist_km"] + 1.5) ** 2) * (1.0 + elev_diff_km * 2.0))
            blended_temp += c["adj_temp_c"] * w
            total_w += w
            if c["dewpoint_c"] is not None:
                blended_dewp += c["dewpoint_c"] * w
                dewp_w += w
            if c.get("pressure_hpa") is not None:
                blended_press += c["pressure_hpa"] * w
                press_w += w

        closest = valid_for_blend[0]
        # Find nearest station within 25km reporting active present weather (e.g. -RA, -DZ, SN, TS)
        active_wx_station = next(
            (
                c
                for c in valid_for_blend
                if c.get("wx_condition") is not None and c["dist_km"] <= 25.0
            ),
            None,
        )

        composite = {
            "temperature_c": round(blended_temp / total_w, 2) if total_w > 0 else closest["adj_temp_c"],
            "dewpoint_c": round(blended_dewp / dewp_w, 2) if dewp_w > 0 else closest["dewpoint_c"],
            "pressure_hpa": round(blended_press / press_w, 1) if press_w > 0 else closest.get("pressure_hpa"),
            "humidity": closest["humidity"],
            "wind_kmh": closest["wind_kmh"],
            "gust_kmh": closest.get("gust_kmh"),
            "wind_dir": closest.get("wind_dir"),
            "visibility_km": closest.get("visibility_km"),
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


station_service = HyperlocalStationService()
