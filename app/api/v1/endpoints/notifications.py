from fastapi import APIRouter, HTTPException, Query, Body, Request, status
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone, timedelta
import time

from app.models.notifications import (
    PushSubscriptionRegisterRequest,
    PushSubscriptionResponse,
    PushSubscriptionUnregisterRequest,
    NotificationStatsResponse,
)
from app.services.push_worker import push_worker
from app.services.weather_provider import weather_service

router = APIRouter(prefix="/notifications", tags=["Notifications"])

# Lightweight in-memory widget telemetry relay keyed by client IP (+ global fallback)
_WIDGET_SYNC_CACHE: Dict[str, Dict[str, Any]] = {}
_WIDGET_STALE_REFRESH_MS = 15 * 60 * 1000.0  # 15 minutes


def _wmo_to_base_symbol(code: int, is_day: bool) -> str:
    if code == 0:
        return "sun.max" if is_day else "moon.stars"
    if code in (1, 2):
        return "cloud.sun" if is_day else "cloud.moon"
    if code in (3, 45, 48):
        return "cloud"
    if code in (51, 53, 55, 61, 63, 65, 80, 81, 82):
        return "cloud.rain"
    if code in (56, 57, 66, 67, 71, 73, 75, 77, 85, 86):
        return "cloud.snow"
    if code in (95, 96, 99):
        return "cloud.bolt.rain"
    return "cloud.sun" if is_day else "cloud.moon"


def _compute_rain_strings(rates: List[float]) -> tuple[str, str, str]:
    threshold = 0.12
    first_idx = next((i for i, r in enumerate(rates) if r >= threshold), -1)
    if first_idx > 0:
        last_idx = first_idx
        for i in range(first_idx, len(rates)):
            if rates[i] >= threshold:
                last_idx = i
            elif i - first_idx > 3:
                break
        dur = max(5, last_idx - first_idx)
        return (
            f"Rain in {first_idx} min",
            f"Rain starting in {first_idx} min",
            f"Light rain stopping {dur} min later",
        )
    if first_idx == 0:
        stop_idx = next(
            (i for i in range(3, len(rates)) if rates[i] < threshold),
            28,
        )
        return (
            f"Rain for {stop_idx} min",
            "Light rain currently falling",
            f"Stopping in {stop_idx} min",
        )
    return (
        "No precip (1h)",
        "No precipitation expected",
        "Dry conditions for the next hour",
    )


def _loc_cache_key(lat: float, lon: float, unit: str) -> str:
    return f"loc:{lat:.2f}:{lon:.2f}:{unit.upper()}"


async def _refresh_stale_widget_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    lat = float(entry.get("latitude", 40.7128))
    lon = float(entry.get("longitude", -74.0060))
    unit = str(entry.get("unitSymbol", "F")).upper()
    loc_name = str(entry.get("locationName") or "Current Location")

    def conv(c_val: float) -> int:
        val = c_val if unit == "C" else (c_val * 9.0 / 5.0 + 32.0)
        return int(round(val))

    wx = await weather_service.get_forecast(latitude=lat, longitude=lon, city_name=loc_name)
    cur = wx.current
    tz = timezone(timedelta(seconds=wx.utcOffsetSeconds or 0))
    now_local = datetime.now(tz)
    is_day = cur.isDay if cur.isDay is not None else (6 <= now_local.hour < 20)
    wmo = cur.observedWeatherCode if cur.observedWeatherCode is not None else 0
    cond_label = str(cur.condition.value if hasattr(cur.condition, "value") else cur.condition)

    minute_rates: List[float] = [0.0] * 61
    if wx.dopplerTelemetry and len(wx.dopplerTelemetry.minuteByMinuteRates) >= 61:
        minute_rates = list(wx.dopplerTelemetry.minuteByMinuteRates[:61])

    # Ground-truth clutter suppression matching the mobile app's buildSixtyMinuteNowcast:
    # prevent weak overcast virga / ground clutter (< 20 dBZ, < 0.35 mm/hr) from hijacking dry skies.
    h0 = wx.hourly[0] if wx.hourly else None
    h0_wmo = int(h0.wmoCode) if (h0 and h0.wmoCode is not None) else wmo
    h0_pop = int(h0.precipitationChance or 0) if h0 else 0
    cur_precip = float(cur.precipIntensityMmHr or 0.0)
    cur_dbz = float(wx.dopplerTelemetry.currentDbz or 0.0) if wx.dopplerTelemetry else 0.0
    max_rate = max(minute_rates) if minute_rates else 0.0

    if wmo < 51 and h0_wmo < 51 and cur_precip < 0.12 and cur_dbz < 20.0 and max_rate < 0.35:
        minute_rates = [0.0] * 61
        max_rate = 0.0

    # Downgrade uncorroborated NWP light-drizzle bias under dry overcast cloud decks
    if (
        wmo in (51, 53)
        and cur.localStationWx is None
        and cur_precip < 0.08
        and max_rate < 0.12
        and cur_dbz < 18.0
        and h0_pop < 30
    ):
        wmo = 3
        cond_label = "Overcast"

    cond_sym = _wmo_to_base_symbol(wmo, is_day)

    cur_temp = conv(cur.temperature)
    feels_temp = conv(cur.feelsLike)
    high_temp = conv(cur.tempMax)
    low_temp = conv(cur.tempMin)

    # Only activate the 60-minute precipitation curve UI when meaningful rates exist on the curve
    wet_minutes = sum(1 for r in minute_rates if r >= 0.12)
    is_precip_soon = wet_minutes >= 2
    short_rain, start_line, stop_line = _compute_rain_strings(minute_rates)

    hourly_items: List[Dict[str, Any]] = []
    for idx, hr in enumerate(wx.hourly[:12]):
        hr_local = hr.time.astimezone(tz) if hr.time.tzinfo else hr.time.replace(tzinfo=timezone.utc).astimezone(tz)
        h_num = hr_local.hour
        h12 = 12 if h_num % 12 == 0 else h_num % 12
        ampm = "P" if h_num >= 12 else "A"
        label = "NOW" if idx == 0 else f"{h12}{ampm}"
        hr_day = hr.isDay if hr.isDay is not None else (6 <= h_num < 20)
        hr_wmo = wmo if idx == 0 else hr.wmoCode
        hr_cond_label = (
            cond_label
            if idx == 0
            else str(hr.condition.value if hasattr(hr.condition, "value") else hr.condition)
        )
        hourly_items.append(
            {
                "label": label,
                "temp": conv(hr.temperature),
                "symbolName": _wmo_to_base_symbol(hr_wmo, hr_day),
                "isRain": hr_wmo >= 51,
                "precipProb": int(hr.precipitationChance or 0),
                "conditionLabel": hr_cond_label,
            }
        )

    daily_items: List[Dict[str, Any]] = []
    for idx, day in enumerate(wx.daily[:7]):
        day_dt = day.date.astimezone(tz) if day.date.tzinfo else day.date
        d_label = "TODAY" if idx == 0 else day_dt.strftime("%a").upper()
        daily_items.append(
            {
                "dayLabel": d_label,
                "precipPct": int(day.precipitationChance or 0),
                "symbolName": f"{_wmo_to_base_symbol(day.wmoCode, True)}.fill",
                "minTemp": conv(day.minTemperature),
                "maxTemp": conv(day.maxTemperature),
                "isToday": idx == 0,
            }
        )

    summary = (
        cur.nowcastSummary
        or f"{cond_label} throughout the day with temperatures reaching a high of {high_temp}°{unit} and an overnight low of {low_temp}°{unit}."
    )
    now_ms = time.time() * 1000.0
    return {
        **entry,
        "timestampMs": now_ms,
        "syncedAtMs": now_ms,
        "currentTemp": cur_temp,
        "feelsLike": feels_temp,
        "highTemp": high_temp,
        "lowTemp": low_temp,
        "conditionLabel": cond_label,
        "conditionSymbol": cond_sym,
        "isPrecipSoon": is_precip_soon,
        "shortRainLabel": short_rain,
        "rainStartingLine": start_line,
        "rainStoppingLine": stop_line,
        "dailySummaryLine": summary,
        "minuteRates": minute_rates,
        "hourlyItems": hourly_items or entry.get("hourlyItems", []),
        "dailyItems": daily_items or entry.get("dailyItems", []),
    }


def _extract_client_ip(request: Request) -> str:
    cf_ip = request.headers.get("cf-connecting-ip")
    if cf_ip:
        return cf_ip.strip()
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "default"



@router.post(
    "/register",
    response_model=PushSubscriptionResponse,
    status_code=status.HTTP_200_OK,
    summary="Register or update client push notification subscription",
    description=(
        "Registers a client device for spatial precipitation and severe weather push notifications. "
        "Server quantizes coordinates to 0.05° (~5km) cells and overwrites in-place without retaining "
        "historical trails."
    ),
)
async def register_subscription(
    payload: PushSubscriptionRegisterRequest,
) -> PushSubscriptionResponse:
    if not payload.device_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="device_id cannot be empty",
        )
    return await push_worker.register_subscription(payload)


@router.delete(
    "/unregister",
    status_code=status.HTTP_200_OK,
    summary="Unregister client push notification subscription",
    description="Removes device subscription and stops push notifications.",
)
async def unregister_subscription(
    device_id: Optional[str] = Query(None, description="Device identifier to unregister"),
    body: Optional[PushSubscriptionUnregisterRequest] = Body(None),
) -> Dict[str, Any]:
    target_id = (device_id or (body.device_id if body else None) or "").strip()
    if not target_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="device_id query parameter or body is required",
        )

    deleted = await push_worker.unregister_subscription(target_id)
    return {
        "device_id": target_id,
        "unregistered": deleted,
        "status": "success" if deleted else "not_found",
    }


@router.get(
    "/stats",
    response_model=NotificationStatsResponse,
    summary="Retrieve spatial subscription registry diagnostics and stats",
    description="Returns aggregate subscription numbers, spatial cells, cell state machine counts, and recent dispatch metrics.",
)
async def get_notification_stats() -> NotificationStatsResponse:
    return await push_worker.get_stats()


@router.post(
    "/evaluate",
    summary="Manually trigger an anti-spam spatial evaluation pass",
    description="Executes a single cycle of the O(1) cell evaluation engine for testing and verification.",
)
async def trigger_evaluation() -> Dict[str, Any]:
    return await push_worker.evaluate_and_dispatch()


@router.post(
    "/widget-sync",
    summary="Sync active weather telemetry snapshot for iOS WidgetKit extension",
)
async def post_widget_sync(
    request: Request,
    payload: Dict[str, Any] = Body(...),
) -> Dict[str, Any]:
    client_ip = _extract_client_ip(request)
    entry = {
        **payload,
        "syncedAtMs": time.time() * 1000.0,
    }
    _WIDGET_SYNC_CACHE[client_ip] = entry

    dev_id = str(payload.get("deviceId") or "").strip()
    if dev_id:
        _WIDGET_SYNC_CACHE[f"dev:{dev_id}"] = entry

    if not bool(payload.get("locationRequired")):
        try:
            p_lat = float(payload["latitude"])
            p_lon = float(payload["longitude"])
            p_unit = str(payload.get("unitSymbol") or "F")
            _WIDGET_SYNC_CACHE[_loc_cache_key(p_lat, p_lon, p_unit)] = entry
        except (KeyError, TypeError, ValueError):
            pass

    # Prune if > 500 entries to conserve memory
    while len(_WIDGET_SYNC_CACHE) > 500:
        oldest_key = next(iter(_WIDGET_SYNC_CACHE), None)
        if not oldest_key:
            break
        _WIDGET_SYNC_CACHE.pop(oldest_key, None)
    return {"status": "ok", "client_ip": client_ip}


@router.get(
    "/widget-sync",
    summary="Retrieve active weather telemetry snapshot for iOS WidgetKit extension",
)
async def get_widget_sync(
    request: Request,
    device_id: Optional[str] = Query(None),
    push_device_id: Optional[str] = Query(None),
    lat: Optional[float] = Query(None),
    lon: Optional[float] = Query(None),
    unit: Optional[str] = Query(None),
    location_name: Optional[str] = Query(None),
    gps_live: bool = Query(False),
) -> Dict[str, Any]:
    client_ip = _extract_client_ip(request)
    cache_key: Optional[str] = None
    entry: Optional[Dict[str, Any]] = None

    if device_id and f"dev:{device_id.strip()}" in _WIDGET_SYNC_CACHE:
        cache_key = f"dev:{device_id.strip()}"
        entry = _WIDGET_SYNC_CACHE[cache_key]
    elif lat is not None and lon is not None and not gps_live:
        loc_key = _loc_cache_key(lat, lon, unit or "F")
        if loc_key in _WIDGET_SYNC_CACHE:
            cache_key = loc_key
            entry = _WIDGET_SYNC_CACHE[loc_key]
        elif client_ip in _WIDGET_SYNC_CACHE:
            ip_entry = _WIDGET_SYNC_CACHE[client_ip]
            if (
                ip_entry.get("locationRequired") is True
                or (
                    abs(float(ip_entry.get("latitude", 0.0)) - lat) <= 0.05
                    and abs(float(ip_entry.get("longitude", 0.0)) - lon) <= 0.05
                )
            ):
                cache_key = client_ip
                entry = ip_entry
    elif client_ip in _WIDGET_SYNC_CACHE:
        cache_key = client_ip
        entry = _WIDGET_SYNC_CACHE[client_ip]

    if gps_live and lat is not None and lon is not None:
        target_push_id = (
            (push_device_id or "").strip()
            or str((entry or {}).get("pushDeviceId") or "").strip()
        )
        if target_push_id:
            await push_worker.update_subscription_location_if_moved(
                target_push_id, lat, lon, city=location_name
            )

        effective_unit = (unit or str((entry or {}).get("unitSymbol") or "F")).upper()
        moved = (
            (entry is None)
            or bool(entry.get("locationRequired"))
            or abs(float(entry.get("latitude", 0.0)) - lat) > 0.02
            or abs(float(entry.get("longitude", 0.0)) - lon) > 0.02
        )
        if moved:
            loc_label = (location_name or "").strip()
            if not loc_label:
                loc_label = "Current Location"
            seed = {
                **(entry or {}),
                "latitude": lat,
                "longitude": lon,
                "unitSymbol": effective_unit,
                "locationName": loc_label,
                "locationRequired": False,
            }
            if device_id and device_id.strip():
                seed["deviceId"] = device_id.strip()
            if target_push_id:
                seed["pushDeviceId"] = target_push_id

            refreshed = await _refresh_stale_widget_entry(seed)
            if seed.get("deviceId"):
                _WIDGET_SYNC_CACHE[f"dev:{seed['deviceId']}"] = refreshed
            _WIDGET_SYNC_CACHE[_loc_cache_key(lat, lon, effective_unit)] = refreshed
            _WIDGET_SYNC_CACHE[client_ip] = refreshed
            return refreshed
        else:
            if target_push_id and entry and not entry.get("pushDeviceId"):
                entry["pushDeviceId"] = target_push_id

    if entry and entry.get("locationRequired") is True:
        return entry

    if not entry and lat is not None and lon is not None:
        seed = {
            "latitude": lat,
            "longitude": lon,
            "unitSymbol": (unit or "F").upper(),
            "locationName": location_name or "Current Location",
        }
        if device_id and device_id.strip():
            seed["deviceId"] = device_id.strip()
        refreshed = await _refresh_stale_widget_entry(seed)
        _WIDGET_SYNC_CACHE[_loc_cache_key(lat, lon, unit or "F")] = refreshed
        return refreshed

    if not entry:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No widget snapshot synced yet")

    now_ms = time.time() * 1000.0
    entry_ts = float(entry.get("timestampMs") or entry.get("syncedAtMs") or 0.0)
    if now_ms - entry_ts > _WIDGET_STALE_REFRESH_MS:
        try:
            refreshed = await _refresh_stale_widget_entry(entry)
            if cache_key:
                _WIDGET_SYNC_CACHE[cache_key] = refreshed
            dev_id = str(refreshed.get("deviceId") or "").strip()
            if dev_id:
                _WIDGET_SYNC_CACHE[f"dev:{dev_id}"] = refreshed
            try:
                _WIDGET_SYNC_CACHE[
                    _loc_cache_key(
                        float(refreshed["latitude"]),
                        float(refreshed["longitude"]),
                        str(refreshed.get("unitSymbol") or "F"),
                    )
                ] = refreshed
            except (KeyError, TypeError, ValueError):
                pass
            _WIDGET_SYNC_CACHE[client_ip] = refreshed
            return refreshed
        except Exception:
            pass
    return entry

