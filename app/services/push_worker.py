import asyncio
import logging
import math
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

import httpx

from app.config import settings
from app.models.notifications import (
    PushSubscriptionRegisterRequest,
    PushSubscriptionResponse,
    NotificationStatsResponse,
)
from app.models.weather import WeatherAlertModel
from app.services.db import db_manager
from app.services.doppler_service import doppler_service
from app.services.http_client import http_pool
from app.services.weather_provider import weather_service

logger = logging.getLogger("zus.push_worker")

# 90-minute storm cooldown window
COOLDOWN_SECONDS = 90 * 60

# Precipitation sensitivity thresholds
PRECIP_SENSITIVITY_THRESHOLDS = {
    "any": 0.25,       # 0.25 mm/h (~18 dBZ)
    "moderate": 1.5,   # 1.50 mm/h (~30 dBZ)
    "heavy": 6.0,      # 6.00 mm/h (~42 dBZ)
}

PRECIP_DBZ_THRESHOLDS = {
    "any": 18.0,
    "moderate": 30.0,
    "heavy": 42.0,
}


def quantize_cell(lat: float, lon: float) -> Tuple[float, float]:
    """
    Quantizes geographic coordinates to a 0.05° (~5km) spatial grid cell.
    cell_lat = round(lat * 20) / 20
    cell_lon = round(lon * 20) / 20
    """
    cell_lat = round(lat * 20.0) / 20.0
    cell_lon = round(lon * 20.0) / 20.0
    return round(cell_lat, 4), round(cell_lon, 4)


def is_in_dnd_window(now_utc: datetime, tz_name: str, start_time: str, end_time: str) -> bool:
    """
    Evaluates whether the given UTC timestamp falls within the user's local DND window.
    Supports exact-minute local time and overnight wrap-around (e.g. 22:15 -> 06:45).
    """
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = ZoneInfo("UTC")

    local_dt = now_utc.astimezone(tz)
    cur_min = local_dt.hour * 60 + local_dt.minute

    try:
        s_parts = [int(p) for p in start_time.strip().split(":")]
        e_parts = [int(p) for p in end_time.strip().split(":")]
        s_min = s_parts[0] * 60 + s_parts[1]
        e_min = e_parts[0] * 60 + e_parts[1]
    except Exception:
        s_min = 22 * 60
        e_min = 7 * 60

    if s_min == e_min:
        return False

    if s_min < e_min:
        # Same-day window (e.g. 13:00 to 15:00)
        return s_min <= cur_min < e_min
    else:
        # Overnight wrap-around window (e.g. 22:15 to 06:45)
        return cur_min >= s_min or cur_min < e_min


class PushWorker:
    """
    Scalable O(1) Spatial Cell Push Notification & Anti-Spam Engine:
    - Quantizes devices into 0.05° (~5km) spatial cells
    - O(1) RAM NEXRAD Doppler 60-min nowcast evaluation per cell
    - Anti-bounce hysteresis state machine (IDLE, APPROACHING, ACTIVE, COOLDOWN) + 90-min cooldown
    - Severe weather (NWS) polygon deduplication & DND breakthrough
    - Batch Expo Push Dispatch (up to 100 per chunk)
    """

    def __init__(self):
        self._db_path = self._resolve_db_path()
        self._lock = asyncio.Lock()
        self._sqlite_lock = threading_lock = None  # thread-safety handled per connection
        self._cell_alerts_cache: Dict[Tuple[float, float], Tuple[float, List[WeatherAlertModel]]] = {}
        self._cell_alert_cache_ttl = 300.0  # 5 minutes

        # Diagnostic counters
        self._last_run_at: Optional[str] = None
        self._last_run_duration_ms: Optional[float] = None
        self._recent_dispatches = 0
        self._recent_suppressed_dnd = 0
        self._recent_suppressed_cooldown = 0
        self._recent_suppressed_severe_active = 0

        self._init_sqlite()

    def _resolve_db_path(self) -> Path:
        try:
            settings.data_dir.mkdir(parents=True, exist_ok=True)
            test_file = settings.data_dir / ".write_test"
            test_file.touch()
            test_file.unlink()
            return settings.data_dir / "push_subscriptions.db"
        except Exception:
            fallback = Path(__file__).resolve().parent.parent.parent / "data"
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback / "push_subscriptions.db"

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _init_sqlite(self):
        with self._get_connection() as conn:
            # 1. Device subscriptions table (overwriting coordinates in-place, zero history trail)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS push_subscriptions (
                    device_id TEXT PRIMARY KEY,
                    expo_push_token TEXT,
                    lat REAL NOT NULL,
                    lon REAL NOT NULL,
                    cell_lat REAL NOT NULL,
                    cell_lon REAL NOT NULL,
                    timezone TEXT NOT NULL DEFAULT 'UTC',
                    notifications_enabled INTEGER NOT NULL DEFAULT 1,
                    dnd_enabled INTEGER NOT NULL DEFAULT 0,
                    dnd_start_time TEXT NOT NULL DEFAULT '22:00',
                    dnd_end_time TEXT NOT NULL DEFAULT '07:00',
                    dnd_severe_breakthrough INTEGER NOT NULL DEFAULT 1,
                    precip_alerts_enabled INTEGER NOT NULL DEFAULT 1,
                    precip_sensitivity TEXT NOT NULL DEFAULT 'any',
                    severe_alerts_enabled INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_push_subs_cell ON push_subscriptions (cell_lat, cell_lon);"
            )

            # 2. Dispatched NWS alerts ledger (for (device_id, alert_id) deduplication)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dispatched_nws_alerts (
                    device_id TEXT NOT NULL,
                    alert_id TEXT NOT NULL,
                    dispatched_at TEXT NOT NULL,
                    PRIMARY KEY (device_id, alert_id)
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_dispatched_alert_dev ON dispatched_nws_alerts (device_id);"
            )

            # 3. Device precipitation notification history (for 90-min cooldown tracking)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS device_precip_history (
                    device_id TEXT PRIMARY KEY,
                    last_precip_notified_at TEXT NOT NULL,
                    last_precip_event TEXT
                );
                """
            )

            # 4. Cell precipitation state machine persistence
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cell_precip_state (
                    cell_key TEXT PRIMARY KEY,
                    state TEXT NOT NULL DEFAULT 'IDLE',
                    state_updated_at TEXT NOT NULL,
                    rain_stopped_at TEXT
                );
                """
            )
            conn.commit()

    async def register_subscription(self, req: PushSubscriptionRegisterRequest) -> PushSubscriptionResponse:
        """
        Registers or updates a client push subscription.
        Quantizes coordinates to 0.05° spatial cell and overwrites in-place.
        """
        cell_lat, cell_lon = quantize_cell(req.lat, req.lon)
        now_iso = datetime.now(timezone.utc).isoformat()

        def _do_upsert():
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO push_subscriptions (
                        device_id, expo_push_token, lat, lon, cell_lat, cell_lon, timezone,
                        notifications_enabled, dnd_enabled, dnd_start_time, dnd_end_time,
                        dnd_severe_breakthrough, precip_alerts_enabled, precip_sensitivity,
                        severe_alerts_enabled, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(device_id) DO UPDATE SET
                        expo_push_token = excluded.expo_push_token,
                        lat = excluded.lat,
                        lon = excluded.lon,
                        cell_lat = excluded.cell_lat,
                        cell_lon = excluded.cell_lon,
                        timezone = excluded.timezone,
                        notifications_enabled = excluded.notifications_enabled,
                        dnd_enabled = excluded.dnd_enabled,
                        dnd_start_time = excluded.dnd_start_time,
                        dnd_end_time = excluded.dnd_end_time,
                        dnd_severe_breakthrough = excluded.dnd_severe_breakthrough,
                        precip_alerts_enabled = excluded.precip_alerts_enabled,
                        precip_sensitivity = excluded.precip_sensitivity,
                        severe_alerts_enabled = excluded.severe_alerts_enabled,
                        updated_at = excluded.updated_at;
                    """,
                    (
                        req.device_id,
                        req.expo_push_token,
                        req.lat,
                        req.lon,
                        cell_lat,
                        cell_lon,
                        req.timezone,
                        1 if req.notifications_enabled else 0,
                        1 if req.dnd_enabled else 0,
                        req.dnd_start_time,
                        req.dnd_end_time,
                        1 if req.dnd_severe_breakthrough else 0,
                        1 if req.precip_alerts_enabled else 0,
                        req.precip_sensitivity,
                        1 if req.severe_alerts_enabled else 0,
                        now_iso,
                    ),
                )
                conn.commit()

        await asyncio.to_thread(_do_upsert)

        return PushSubscriptionResponse(
            device_id=req.device_id,
            expo_push_token=req.expo_push_token,
            lat=req.lat,
            lon=req.lon,
            cell_lat=cell_lat,
            cell_lon=cell_lon,
            timezone=req.timezone,
            notifications_enabled=req.notifications_enabled,
            dnd_enabled=req.dnd_enabled,
            dnd_start_time=req.dnd_start_time,
            dnd_end_time=req.dnd_end_time,
            dnd_severe_breakthrough=req.dnd_severe_breakthrough,
            precip_alerts_enabled=req.precip_alerts_enabled,
            precip_sensitivity=req.precip_sensitivity,
            severe_alerts_enabled=req.severe_alerts_enabled,
            updated_at=now_iso,
        )

    async def unregister_subscription(self, device_id: str) -> bool:
        """
        Removes a client device subscription.
        """
        def _do_delete():
            with self._get_connection() as conn:
                cur = conn.execute("DELETE FROM push_subscriptions WHERE device_id = ?", (device_id,))
                conn.commit()
                return cur.rowcount > 0

        return await asyncio.to_thread(_do_delete)

    async def get_stats(self) -> NotificationStatsResponse:
        """
        Returns real-time diagnostics and subscription statistics.
        """
        def _do_stats():
            with self._get_connection() as conn:
                row_total = conn.execute("SELECT COUNT(*) AS c FROM push_subscriptions").fetchone()
                total = row_total["c"] if row_total else 0

                row_active = conn.execute(
                    """
                    SELECT COUNT(*) AS c FROM push_subscriptions
                    WHERE notifications_enabled = 1 AND expo_push_token IS NOT NULL AND expo_push_token != ''
                    """
                ).fetchone()
                active = row_active["c"] if row_active else 0

                row_cells = conn.execute(
                    """
                    SELECT COUNT(DISTINCT cell_lat || ',' || cell_lon) AS c
                    FROM push_subscriptions
                    WHERE notifications_enabled = 1
                    """
                ).fetchone()
                unique_cells = row_cells["c"] if row_cells else 0

                # Cell states breakdown
                cell_states: Dict[str, int] = {"IDLE": 0, "APPROACHING": 0, "ACTIVE": 0, "COOLDOWN": 0}
                for r in conn.execute("SELECT state, COUNT(*) as c FROM cell_precip_state GROUP BY state"):
                    st = r["state"]
                    if st in cell_states:
                        cell_states[st] = r["c"]

                return total, active, unique_cells, cell_states

        total, active, unique_cells, cell_states = await asyncio.to_thread(_do_stats)

        return NotificationStatsResponse(
            total_subscriptions=total,
            active_subscriptions=active,
            unique_cells=unique_cells,
            cell_states=cell_states,
            last_run_at=self._last_run_at,
            last_run_duration_ms=self._last_run_duration_ms,
            recent_dispatches=self._recent_dispatches,
            recent_suppressed_dnd=self._recent_suppressed_dnd,
            recent_suppressed_cooldown=self._recent_suppressed_cooldown,
            recent_suppressed_severe_active=self._recent_suppressed_severe_active,
            status="operational",
        )

    async def evaluate_and_dispatch(self) -> Dict[str, Any]:
        """
        Main worker execution cycle:
        1. Query and group all active subscriptions by unique 5km grid cell.
        2. Query in-memory Doppler radar nowcast in O(1) RAM time per cell.
        3. Evaluate anti-bounce hysteresis state machine & 90-min storm cooldown.
        4. Query and cache NWS polygon severe alerts per cell.
        5. Check local DND window & severe breakthrough.
        6. Batch dispatch up to 100 push messages at a time via Expo Push API.
        """
        async with self._lock:
            start_t = time.perf_counter()
            now_utc = datetime.now(timezone.utc)
            self._last_run_at = now_utc.isoformat()

            dispatched_count = 0
            suppressed_dnd = 0
            suppressed_cooldown = 0
            suppressed_severe_active = 0

            # 1. Fetch active subscriptions and group by unique cell
            def _load_subs_and_states():
                with self._get_connection() as conn:
                    subs = conn.execute(
                        """
                        SELECT device_id, expo_push_token, lat, lon, cell_lat, cell_lon,
                               timezone, notifications_enabled, dnd_enabled, dnd_start_time,
                               dnd_end_time, dnd_severe_breakthrough, precip_alerts_enabled,
                               precip_sensitivity, severe_alerts_enabled
                        FROM push_subscriptions
                        WHERE notifications_enabled = 1
                        """
                    ).fetchall()

                    # Load cell states
                    c_states = {}
                    for row in conn.execute("SELECT cell_key, state, state_updated_at, rain_stopped_at FROM cell_precip_state"):
                        c_states[row["cell_key"]] = {
                            "state": row["state"],
                            "state_updated_at": row["state_updated_at"],
                            "rain_stopped_at": row["rain_stopped_at"],
                        }

                    # Load device precip history
                    dev_history = {}
                    for row in conn.execute("SELECT device_id, last_precip_notified_at FROM device_precip_history"):
                        dev_history[row["device_id"]] = row["last_precip_notified_at"]

                    # Load dispatched NWS alerts
                    dispatched_alerts: Set[Tuple[str, str]] = set()
                    for row in conn.execute("SELECT device_id, alert_id FROM dispatched_nws_alerts"):
                        dispatched_alerts.add((row["device_id"], row["alert_id"]))

                    return [dict(s) for s in subs], c_states, dev_history, dispatched_alerts

            raw_subs, cell_states_map, dev_history_map, dispatched_nws_set = await asyncio.to_thread(_load_subs_and_states)

            if not raw_subs:
                self._last_run_duration_ms = round((time.perf_counter() - start_t) * 1000.0, 2)
                return {
                    "evaluated_cells": 0,
                    "dispatched": 0,
                    "duration_ms": self._last_run_duration_ms,
                }

            # Group subscriptions by cell
            cells: Dict[Tuple[float, float], List[Dict[str, Any]]] = {}
            for s in raw_subs:
                cell_k = (float(s["cell_lat"]), float(s["cell_lon"]))
                cells.setdefault(cell_k, []).append(s)

            messages_to_dispatch: List[Dict[str, Any]] = []
            updated_cell_states: Dict[str, Tuple[str, str, Optional[str]]] = {}
            new_device_precip_notifications: Dict[str, str] = {}
            new_dispatched_nws_alerts: List[Tuple[str, str, str]] = []

            # Clean expired cell alert caches (> 5 minutes)
            now_ts = time.time()
            self._cell_alerts_cache = {
                k: v for k, v in self._cell_alerts_cache.items() if (now_ts - v[0]) < self._cell_alert_cache_ttl
            }

            client = http_pool.get_client()

            # 2. Iterate each unique cell (O(1) Doppler evaluation per cell)
            for (cell_lat, cell_lon), sub_list in cells.items():
                cell_key = f"{cell_lat:.4f},{cell_lon:.4f}"
                curr_cell_info = cell_states_map.get(cell_key, {
                    "state": "IDLE",
                    "state_updated_at": now_utc.isoformat(),
                    "rain_stopped_at": None,
                })
                current_cell_state = curr_cell_info["state"]
                rain_stopped_dt: Optional[datetime] = None
                if curr_cell_info.get("rain_stopped_at"):
                    try:
                        rain_stopped_dt = datetime.fromisoformat(curr_cell_info["rain_stopped_at"])
                    except Exception:
                        rain_stopped_dt = None

                # Fetch in-memory Doppler radar telemetry in O(1) RAM time
                try:
                    telemetry = await doppler_service.get_point_nowcast(cell_lat, cell_lon)
                except Exception as e:
                    logger.warning(f"Failed to query doppler telemetry for cell {cell_key}: {e}")
                    continue

                rates_60m = telemetry.minuteByMinuteRates if telemetry.minuteByMinuteRates else [0.0] * 61
                is_raining_now = (telemetry.precipRateMmHr >= 0.25) or (telemetry.currentDbz >= 18.0)

                # Fetch & cache active NWS alerts for this cell
                cell_alerts: List[WeatherAlertModel] = []
                if db_manager.is_us_region(cell_lat, cell_lon):
                    if (cell_lat, cell_lon) in self._cell_alerts_cache:
                        cell_alerts = self._cell_alerts_cache[(cell_lat, cell_lon)][1]
                    else:
                        try:
                            cell_alerts = await weather_service._fetch_nws_polygon_alerts(client, cell_lat, cell_lon)
                            self._cell_alerts_cache[(cell_lat, cell_lon)] = (now_ts, cell_alerts)
                        except Exception as e:
                            logger.warning(f"Failed to fetch NWS alerts for cell {cell_key}: {e}")
                            cell_alerts = []

                # Filter severe safety alerts
                severe_alerts = [
                    a for a in cell_alerts
                    if (a.severity in ["Extreme", "Severe"]) or (
                        a.urgency in ["Immediate", "Expected"] and a.severity not in ["Minor", "Unknown"]
                    )
                ]
                cell_has_active_severe = len(severe_alerts) > 0

                # -------------------------------------------------------------
                # Tier A: Next-Hour Precipitation & Hysteresis State Machine
                # -------------------------------------------------------------
                new_state = current_cell_state
                trigger_precip_onset = False
                onset_minute: Optional[int] = None
                max_upcoming_rate = 0.0

                if is_raining_now:
                    new_state = "ACTIVE"
                    rain_stopped_dt = None
                else:
                    if current_cell_state == "ACTIVE":
                        # Rain just stopped; enter 90-minute cooldown
                        new_state = "COOLDOWN"
                        rain_stopped_dt = now_utc
                    elif current_cell_state == "COOLDOWN":
                        if rain_stopped_dt and (now_utc - rain_stopped_dt).total_seconds() >= COOLDOWN_SECONDS:
                            new_state = "IDLE"
                            rain_stopped_dt = None
                        else:
                            new_state = "COOLDOWN"

                    if new_state == "IDLE":
                        # Scan 1..60 minutes for rain onset
                        for m_idx in range(1, len(rates_60m)):
                            if rates_60m[m_idx] >= 0.25:
                                onset_minute = m_idx
                                max_upcoming_rate = max(rates_60m[m_idx:])
                                break

                        if onset_minute is not None:
                            new_state = "APPROACHING"
                            trigger_precip_onset = True

                    elif new_state == "APPROACHING":
                        # If storm has dissipated completely, revert to IDLE
                        any_upcoming = any(r >= 0.25 for r in rates_60m[1:])
                        if not any_upcoming:
                            new_state = "IDLE"

                # Persist updated cell state
                rain_stopped_str = rain_stopped_dt.isoformat() if rain_stopped_dt else None
                updated_cell_states[cell_key] = (new_state, now_utc.isoformat(), rain_stopped_str)

                # Anti-bounce rule: suppress Tier A rain onset if Tier B severe alert is active in this cell
                if trigger_precip_onset and cell_has_active_severe:
                    trigger_precip_onset = False
                    suppressed_severe_active += len(sub_list)

                # Evaluate subscribers for Tier A onset
                if trigger_precip_onset and onset_minute is not None:
                    # Determine rain intensity label
                    if max_upcoming_rate >= 6.0:
                        intensity_label = "Heavy rain"
                    elif max_upcoming_rate >= 1.5:
                        intensity_label = "Moderate rain"
                    else:
                        intensity_label = "Light rain"

                    if onset_minute <= 2:
                        precip_body = f"{intensity_label} starting in a couple of minutes."
                    else:
                        precip_body = f"{intensity_label} starting in ~{onset_minute} min."

                    for sub in sub_list:
                        if not sub.get("precip_alerts_enabled", 1):
                            continue

                        # Check subscriber sensitivity threshold
                        sens = sub.get("precip_sensitivity", "any")
                        req_rate = PRECIP_SENSITIVITY_THRESHOLDS.get(sens, 0.25)
                        if max_upcoming_rate < req_rate:
                            continue

                        dev_id = sub["device_id"]

                        # Check DND window (Tier A is ALWAYS suppressed during DND)
                        if sub.get("dnd_enabled"):
                            if is_in_dnd_window(
                                now_utc,
                                sub.get("timezone", "UTC"),
                                sub.get("dnd_start_time", "22:00"),
                                sub.get("dnd_end_time", "07:00"),
                            ):
                                suppressed_dnd += 1
                                continue

                        # Check per-device 90-minute storm cooldown
                        last_notified_str = dev_history_map.get(dev_id)
                        if last_notified_str:
                            try:
                                last_notified_dt = datetime.fromisoformat(last_notified_str)
                                if (now_utc - last_notified_dt).total_seconds() < COOLDOWN_SECONDS:
                                    suppressed_cooldown += 1
                                    continue
                            except Exception:
                                pass

                        # Valid Tier A push message
                        token = sub.get("expo_push_token")
                        messages_to_dispatch.append({
                            "to": token,
                            "title": "Precipitation Alert",
                            "body": precip_body,
                            "sound": "default",
                            "priority": "default",
                            "interruptionLevel": "active",
                            "channelId": "weather-alerts",
                            "data": {
                                "type": "precip",
                                "onset_minutes": onset_minute,
                                "max_rate_mmhr": round(max_upcoming_rate, 2),
                                "cell": [cell_lat, cell_lon],
                            },
                            "device_id": dev_id,
                            "is_severe": False,
                        })
                        new_device_precip_notifications[dev_id] = now_utc.isoformat()
                        dev_history_map[dev_id] = now_utc.isoformat()

                # -------------------------------------------------------------
                # Tier B: Critical & Severe Safety (NWS Warnings)
                # -------------------------------------------------------------
                for alert in severe_alerts:
                    for sub in sub_list:
                        if not sub.get("severe_alerts_enabled", 1):
                            continue

                        dev_id = sub["device_id"]
                        dedup_key = (dev_id, alert.id)
                        if dedup_key in dispatched_nws_set:
                            continue

                        in_dnd = False
                        if sub.get("dnd_enabled"):
                            in_dnd = is_in_dnd_window(
                                now_utc,
                                sub.get("timezone", "UTC"),
                                sub.get("dnd_start_time", "22:00"),
                                sub.get("dnd_end_time", "07:00"),
                            )

                        if in_dnd and not sub.get("dnd_severe_breakthrough", 1):
                            suppressed_dnd += 1
                            continue

                        # Determine interruption level
                        interruption_level = "critical" if alert.severity == "Extreme" else "time-sensitive"
                        token = sub.get("expo_push_token")

                        messages_to_dispatch.append({
                            "to": token,
                            "title": f"⚠️ {alert.event}",
                            "body": alert.headline or f"{alert.event} in effect for your area.",
                            "sound": "default",
                            "priority": "high",
                            "interruptionLevel": interruption_level,
                            "channelId": "severe-weather",
                            "data": {
                                "type": "severe",
                                "alert_id": alert.id,
                                "event": alert.event,
                                "severity": alert.severity,
                                "urgency": alert.urgency,
                                "cell": [cell_lat, cell_lon],
                            },
                            "device_id": dev_id,
                            "is_severe": True,
                        })
                        dispatched_nws_set.add(dedup_key)
                        new_dispatched_nws_alerts.append((dev_id, alert.id, now_utc.isoformat()))

            # 3. Persist state updates and ledgers to SQLite
            def _persist_cycle_data():
                with self._get_connection() as conn:
                    # Update cell states
                    for c_k, (st, st_up, r_st) in updated_cell_states.items():
                        conn.execute(
                            """
                            INSERT INTO cell_precip_state (cell_key, state, state_updated_at, rain_stopped_at)
                            VALUES (?, ?, ?, ?)
                            ON CONFLICT(cell_key) DO UPDATE SET
                                state = excluded.state,
                                state_updated_at = excluded.state_updated_at,
                                rain_stopped_at = excluded.rain_stopped_at;
                            """,
                            (c_k, st, st_up, r_st),
                        )

                    # Update device precip notifications
                    for d_id, t_iso in new_device_precip_notifications.items():
                        conn.execute(
                            """
                            INSERT INTO device_precip_history (device_id, last_precip_notified_at)
                            VALUES (?, ?)
                            ON CONFLICT(device_id) DO UPDATE SET
                                last_precip_notified_at = excluded.last_precip_notified_at;
                            """,
                            (d_id, t_iso),
                        )

                    # Insert dispatched NWS alerts
                    for d_id, a_id, t_iso in new_dispatched_nws_alerts:
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO dispatched_nws_alerts (device_id, alert_id, dispatched_at)
                            VALUES (?, ?, ?);
                            """,
                            (d_id, a_id, t_iso),
                        )

                    conn.commit()

            await asyncio.to_thread(_persist_cycle_data)

            # 4. Batch Dispatch to Expo Push API in chunks of up to 100
            real_expo_payloads: List[Dict[str, Any]] = []
            for msg in messages_to_dispatch:
                tok = msg.get("to")
                dispatched_count += 1
                # Check for live token format
                if tok and isinstance(tok, str) and (tok.startswith("ExponentPushToken[") or tok.startswith("ExpoPushToken[")):
                    payload = {
                        "to": tok,
                        "title": msg["title"],
                        "body": msg["body"],
                        "sound": msg["sound"],
                        "priority": msg["priority"],
                        "interruptionLevel": msg["interruptionLevel"],
                        "channelId": msg["channelId"],
                        "data": msg["data"],
                    }
                    real_expo_payloads.append(payload)

            if real_expo_payloads:
                await self._dispatch_expo_batches(client, real_expo_payloads)

            # 5. Record run statistics
            self._recent_dispatches = dispatched_count
            self._recent_suppressed_dnd = suppressed_dnd
            self._recent_suppressed_cooldown = suppressed_cooldown
            self._recent_suppressed_severe_active = suppressed_severe_active
            self._last_run_duration_ms = round((time.perf_counter() - start_t) * 1000.0, 2)

            return {
                "evaluated_cells": len(cells),
                "dispatched": dispatched_count,
                "suppressed_dnd": suppressed_dnd,
                "suppressed_cooldown": suppressed_cooldown,
                "suppressed_severe_active": suppressed_severe_active,
                "duration_ms": self._last_run_duration_ms,
            }

    async def _dispatch_expo_batches(self, client: httpx.AsyncClient, payloads: List[Dict[str, Any]]):
        """
        Dispatches push payloads in batches of up to 100 to Expo Push API.
        """
        for i in range(0, len(payloads), 100):
            chunk = payloads[i : i + 100]
            try:
                resp = await client.post(
                    "https://exp.host/--/api/v2/push/send",
                    json=chunk,
                    headers={
                        "Accept": "application/json",
                        "Accept-Encoding": "gzip, deflate",
                        "Content-Type": "application/json",
                    },
                    timeout=10.0,
                )
                if resp.status_code != 200:
                    logger.warning(f"Expo push batch HTTP {resp.status_code}: {resp.text[:200]}")
            except Exception as e:
                logger.warning(f"Failed to deliver Expo push batch: {e}")


push_worker = PushWorker()
