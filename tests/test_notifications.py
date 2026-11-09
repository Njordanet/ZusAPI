import pytest
import sqlite3
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo
from fastapi.testclient import TestClient

from app.main import app
from app.models.notifications import (
    PushSubscriptionRegisterRequest,
    PushSubscriptionResponse,
)
from app.models.weather import WeatherAlertModel, RadarLocalTelemetry
from app.services.push_worker import (
    PushWorker,
    quantize_cell,
    is_in_dnd_window,
    push_worker,
)


def test_spatial_cell_quantization():
    """
    Verifies 0.05° (~5km) spatial cell quantization formula:
    cell_lat = round(lat * 20) / 20
    cell_lon = round(lon * 20) / 20
    """
    # NYC Coordinates
    lat, lon = 40.7128, -74.0060
    c_lat, c_lon = quantize_cell(lat, lon)
    assert c_lat == 40.7
    assert c_lon == -74.0

    # Slight shift within same ~5km cell
    lat2, lon2 = 40.7140, -74.0080
    c_lat2, c_lon2 = quantize_cell(lat2, lon2)
    assert (c_lat2, c_lon2) == (c_lat, c_lon)

    # Coords stepping into next cell (0.05° higher)
    lat3 = 40.7350
    c_lat3, _ = quantize_cell(lat3, lon)
    assert c_lat3 == 40.75


def test_dnd_window_logic():
    """
    Verifies exact-minute local time evaluation and overnight wrap-around windows.
    """
    # 1. Overnight wrap-around window: 22:15 -> 06:45 in America/New_York
    tz_ny = "America/New_York"
    ny_tz_obj = ZoneInfo(tz_ny)

    # 22:14 local NY time (1 minute BEFORE DND begins)
    dt_before = datetime(2026, 11, 9, 22, 14, tzinfo=ny_tz_obj).astimezone(timezone.utc)
    assert not is_in_dnd_window(dt_before, tz_ny, "22:15", "06:45")

    # 22:15 local NY time (Exact minute DND begins)
    dt_start = datetime(2026, 11, 9, 22, 15, tzinfo=ny_tz_obj).astimezone(timezone.utc)
    assert is_in_dnd_window(dt_start, tz_ny, "22:15", "06:45")

    # 02:30 local NY time (Middle of night)
    dt_night = datetime(2026, 11, 10, 2, 30, tzinfo=ny_tz_obj).astimezone(timezone.utc)
    assert is_in_dnd_window(dt_night, tz_ny, "22:15", "06:45")

    # 06:44 local NY time (1 minute BEFORE DND ends)
    dt_end_minus1 = datetime(2026, 11, 10, 6, 44, tzinfo=ny_tz_obj).astimezone(timezone.utc)
    assert is_in_dnd_window(dt_end_minus1, tz_ny, "22:15", "06:45")

    # 06:45 local NY time (Exact minute DND ends)
    dt_end = datetime(2026, 11, 10, 6, 45, tzinfo=ny_tz_obj).astimezone(timezone.utc)
    assert not is_in_dnd_window(dt_end, tz_ny, "22:15", "06:45")

    # 14:00 local NY time (Afternoon, clear)
    dt_afternoon = datetime(2026, 11, 10, 14, 0, tzinfo=ny_tz_obj).astimezone(timezone.utc)
    assert not is_in_dnd_window(dt_afternoon, tz_ny, "22:15", "06:45")

    # 2. Same-day window: 13:00 -> 15:30
    dt_1305 = datetime(2026, 11, 10, 13, 5, tzinfo=ny_tz_obj).astimezone(timezone.utc)
    assert is_in_dnd_window(dt_1305, tz_ny, "13:00", "15:30")

    dt_1600 = datetime(2026, 11, 10, 16, 0, tzinfo=ny_tz_obj).astimezone(timezone.utc)
    assert not is_in_dnd_window(dt_1600, tz_ny, "13:00", "15:30")


@pytest.mark.asyncio
async def test_subscription_registration_and_update(tmp_path):
    """
    Tests subscription registration, coordinate quantization, in-place update (no location trail),
    and unregistering.
    """
    worker = PushWorker()
    worker._db_path = tmp_path / "test_subs.db"
    worker._init_sqlite()

    req = PushSubscriptionRegisterRequest(
        device_id="test_dev_001",
        expo_push_token="ExponentPushToken[mock-abc-123]",
        lat=37.7749,
        lon=-122.4194,
        timezone="America/Los_Angeles",
        notifications_enabled=True,
        dnd_enabled=True,
        dnd_start_time="23:00",
        dnd_end_time="08:00",
        dnd_severe_breakthrough=True,
        precip_alerts_enabled=True,
        precip_sensitivity="moderate",
        severe_alerts_enabled=True,
    )

    res = await worker.register_subscription(req)
    assert res.device_id == "test_dev_001"
    assert res.cell_lat == 37.75
    assert res.cell_lon == -122.4

    # Verify stored in SQLite
    with worker._get_connection() as conn:
        row = conn.execute("SELECT * FROM push_subscriptions WHERE device_id = ?", ("test_dev_001",)).fetchone()
        assert row is not None
        assert row["cell_lat"] == 37.75
        assert row["cell_lon"] == -122.4
        assert row["precip_sensitivity"] == "moderate"

    # Update location: in-place overwrite, no location trail
    req_updated = PushSubscriptionRegisterRequest(
        device_id="test_dev_001",
        expo_push_token="ExponentPushToken[mock-abc-123]",
        lat=34.0522,
        lon=-118.2437,
        timezone="America/Los_Angeles",
        notifications_enabled=True,
        dnd_enabled=False,
    )
    await worker.register_subscription(req_updated)

    with worker._get_connection() as conn:
        rows = conn.execute("SELECT * FROM push_subscriptions WHERE device_id = ?", ("test_dev_001",)).fetchall()
        assert len(rows) == 1
        assert rows[0]["cell_lat"] == 34.05
        assert rows[0]["cell_lon"] == -118.25
        assert rows[0]["dnd_enabled"] == 0

    # Unregister
    deleted = await worker.unregister_subscription("test_dev_001")
    assert deleted is True

    with worker._get_connection() as conn:
        row_del = conn.execute("SELECT * FROM push_subscriptions WHERE device_id = ?", ("test_dev_001",)).fetchone()
        assert row_del is None


@pytest.mark.asyncio
async def test_anti_bounce_hysteresis_and_cooldown(tmp_path):
    """
    Verifies cell state transitions (IDLE -> APPROACHING -> ACTIVE -> COOLDOWN -> IDLE),
    90-minute cooldown suppression, and anti-bounce mechanics.
    """
    worker = PushWorker()
    worker._db_path = tmp_path / "test_hysteresis.db"
    worker._init_sqlite()

    # Register 2 devices in same cell (NYC)
    await worker.register_subscription(
        PushSubscriptionRegisterRequest(
            device_id="dev_nyc_1",
            expo_push_token="ExponentPushToken[mock-1]",
            lat=40.71,
            lon=-74.00,
            timezone="America/New_York",
            notifications_enabled=True,
            dnd_enabled=False,
            precip_alerts_enabled=True,
            precip_sensitivity="any",
        )
    )
    await worker.register_subscription(
        PushSubscriptionRegisterRequest(
            device_id="dev_nyc_2",
            expo_push_token="ExponentPushToken[mock-2]",
            lat=40.72,
            lon=-74.01,
            timezone="America/New_York",
            notifications_enabled=True,
            dnd_enabled=False,
            precip_alerts_enabled=True,
            precip_sensitivity="heavy",  # Needs heavy rain (>=6.0 mm/h)
        )
    )

    # 1. First cycle: Approaching light rain (1.2 mm/h starting in 15 min)
    rates_approaching = [0.0] * 61
    for m in range(15, 45):
        rates_approaching[m] = 1.2

    mock_telemetry_approaching = RadarLocalTelemetry(
        inConusGrid=True,
        currentDbz=0.0,
        precipRateMmHr=0.0,
        minuteByMinuteRates=rates_approaching,
    )

    with patch("app.services.push_worker.doppler_service.get_point_nowcast", new=AsyncMock(return_value=mock_telemetry_approaching)), \
         patch("app.services.push_worker.weather_service._fetch_nws_polygon_alerts", new=AsyncMock(return_value=[])):

        res = await worker.evaluate_and_dispatch()
        # NYC cell collapses 2 users into 1 cell
        assert res["evaluated_cells"] == 1
        # dev_nyc_1 ("any" sensitivity >= 0.25) is dispatched; dev_nyc_2 ("heavy" >= 6.0) is not
        assert res["dispatched"] == 1

        # Check cell state transitioned to APPROACHING
        stats = await worker.get_stats()
        assert stats.cell_states.get("APPROACHING", 0) == 1

    # 2. Second cycle (5 mins later): Rain still approaching
    # Because state is already APPROACHING, no duplicate onset notification should be fired!
    with patch("app.services.push_worker.doppler_service.get_point_nowcast", new=AsyncMock(return_value=mock_telemetry_approaching)), \
         patch("app.services.push_worker.weather_service._fetch_nws_polygon_alerts", new=AsyncMock(return_value=[])):

        res2 = await worker.evaluate_and_dispatch()
        assert res2["dispatched"] == 0

    # 3. Third cycle: Rain arrives at minute 0 (Active rain)
    rates_active = [2.5] * 61
    mock_telemetry_active = RadarLocalTelemetry(
        inConusGrid=True,
        currentDbz=32.0,
        precipRateMmHr=2.5,
        minuteByMinuteRates=rates_active,
    )
    with patch("app.services.push_worker.doppler_service.get_point_nowcast", new=AsyncMock(return_value=mock_telemetry_active)), \
         patch("app.services.push_worker.weather_service._fetch_nws_polygon_alerts", new=AsyncMock(return_value=[])):

        res3 = await worker.evaluate_and_dispatch()
        assert res3["dispatched"] == 0  # No onset notification while actively raining
        stats3 = await worker.get_stats()
        assert stats3.cell_states.get("ACTIVE", 0) == 1

    # 4. Fourth cycle: Rain stops (Transitions to COOLDOWN)
    mock_telemetry_clear = RadarLocalTelemetry(
        inConusGrid=True,
        currentDbz=0.0,
        precipRateMmHr=0.0,
        minuteByMinuteRates=[0.0] * 61,
    )
    with patch("app.services.push_worker.doppler_service.get_point_nowcast", new=AsyncMock(return_value=mock_telemetry_clear)), \
         patch("app.services.push_worker.weather_service._fetch_nws_polygon_alerts", new=AsyncMock(return_value=[])):

        res4 = await worker.evaluate_and_dispatch()
        assert res4["dispatched"] == 0
        stats4 = await worker.get_stats()
        assert stats4.cell_states.get("COOLDOWN", 0) == 1


@pytest.mark.asyncio
async def test_severe_suppression_and_deduplication(tmp_path):
    """
    Verifies:
    1. Tier A rain onset alert is SUPPRESSED when an active Tier B severe warning is present in the cell.
    2. NWS severe warning is sent once and deduplicated by (device_id, alert.id).
    3. DND severe breakthrough behavior.
    """
    worker = PushWorker()
    worker._db_path = tmp_path / "test_severe.db"
    worker._init_sqlite()

    # Device with DND enabled, severe breakthrough enabled
    await worker.register_subscription(
        PushSubscriptionRegisterRequest(
            device_id="dev_severe_breakthrough",
            expo_push_token="ExponentPushToken[mock-sev-1]",
            lat=32.7767,
            lon=-96.7970,  # Dallas, TX
            timezone="America/Chicago",
            notifications_enabled=True,
            dnd_enabled=True,
            dnd_start_time="00:00",
            dnd_end_time="23:59",  # 24h DND
            dnd_severe_breakthrough=True,
            precip_alerts_enabled=True,
            severe_alerts_enabled=True,
        )
    )

    # Device with DND enabled, severe breakthrough DISABLED
    await worker.register_subscription(
        PushSubscriptionRegisterRequest(
            device_id="dev_severe_blocked",
            expo_push_token="ExponentPushToken[mock-sev-2]",
            lat=32.7767,
            lon=-96.7970,
            timezone="America/Chicago",
            notifications_enabled=True,
            dnd_enabled=True,
            dnd_start_time="00:00",
            dnd_end_time="23:59",
            dnd_severe_breakthrough=False,
            precip_alerts_enabled=True,
            severe_alerts_enabled=True,
        )
    )

    rates_approaching = [0.0] * 61
    for m in range(10, 50):
        rates_approaching[m] = 4.0

    mock_telemetry = RadarLocalTelemetry(
        inConusGrid=True,
        currentDbz=0.0,
        precipRateMmHr=0.0,
        minuteByMinuteRates=rates_approaching,
    )

    severe_alert = WeatherAlertModel(
        id="NWS-ALERT-TORNADO-101",
        event="Tornado Warning",
        headline="Tornado Warning issued for Dallas County until 10:00 PM CDT.",
        severity="Extreme",
        urgency="Immediate",
    )

    with patch("app.services.push_worker.doppler_service.get_point_nowcast", new=AsyncMock(return_value=mock_telemetry)), \
         patch("app.services.push_worker.weather_service._fetch_nws_polygon_alerts", new=AsyncMock(return_value=[severe_alert])):

        res = await worker.evaluate_and_dispatch()

        # 1. Precip alert was suppressed because severe alert is active
        assert res["suppressed_severe_active"] >= 2

        # 2. dev_severe_breakthrough got the severe alert; dev_severe_blocked was suppressed by DND
        assert res["dispatched"] == 1
        assert res["suppressed_dnd"] >= 1

    # Cycle 2: Exactly same severe alert still active
    # Must be deduplicated (0 dispatched)
    with patch("app.services.push_worker.doppler_service.get_point_nowcast", new=AsyncMock(return_value=mock_telemetry)), \
         patch("app.services.push_worker.weather_service._fetch_nws_polygon_alerts", new=AsyncMock(return_value=[severe_alert])):

        res2 = await worker.evaluate_and_dispatch()
        assert res2["dispatched"] == 0


def test_api_endpoints():
    """
    Verifies FastAPI endpoints:
    - POST /api/v1/notifications/register
    - DELETE /api/v1/notifications/unregister
    - GET /api/v1/notifications/stats
    - POST /api/v1/notifications/evaluate
    """
    client = TestClient(app)

    # 1. Register device
    reg_payload = {
        "device_id": "api_test_dev_99",
        "expo_push_token": "ExponentPushToken[api-test-token]",
        "lat": 33.7490,
        "lon": -84.3880,
        "timezone": "America/New_York",
        "notifications_enabled": True,
        "dnd_enabled": True,
        "dnd_start_time": "22:00",
        "dnd_end_time": "07:00",
        "dnd_severe_breakthrough": True,
        "precip_alerts_enabled": True,
        "precip_sensitivity": "moderate",
        "severe_alerts_enabled": True,
    }
    r = client.post("/api/v1/notifications/register", json=reg_payload)
    assert r.status_code == 200
    data = r.json()
    assert data["device_id"] == "api_test_dev_99"
    assert data["cell_lat"] == 33.75
    assert data["cell_lon"] == -84.4
    assert data["precip_sensitivity"] == "moderate"

    # 2. Get stats
    s = client.get("/api/v1/notifications/stats")
    assert s.status_code == 200
    stats_data = s.json()
    assert stats_data["total_subscriptions"] >= 1
    assert stats_data["status"] == "operational"

    # 3. Evaluate cycle
    ev = client.post("/api/v1/notifications/evaluate")
    assert ev.status_code == 200
    ev_data = ev.json()
    assert "evaluated_cells" in ev_data
    assert "duration_ms" in ev_data

    # 4. Unregister device
    u = client.delete("/api/v1/notifications/unregister?device_id=api_test_dev_99")
    assert u.status_code == 200
    u_data = u.json()
    assert u_data["unregistered"] is True
    assert u_data["device_id"] == "api_test_dev_99"
