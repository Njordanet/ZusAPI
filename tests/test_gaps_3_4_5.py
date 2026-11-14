import pytest
import math
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, patch

from app.models.weather import (
    WeatherCondition,
    WeatherDataModel,
    HourlyForecastModel,
    DailyForecastModel,
    WeatherResponse,
    RadarLocalTelemetry,
    NowcastSummary,
    NowcastPoint,
    WeatherData,
    HourlyForecast,
    DailyForecast,
)
from app.services.weather_provider import (
    calc_wet_bulb_c,
    calc_snow_liquid_ratio,
    classify_precip_type,
    calc_dobson_units,
    weather_service,
)
from app.services.station_service import parse_metar_wx
from app.services.doppler_service import DopplerRadarService, DopplerGridFrame
from app.api.v1.endpoints.standard import (
    _darksky_icon,
    _build_darksky_payload,
)
from fastapi.testclient import TestClient
from app.main import app


# ==============================================================================
# GAP 3 TESTS: Doppler Curved Trajectory & Telemetry
# ==============================================================================

@pytest.mark.asyncio
async def test_gap3_point_telemetry_curved_fields():
    """Verify that get_point_telemetry samples cum_drow/dcol/hrrr_trend fields and sets tangent motion."""
    svc = DopplerRadarService()
    # Create fake frames
    now = datetime.now(timezone.utc)
    dummy_grid = np.full((svc.GRID_H, svc.GRID_W), 100, dtype=np.uint8)  # ~17.5 dBZ echo
    live_frame = DopplerGridFrame(
        frame_id="live_0",
        dt=now,
        label="Live",
        offset_minutes=0,
        is_future=False,
        grid=dummy_grid,
        raw_bytes=b"",
    )
    ref_frame = DopplerGridFrame(
        frame_id="ref_0",
        dt=now - timedelta(minutes=15),
        label="-15m",
        offset_minutes=-15,
        is_future=False,
        grid=dummy_grid,
        raw_bytes=b"",
    )
    svc._frames = {"ref": ref_frame, "live": live_frame}

    # Populate curved fields: simulated storm moving northeast (drow < 0, dcol > 0)
    th, tw = 110, 245
    for s in range(1, 9):
        svc._cum_drow_fields[s] = np.full((th, tw), -4.0 * s, dtype=np.float32)
        svc._cum_dcol_fields[s] = np.full((th, tw), 6.0 * s, dtype=np.float32)
        svc._hrrr_trend_fields[s] = np.full((th, tw), 2.0, dtype=np.float32)  # +1.0 dBZ trend

    # Coordinate in CONUS (e.g. NYC ~ 40.71, -74.0)
    telem = await svc.get_point_telemetry(lat=40.7128, lon=-74.0060)

    assert telem.inConusGrid is True
    assert len(telem.minuteByMinuteRates) == 61
    assert len(telem.minuteRates60m) == 13
    assert telem.stormMotionKmh > 0
    # Tangent motion bearing: vx > 0 (east), vy > 0 (north) -> motion towards NE (~50-60 deg)
    assert 30 <= telem.stormMotionDegrees <= 70
    assert telem.stormMotionCompass in ("NE", "ENE", "NNE")
    # All minute rates should be positive since dummy grid has 90 (> 7.5 dBZ)
    assert all(r >= 0 for r in telem.minuteByMinuteRates)
    assert telem.currentDbz > 0


@pytest.mark.asyncio
async def test_gap3_point_telemetry_fallback_clean():
    """Verify clean fallback to _estimate_regional_motion_15m when _cum_drow_fields is empty."""
    svc = DopplerRadarService()
    now = datetime.now(timezone.utc)
    dummy_grid = np.zeros((svc.GRID_H, svc.GRID_W), dtype=np.uint8)
    svc._frames = {
        "live": DopplerGridFrame(
            frame_id="live_0", dt=now, label="Live", offset_minutes=0, is_future=False, grid=dummy_grid, raw_bytes=b""
        )
    }
    svc._cum_drow_fields = {}
    svc._cum_dcol_fields = {}
    svc._hrrr_trend_fields = {}

    telem = await svc.get_point_telemetry(lat=40.7128, lon=-74.0060)
    assert telem.inConusGrid is True
    assert len(telem.minuteByMinuteRates) == 61
    assert telem.precipRateMmHr == 0.0
    assert telem.intensityCategory == "Clear"


# ==============================================================================
# GAP 4 TESTS: WeatherCondition, Stull Wet-Bulb, Precip Phase, Snow SLR
# ==============================================================================

def test_gap4_wmo_code_backwards_compatibility():
    """WMO codes 56, 57, 66, 67, 77 map to WeatherCondition.SNOW while strictly respecting the 11 enum strings."""
    assert WeatherCondition.from_wmo_code(56) == WeatherCondition.SNOW
    assert WeatherCondition.from_wmo_code(57) == WeatherCondition.SNOW
    assert WeatherCondition.from_wmo_code(66) == WeatherCondition.SNOW
    assert WeatherCondition.from_wmo_code(67) == WeatherCondition.SNOW
    assert WeatherCondition.from_wmo_code(77) == WeatherCondition.SNOW

    # Verify existing mappings unchanged
    assert WeatherCondition.from_wmo_code(0) == WeatherCondition.CLEAR
    assert WeatherCondition.from_wmo_code(1) == WeatherCondition.PARTLY_CLOUDY
    assert WeatherCondition.from_wmo_code(3) == WeatherCondition.OVERCAST
    assert WeatherCondition.from_wmo_code(61) == WeatherCondition.RAIN
    assert WeatherCondition.from_wmo_code(65) == WeatherCondition.HEAVY_RAIN
    assert WeatherCondition.from_wmo_code(95) == WeatherCondition.THUNDERSTORM

    # Verify only 11 enum strings exist
    valid_conditions = {
        "Clear", "Partly Cloudy", "Overcast", "Foggy", "Drizzle",
        "Rain", "Heavy Rain", "Snow", "Thunderstorm", "Severe Thunderstorm", "Unknown"
    }
    assert set(c.value for c in WeatherCondition) == valid_conditions


def test_gap4_wet_bulb_stull_formula():
    """Stull wet-bulb formula calculation."""
    # At 20°C and 50% RH, wet-bulb should be ~13.7°C
    tw = calc_wet_bulb_c(20.0, 50.0)
    assert 13.0 <= tw <= 14.5

    # At 0°C and 100% RH, wet-bulb should be ~0.0°C
    tw_zero = calc_wet_bulb_c(0.0, 100.0)
    assert -0.5 <= tw_zero <= 0.5

    # At -5°C and 80% RH, wet-bulb should be negative
    tw_sub = calc_wet_bulb_c(-5.0, 80.0)
    assert tw_sub < -5.0


def test_gap4_classify_precip_type():
    """Classifies precipType into 'rain', 'snow', or 'sleet'."""
    # 1. WMO codes
    assert classify_precip_type(temp_c=2.0, wmo_code=56) == "sleet"
    assert classify_precip_type(temp_c=2.0, wmo_code=66) == "sleet"
    assert classify_precip_type(temp_c=2.0, wmo_code=77) == "sleet"

    # 2. METAR tokens
    assert classify_precip_type(temp_c=2.0, metar_wx="-FZRA") == "sleet"
    assert classify_precip_type(temp_c=2.0, metar_wx="PL") == "sleet"
    assert classify_precip_type(temp_c=2.0, metar_wx="+FZDZ") == "sleet"
    assert classify_precip_type(temp_c=2.0, metar_wx="IC") == "sleet"
    assert classify_precip_type(temp_c=-2.0, metar_wx="-SN") == "snow"

    # 3. Thermodynamic boundaries via wet-bulb
    # Tw <= -0.4°C -> snow
    assert classify_precip_type(temp_c=-2.0, rh_pct=90.0) == "snow"
    # -0.4°C < Tw <= +0.8°C -> sleet
    assert classify_precip_type(temp_c=1.0, rh_pct=70.0, wet_bulb_c=0.2) == "sleet"
    # Tw > +0.8°C -> rain
    assert classify_precip_type(temp_c=8.0, rh_pct=80.0) == "rain"


def test_gap4_snow_liquid_ratio():
    """Kuchera/Cobb SLR calculations."""
    assert calc_snow_liquid_ratio(temp_c=0.0, precip_type="snow") == 10.0
    assert calc_snow_liquid_ratio(temp_c=-15.0, precip_type="snow") == 18.0
    assert calc_snow_liquid_ratio(temp_c=-7.5, precip_type="snow") == 14.0
    assert calc_snow_liquid_ratio(temp_c=1.0, precip_type="sleet") == 3.0
    assert calc_snow_liquid_ratio(temp_c=10.0, precip_type="rain") == 0.0


def test_gap4_models_and_aliases_backward_compatible():
    """Verify new optional fields on WeatherDataModel, HourlyForecastModel, DailyForecastModel and aliases."""
    # Instantiating with bare minimum fields (simulating old decoders / minimal payloads)
    cur = WeatherDataModel(
        cityName="Testville",
        temperature=15.0,
        feelsLike=15.0,
        condition=WeatherCondition.CLEAR,
        tempMin=10.0,
        tempMax=20.0,
        humidity=50,
        windSpeed=10.0,
        windDirection=180,
        uvIndex=3.0,
        airPressure=1013.25,
    )
    assert cur.wetBulbTemperature is None
    assert cur.precipType is None
    assert cur.ozone is None

    hourly = HourlyForecastModel(
        time=datetime.now(timezone.utc),
        temperature=15.0,
        condition=WeatherCondition.CLEAR,
        precipitationChance=0,
    )
    assert hourly.wetBulbTemperature is None
    assert hourly.precipType is None
    assert hourly.snowAccumulationCm is None

    daily = DailyForecastModel(
        date=datetime.now(timezone.utc),
        condition=WeatherCondition.CLEAR,
        minTemperature=10.0,
        maxTemperature=20.0,
        precipitationChance=0,
    )
    assert daily.precipType is None
    assert daily.snowAccumulationCm is None
    assert daily.snowToLiquidRatio is None

    # Verify type aliases
    assert WeatherData is WeatherDataModel
    assert HourlyForecast is HourlyForecastModel
    assert DailyForecast is DailyForecastModel


def test_gap4_metar_freezing_tokens():
    """METAR parser properly maps freezing rain/drizzle to WeatherCondition.SNOW."""
    cond, wmo, mmhr = parse_metar_wx("-FZRA")
    assert cond == WeatherCondition.SNOW
    assert wmo == 66
    assert mmhr > 0

    cond, wmo, mmhr = parse_metar_wx("+FZDZ")
    assert cond == WeatherCondition.SNOW
    assert wmo == 57


# ==============================================================================
# GAP 5 TESTS: Dobson Ozone, Minutely Probability, Extend=Hourly, Dark Sky Interop
# ==============================================================================

def test_gap5_dobson_units():
    """Ozone calculation blends latitude/season baseline in realistic 260-360 DU range."""
    # Mid-latitudes (NYC ~40.7)
    du_nyc = calc_dobson_units(40.7128, surface_ozone_ugm3=60.0)
    assert 260.0 <= du_nyc <= 360.0

    # Tropics (~10.0)
    du_tropics = calc_dobson_units(10.0, surface_ozone_ugm3=None)
    assert 260.0 <= du_tropics <= 350.0

    # High latitudes (~65.0)
    du_arctic = calc_dobson_units(65.0, surface_ozone_ugm3=80.0)
    assert 260.0 <= du_arctic <= 400.0


def test_gap5_darksky_icon_sleet():
    """_darksky_icon emits 'sleet' when precip_type is 'sleet'."""
    assert _darksky_icon(WeatherCondition.RAIN, precip_type="sleet") == "sleet"
    assert _darksky_icon(WeatherCondition.SNOW, precip_type="sleet") == "sleet"
    assert _darksky_icon(WeatherCondition.SNOW) == "snow"
    assert _darksky_icon(WeatherCondition.RAIN) == "rain"


@pytest.mark.asyncio
async def test_gap5_darksky_payload_features():
    """Verify _build_darksky_payload with extend=hourly, argmax daily times, and minutely prob calibration."""
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    # Create 72 hours of dummy hourly forecasts
    dummy_hourly: list[HourlyForecastModel] = []
    for h in range(72):
        t = now + timedelta(hours=h)
        temp = 20.0 + 5.0 * math.sin(h * math.pi / 12.0)
        feels = temp + 1.0
        dummy_hourly.append(
            HourlyForecastModel(
                time=t,
                temperature=temp,
                feelsLike=feels,
                condition=WeatherCondition.PARTLY_CLOUDY,
                precipitationChance=30 if h == 10 else 0,
                precipMm=2.5 if h == 10 else 0.0,
                precipMmMin=1.5 if h == 10 else 0.0,
                precipMmMax=3.5 if h == 10 else 0.0,
                uvIndex=7.0 if (h % 24) == 13 else 1.0,
                precipType="rain" if temp > 1.0 else "snow",
                snowAccumulationCm=0.0,
            )
        )

    # 3 dummy daily forecasts
    dummy_daily: list[DailyForecastModel] = []
    for d in range(3):
        dt = now + timedelta(days=d)
        dummy_daily.append(
            DailyForecastModel(
                date=dt,
                condition=WeatherCondition.PARTLY_CLOUDY,
                minTemperature=15.0,
                maxTemperature=25.0,
                precipitationChance=30 if d == 0 else 0,
                precipSumMm=2.5 if d == 0 else 0.0,
                precipType="rain",
            )
        )

    cur_model = WeatherDataModel(
        cityName="Test City",
        temperature=21.0,
        feelsLike=22.0,
        condition=WeatherCondition.PARTLY_CLOUDY,
        tempMin=15.0,
        tempMax=25.0,
        humidity=55,
        windSpeed=12.0,
        windDirection=210,
        uvIndex=4.0,
        airPressure=1014.0,
        precipType="rain",
        ozone=325.5,
    )

    mock_resp = WeatherResponse(
        current=cur_model,
        hourly=dummy_hourly,
        daily=dummy_daily,
        dopplerTelemetry=RadarLocalTelemetry(
            inConusGrid=True,
            currentDbz=32.0,
            precipRateMmHr=1.8,
            minuteByMinuteRates=[1.8] * 61,
        ),
        timezone="America/New_York",
        utcOffsetSeconds=-18000,
    )

    with patch.object(weather_service, "get_forecast", new=AsyncMock(return_value=mock_resp)):
        # 1. Default extend (should slice hourly.data to 48)
        p_default = await _build_darksky_payload(lat=40.7128, lon=-74.0060, extend=None)
        assert len(p_default["hourly"]["data"]) == 48

        # 2. extend="hourly" (should return all available up to 168, here 72)
        p_extended = await _build_darksky_payload(lat=40.7128, lon=-74.0060, extend="hourly")
        assert len(p_extended["hourly"]["data"]) == 72

        # 3. Ozone in currently uses cur.ozone
        assert p_default["currently"]["ozone"] == 325.5

        # 4. Minutely calibrated probability for mmhr >= 0.1
        minutely_pts = p_default["minutely"]["data"]
        assert len(minutely_pts) == 61
        assert minutely_pts[0]["precipProbability"] >= 0.80
        assert "precipIntensityError" in minutely_pts[0]
        assert minutely_pts[0]["precipType"] == "rain"

        # 5. Daily argmax/argmin timestamps and fields
        d0 = p_default["daily"]["data"][0]
        assert "temperatureHighTime" in d0
        assert "temperatureLowTime" in d0
        assert "apparentTemperatureHigh" in d0
        assert "apparentTemperatureHighTime" in d0
        assert "apparentTemperatureLow" in d0
        assert "apparentTemperatureLowTime" in d0
        assert "precipIntensityMaxTime" in d0
        assert "uvIndexTime" in d0
        assert d0["ozone"] == 325.5
        # Since precipType is "rain", precipAccumulation should NOT be emitted
        assert "precipAccumulation" not in d0

        # Now test with snow daily to verify precipAccumulation is emitted
        dummy_daily[0].precipType = "snow"
        dummy_daily[0].snowAccumulationCm = 5.0
        p_snow = await _build_darksky_payload(lat=40.7128, lon=-74.0060)
        d0_snow = p_snow["daily"]["data"][0]
        assert d0_snow["precipType"] == "snow"
        assert "precipAccumulation" in d0_snow
        # 5.0 cm / 2.54 = ~1.97 inches
        assert 1.9 <= d0_snow["precipAccumulation"] <= 2.1


def test_gap5_endpoints_client():
    """Verify HTTP route access for /forecast/{key}/{coords}, /api/v1/darksky/{coords}, and /api/v1/standard/darksky."""
    client = TestClient(app)

    # Mock weather_service.get_forecast to avoid live network
    cur = WeatherDataModel(
        cityName="Standardville",
        temperature=20.0,
        feelsLike=20.0,
        condition=WeatherCondition.CLEAR,
        tempMin=15.0,
        tempMax=25.0,
        humidity=50,
        windSpeed=10.0,
        windDirection=180,
        uvIndex=5.0,
        airPressure=1013.25,
        ozone=318.0,
    )
    mock_resp = WeatherResponse(
        current=cur,
        hourly=[
            HourlyForecastModel(
                time=datetime.now(timezone.utc) + timedelta(hours=i),
                temperature=20.0,
                condition=WeatherCondition.CLEAR,
                precipitationChance=0,
            )
            for i in range(60)
        ],
        daily=[
            DailyForecastModel(
                date=datetime.now(timezone.utc) + timedelta(days=i),
                condition=WeatherCondition.CLEAR,
                minTemperature=15.0,
                maxTemperature=25.0,
                precipitationChance=0,
            )
            for i in range(5)
        ],
        timezone="America/New_York",
        utcOffsetSeconds=-18000,
    )

    with patch.object(weather_service, "get_forecast", new=AsyncMock(return_value=mock_resp)):
        # 1. /forecast/{api_key}/{coords}
        r1 = client.get("/forecast/secret_key/40.7128,-74.0060?extend=hourly")
        assert r1.status_code == 200
        data1 = r1.json()
        assert len(data1["hourly"]["data"]) == 60  # all available up to 168
        assert data1["currently"]["ozone"] == 318.0

        # 2. /api/v1/darksky/{coords}
        r2 = client.get("/api/v1/darksky/40.7128,-74.0060")
        assert r2.status_code == 200
        data2 = r2.json()
        assert len(data2["hourly"]["data"]) == 48  # default sliced to 48

        # 3. /api/v1/standard/darksky/{coords}
        r3 = client.get("/api/v1/standard/darksky/40.7128,-74.0060?extend=hourly")
        assert r3.status_code == 200
        data3 = r3.json()
        assert len(data3["hourly"]["data"]) == 60

        # 4. /api/v1/standard/darksky (Query Params)
        r4 = client.get("/api/v1/standard/darksky?lat=40.7128&lon=-74.0060")
        assert r4.status_code == 200
        data4 = r4.json()
        assert len(data4["hourly"]["data"]) == 48
