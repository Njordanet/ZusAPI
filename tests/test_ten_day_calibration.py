from datetime import datetime, timedelta, timezone
import pytest

from app.models.weather import HourlyForecastModel, WeatherCondition
from app.services.db import db_manager
from app.services.weather_provider import (
    blend_hourly_precip_and_snow,
    calc_horizon_bias_scales,
    eval_decayed_diurnal_bias,
    reconcile_daily_condition_and_snow,
)


def test_horizon_bias_decay_schedule():
    """Short-range bias stays at full strength inside 36h and tapers cleanly out to 96h+."""
    assert calc_horizon_bias_scales(0.0) == (1.0, 1.0)
    assert calc_horizon_bias_scales(36.0) == (1.0, 1.0)

    mid_mean, mid_amp = calc_horizon_bias_scales(66.0)
    assert abs(mid_mean - 0.575) < 1e-4
    assert abs(mid_amp - 0.5) < 1e-4

    assert calc_horizon_bias_scales(96.0) == (0.15, 0.0)
    assert calc_horizon_bias_scales(168.0) == (0.15, 0.0)

    b0, bsin, bcos = 2.0, 1.2, -0.8
    assert eval_decayed_diurnal_bias(b0, bsin, bcos, 12, 12.0) == pytest.approx(
        eval_decayed_diurnal_bias(b0, bsin, bcos, 12, 0.0), abs=1e-4
    )
    assert eval_decayed_diurnal_bias(b0, bsin, bcos, 12, 120.0) == pytest.approx(0.3, abs=1e-4)


def test_conus_region_bounds():
    """CONUS bounds include mainland US locations and exclude AK, HI, and Europe."""
    assert db_manager.is_conus_region(42.2569, -83.7705) is True  # Ann Arbor, MI
    assert db_manager.is_conus_region(40.7128, -74.0060) is True  # NYC
    assert db_manager.is_conus_region(61.2181, -149.9003) is False  # Anchorage, AK
    assert db_manager.is_conus_region(21.3069, -157.8583) is False  # Honolulu, HI
    assert db_manager.is_conus_region(51.5074, -0.1278) is False  # London, UK


def test_per_model_snow_prevents_cross_model_artifact():
    """Warm wet model + dry cold model must not synthesize snow."""
    entries = [
        {
            "weight": 0.5,
            "temp_c": 6.0,
            "dew_c": 4.0,
            "rh_pct": 85.0,
            "precip_mm": 15.0,
            "precip_prob": 75,
            "wmo_code": 63,
        },
        {
            "weight": 0.5,
            "temp_c": -4.0,
            "dew_c": -8.0,
            "rh_pct": 70.0,
            "precip_mm": 0.0,
            "precip_prob": 5,
            "wmo_code": 3,
        },
    ]
    h_pt, h_snow_cm = blend_hourly_precip_and_snow(
        model_entries=entries,
        total_w=1.0,
        blended_temp_c=1.0,
        blended_rh=78.0,
        blended_pop=40,
        top_code=63,
        lead_hours=120.0,
    )
    assert h_pt == "rain"
    assert h_snow_cm == 0.0


def test_per_model_snow_preserves_genuine_cold_snow():
    """Genuinely cold, wet model produces positive weighted snow accumulation."""
    entries = [
        {
            "weight": 0.6,
            "temp_c": -3.0,
            "dew_c": -4.5,
            "rh_pct": 90.0,
            "precip_mm": 5.0,
            "precip_prob": 70,
            "wmo_code": 73,
        },
        {
            "weight": 0.4,
            "temp_c": -2.0,
            "dew_c": -3.5,
            "rh_pct": 88.0,
            "precip_mm": 2.5,
            "precip_prob": 55,
            "wmo_code": 71,
        },
    ]
    h_pt, h_snow_cm = blend_hourly_precip_and_snow(
        model_entries=entries,
        total_w=1.0,
        blended_temp_c=-2.6,
        blended_rh=89.0,
        blended_pop=64,
        top_code=73,
        lead_hours=72.0,
    )
    assert h_pt == "snow"
    assert h_snow_cm > 3.0


def _make_day_slice(clouds_by_hour, wmos_by_hour=None, snows_by_hour=None, pops_by_hour=None, ptype="rain"):
    base = datetime(2026, 11, 21, 5, 0, tzinfo=timezone.utc)  # 00:00 EST
    tz_delta = timedelta(hours=-5)
    out = []
    for hr in range(24):
        t = base + timedelta(hours=hr)
        local_hr = (t + tz_delta).hour
        is_day = 7 <= local_hr <= 17
        wmo = (wmos_by_hour[hr] if wmos_by_hour else (3 if clouds_by_hour[hr] >= 80 else (2 if clouds_by_hour[hr] >= 26 else 0)))
        snow_cm = snows_by_hour[hr] if snows_by_hour else 0.0
        pop = pops_by_hour[hr] if pops_by_hour else 15
        out.append(
            HourlyForecastModel(
                time=t,
                temperature=3.0,
                condition=WeatherCondition.from_wmo_code(wmo),
                wmoCode=wmo,
                precipitationChance=pop,
                precipMm=0.5 if wmo >= 51 else 0.0,
                cloudCover=clouds_by_hour[hr],
                isDay=is_day,
                precipType=ptype,
                snowAccumulationCm=snow_cm,
            )
        )
    return out


def test_daily_wet_icon_gated_on_low_pop_and_daylight_clouds_prioritized():
    """Low PoP (<30%) future days demote wet codes to daylight-weighted sky icons."""
    tz_delta = timedelta(hours=-5)
    # 40% clouds during daytime (07..17), 10% clouds overnight -> daylight-weighted ~32% -> Partly Cloudy (2)
    clouds = [40 if (7 <= hr <= 17) else 10 for hr in range(24)]
    wmos = [61 if hr in (3, 4, 15) else (2 if 7 <= hr <= 17 else 0) for hr in range(24)]
    snows = [0.2 if hr in (3, 4) else 0.0 for hr in range(24)]
    day_slice = _make_day_slice(clouds, wmos_by_hour=wmos, snows_by_hour=snows, pops_by_hour=[18] * 24, ptype="snow")

    day_code, d_pt, d_snow, _ = reconcile_daily_condition_and_snow(
        day_slice=day_slice,
        d_idx=4,
        d_pop=18,
        d_psum=1.5,
        d_min=-1.0,
        d_max=6.0,
        initial_day_code=61,
        tz_delta=tz_delta,
    )
    assert day_code == 2  # Partly Cloudy, not Rain or Clear
    assert d_snow == 0.0  # Gated out below 25% PoP
    assert d_pt == "rain"


def test_daily_snow_and_mix_reconciliation_when_confident():
    """Confident (>=30% PoP) wintry days reconcile rain+snow mixes to sleet and snow days to snow."""
    tz_delta = timedelta(hours=-5)
    clouds = [90] * 24
    wmos = [61 if hr in (12, 13, 14) else (71 if hr in (2, 3) else 3) for hr in range(24)]
    snows = [0.4 if hr in (2, 3) else 0.0 for hr in range(24)]
    pops = [45 if hr in (2, 3, 12, 13, 14) else 10 for hr in range(24)]
    day_slice = _make_day_slice(clouds, wmos_by_hour=wmos, snows_by_hour=snows, pops_by_hour=pops)
    for hr in (2, 3):
        day_slice[hr].precipType = "snow"
    for hr in (12, 13, 14):
        day_slice[hr].precipType = "rain"

    day_code, d_pt, d_snow, _ = reconcile_daily_condition_and_snow(
        day_slice=day_slice,
        d_idx=2,
        d_pop=45,
        d_psum=4.0,
        d_min=-1.0,
        d_max=5.0,
        initial_day_code=61,
        tz_delta=tz_delta,
    )
    assert d_snow == 0.8
    assert d_pt == "sleet"
    assert day_code == 66
