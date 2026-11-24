import math
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.v1.endpoints.standard import _build_darksky_payload, _is_daytime_ts
from app.api.v1.endpoints.weather import _apply_weather_transforms
from app.config import settings
from app.main import app
from app.models.notifications import PushSubscriptionRegisterRequest
from app.models.weather import (
    DailyForecastModel,
    HourlyForecastModel,
    RadarLocalTelemetry,
    WeatherCondition,
    WeatherDataModel,
    WeatherResponse,
)
from app.services.db import DatabaseManager, _rls_ensemble_step, resolve_backfill_truth
from app.services.doppler_service import DopplerGridFrame, DopplerRadarService, _mmhr_from_dbz
from app.services.push_worker import PushWorker
from app.services.station_service import (
    HyperlocalStationService,
    _parse_obs_timestamp,
    _parse_wind_dir,
    elevation_adjust_dewpoint,
    run_spatial_buddy_qc,
)
from app.services.weather_provider import (
    WeatherProviderService,
    blend_hourly_precip_and_snow,
    calc_apparent_temp_c,
    calc_snow_liquid_ratio,
)


def _make_textured_blob(h: int, w: int, r0: int, c0: int, radius: int = 24) -> np.ndarray:
    """Builds an asymmetric multi-core synthetic reflectivity echo with a unique ZNCC peak (no periodic aliasing)."""
    grid = np.zeros((h, w), dtype=np.uint8)
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    d_main = np.hypot(yy, xx)
    d_lobe1 = np.hypot(yy - radius * 0.35, xx + radius * 0.30)
    d_lobe2 = np.hypot(yy + radius * 0.30, xx - radius * 0.25)
    sig1 = 105.0 * np.exp(-(d_main ** 2) / (2.0 * (radius * 0.45) ** 2))
    sig2 = 75.0 * np.exp(-(d_lobe1 ** 2) / (2.0 * (radius * 0.22) ** 2))
    sig3 = 55.0 * np.exp(-(d_lobe2 ** 2) / (2.0 * (radius * 0.18) ** 2))
    val = 85.0 + sig1 + sig2 + sig3
    mask = d_main <= radius
    patch = np.where(mask, np.clip(np.round(val), 115, 195), 0).astype(np.uint8)
    r_s, c_s = r0 - radius, c0 - radius
    grid[r_s : r_s + patch.shape[0], c_s : c_s + patch.shape[1]] = patch
    return grid


# ==============================================================================
# 1. DOPPLER ADVECTION, FAST HRRR TRACKING & 1340-ROW PADDING
# ==============================================================================


def test_v2_doppler_zonal_advection_45n_no_cos_overspeed(monkeypatch):
    """At 45N, zonal flow in pixels/15m must NOT be inflated by 1/cos(45) (~1.414x)."""
    svc = DopplerRadarService()
    now = datetime.now(timezone.utc)
    # 45.0N -> row 500; col ~3000 -> east by exactly 20 master cols (5 decimated px at step_dec=4) in 15 min
    r0, c0, true_dc_15m = 500, 3000, 20
    g_ref = _make_textured_blob(svc.GRID_H, svc.GRID_W, r0, c0, radius=32)
    g_live = _make_textured_blob(svc.GRID_H, svc.GRID_W, r0, c0 + true_dc_15m, radius=32)

    frames = [
        DopplerGridFrame("p45", now - timedelta(minutes=15), "-15m", -15, False, g_ref, 0),
        DopplerGridFrame("live", now, "Live", 0, False, g_live, 0),
    ]

    monkeypatch.setattr(settings, "zus_accuracy_v2", True)
    obs_dr_v2, obs_dc_v2, cum_dr_v2, cum_dc_v2, _ = svc._build_curved_trajectories_sync(frames, {})

    monkeypatch.setattr(settings, "zus_accuracy_v2", False)
    _, obs_dc_leg, _, cum_dc_leg, _ = svc._build_curved_trajectories_sync(frames, {})

    # Inspect peak near row=500 (500/2699 * 54 ≈ 10 in flow grid, *109 ≈ 20 in cum grid)
    iy_f = round((r0 / (svc.GRID_H - 1)) * 54)
    ix_f = round(((c0 + true_dc_15m) / (svc.GRID_W - 1)) * 122)
    iy_c = round((r0 / (svc.GRID_H - 1)) * 109)
    ix_c = round(((c0 + true_dc_15m) / (svc.GRID_W - 1)) * 244)

    assert abs(float(obs_dr_v2[iy_f, ix_f])) <= 2.0
    assert 17.5 <= float(obs_dc_v2[iy_f, ix_f]) <= 22.0
    assert 70.0 <= float(cum_dc_v2[4][iy_c, ix_c]) <= 84.0
    # Legacy divided by cos(45°) ≈ 0.7071 -> ~28.3 cols/15m (> 26.0)
    assert float(obs_dc_leg[iy_f, ix_f]) > float(obs_dc_v2[iy_f, ix_f]) * 1.25
    assert float(cum_dc_leg[4][iy_c, ix_c]) > float(cum_dc_v2[4][iy_c, ix_c]) * 1.25


def test_v2_hrrr_coarse_to_fine_tracks_fast_storms_and_pads_south(monkeypatch):
    """Coarse-to-fine ZNCC recovers fast (>60 km/h) shifts missed by max_sh=10, and pads 1340->1350 rows."""
    svc = DopplerRadarService()
    # 2km HRRR size (1340x3050), decimated by step_dec=2 -> scale_15m = (15/30)*4 = 2.0
    # True shift: 24 HRRR px east in 30 min = 12 decimated px = 24.0 master cols/15m (>96 km/h)
    r_h, c_h = 650, 1500
    g1 = _make_textured_blob(1350, 3050, r_h, c_h, radius=24)
    g2 = _make_textured_blob(1350, 3050, r_h, c_h + 24, radius=24)

    monkeypatch.setattr(settings, "zus_accuracy_v2", True)
    dr_v2, dc_v2, conf_v2 = svc._compute_mesoscale_flow_grid(
        g1, g2, minutes_apart=30.0, step_dec=2, grid_step_km=2.0, fh=55, fw=123
    )
    iy_f = round((r_h / 1349.0) * 54)
    ix_f = round(((c_h + 12) / 3049.0) * 122)
    assert conf_v2[iy_f, ix_f] >= 0.90
    assert 22.0 <= float(dc_v2[iy_f, ix_f]) <= 26.0
    assert abs(float(dr_v2[iy_f, ix_f])) <= 2.0
    iy_max, ix_max = np.unravel_index(int(np.argmax(conf_v2)), conf_v2.shape)
    assert 20.0 <= float(dc_v2[iy_max, ix_max]) <= 26.0

    # Legacy max_sh=10 (max shift 20 cols/15m) fails to reach 24 cols/15m
    monkeypatch.setattr(settings, "zus_accuracy_v2", False)
    _, dc_leg, _ = svc._compute_mesoscale_flow_grid(
        g1, g2, minutes_apart=30.0, step_dec=2, grid_step_km=2.0, fh=55, fw=123
    )
    assert float(dc_leg[iy_f, ix_f]) <= 20.5

    # HRRR 1340-row bottom padding inside _build_curved_trajectories_sync
    monkeypatch.setattr(settings, "zus_accuracy_v2", True)
    now = datetime.now(timezone.utc)
    past_empty = [
        DopplerGridFrame("p", now - timedelta(minutes=15), "-15m", -15, False, np.zeros((svc.GRID_H, svc.GRID_W), dtype=np.uint8), 0),
        DopplerGridFrame("l", now, "Live", 0, False, np.zeros((svc.GRID_H, svc.GRID_W), dtype=np.uint8), 0),
    ]
    # Place echo near south edge of 1340x3050 HRRR grid (row 1280 -> row 1280/1350 with pad)
    h0 = _make_textured_blob(1340, 3050, 1280, 1500, radius=20)
    h30 = _make_textured_blob(1340, 3050, 1280, 1524, radius=20)
    obs_dr, _, cum_dr, cum_dc, trends = svc._build_curved_trajectories_sync(past_empty, {0: h0, 30: h30})
    assert obs_dr.shape == (55, 123)
    assert 4 in cum_dc and cum_dc[4].shape == (110, 245)
    # Positive southern-edge HRRR east flow should be picked up near bottom rows (col 1500/3049*244 ≈ 120)
    assert float(np.max(cum_dc[2][98:, 115:130])) > 8.0
    assert float(np.max(cum_dc[4][98:, 115:130])) > 20.0


# ==============================================================================
# 2. PHASE-AWARE Z-R, CONE PROBABILITIES & EXTENDED FALLBACK ADVECTION
# ==============================================================================


def test_v2_phase_aware_zr_and_smooth_threshold(monkeypatch):
    """Phase-aware Z-R relationships and smooth 14.5..18.0 dBZ onset ramp."""
    monkeypatch.setattr(settings, "zus_accuracy_v2", True)
    assert _mmhr_from_dbz(14.4) == 0.0
    r_low = _mmhr_from_dbz(15.0)
    r_mid = _mmhr_from_dbz(16.5)
    r_hi = _mmhr_from_dbz(18.0)
    assert 0.0 < r_low < r_mid < r_hi

    # Stratiform rain (200/1.6) vs Convective rain (300/1.4 at CAPE>=1000 & dBZ>=35)
    r_strat = _mmhr_from_dbz(45.0, precip_type="rain", cape_jkg=100.0)
    r_conv = _mmhr_from_dbz(45.0, precip_type="rain", cape_jkg=1500.0)
    assert r_strat == round(((10.0 ** 4.5) / 200.0) ** (1.0 / 1.6), 2)
    assert r_conv == round(((10.0 ** 4.5) / 300.0) ** (1.0 / 1.4), 2)
    assert r_conv != r_strat

    # Cold snow (75/2.0) vs wet snow (130/2.0) vs sleet (150/1.8)
    r_cold_snow = _mmhr_from_dbz(30.0, precip_type="snow", temp_c=-5.0)
    r_wet_snow = _mmhr_from_dbz(30.0, precip_type="snow", temp_c=0.0)
    r_sleet = _mmhr_from_dbz(30.0, precip_type="sleet", temp_c=0.5)
    assert r_cold_snow == round((1000.0 / 75.0) ** 0.5, 2)
    assert r_wet_snow == round((1000.0 / 130.0) ** 0.5, 2)
    assert r_sleet == round((1000.0 / 150.0) ** (1.0 / 1.8), 2)
    assert r_cold_snow > r_wet_snow


@pytest.mark.asyncio
async def test_v2_point_telemetry_cone_probs_and_errors(monkeypatch):
    """get_point_telemetry populates 61 cone probabilities and spatial intensity errors."""
    monkeypatch.setattr(settings, "zus_accuracy_v2", True)
    svc = DopplerRadarService()
    now = datetime.now(timezone.utc)

    lat, lon = 40.7128, -74.0060
    r_pt = int(round((svc.LAT_MAX - lat) / svc.STEP_DEG - 0.5))
    c_pt = int(round((lon - svc.LON_MIN) / svc.STEP_DEG - 0.5))

    grid = np.zeros((svc.GRID_H, svc.GRID_W), dtype=np.uint8)
    # Place echo edge right at point and upstream west so cone spreads across gradient
    grid[r_pt - 8 : r_pt + 8, c_pt - 30 : c_pt + 2] = 140  # ~37.5 dBZ

    svc._frames = {
        "ref": DopplerGridFrame("ref", now - timedelta(minutes=15), "-15m", -15, False, grid, 0),
        "live": DopplerGridFrame("live", now, "Live", 0, False, grid, 0),
    }
    th, tw = 110, 245
    for s in range(1, 9):
        svc._cum_drow_fields[s] = np.zeros((th, tw), dtype=np.float32)
        svc._cum_dcol_fields[s] = np.full((th, tw), 6.0 * s, dtype=np.float32)

    telem = await svc.get_point_telemetry(lat, lon, precip_type="rain", cape_jkg=1200.0)
    assert telem.inConusGrid is True
    assert len(telem.minuteProbabilities60m) == 61
    assert len(telem.minuteIntensityErrors60m) == 61
    assert all(0.0 <= p <= 1.0 for p in telem.minuteProbabilities60m)
    assert max(telem.minuteIntensityErrors60m) > 0.0

    # Outside CONUS returns 61 zeros
    telem_out = await svc.get_point_telemetry(51.5, -0.12)
    assert telem_out.inConusGrid is False
    assert len(telem_out.minuteProbabilities60m) == 61
    assert all(p == 0.0 for p in telem_out.minuteProbabilities60m)


def test_v2_fallback_advection_beyond_2h_not_frozen(monkeypatch):
    """Fallback advection past +2h (fut_mult > 8.5) must continue moving rather than freezing at step 8."""
    svc = DopplerRadarService()
    lat, lon = 39.0, -98.0
    r_c = int(round((svc.LAT_MAX - lat) / svc.STEP_DEG - 0.5))
    c_c = int(round((lon - svc.LON_MIN) / svc.STEP_DEG - 0.5))
    grid = _make_textured_blob(svc.GRID_H, svc.GRID_W, r_c, c_c, radius=18)

    th, tw = 110, 245
    svc._cum_drow_fields = {8: np.zeros((th, tw), dtype=np.float32)}
    svc._cum_dcol_fields = {8: np.full((th, tw), 120.0, dtype=np.float32)}  # +2h shift = 120 px east

    # Tile covering ~(-96.0 lon, 39.0 lat) at z=7 -> east of storm origin
    n = 2 ** 7
    tx = int(((-96.0 + 180.0) / 360.0) * n)
    lat_rad = math.radians(lat)
    ty = int(((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0) * n)

    monkeypatch.setattr(settings, "zus_accuracy_v2", True)
    png_2h = svc._slice_conus_tile_sync(grid, z=7, x=tx, y=ty, advect_drow=0, advect_dcol=120, fut_mult=8.0)
    png_4h = svc._slice_conus_tile_sync(grid, z=7, x=tx, y=ty, advect_drow=0, advect_dcol=240, fut_mult=16.0)
    assert png_2h != png_4h

    # Legacy froze fut_mult=16 onto step_idx=8 (+2h) -> identical PNG bytes!
    monkeypatch.setattr(settings, "zus_accuracy_v2", False)
    png_2h_leg = svc._slice_conus_tile_sync(grid, z=7, x=tx, y=ty, advect_drow=0, advect_dcol=120, fut_mult=8.0)
    png_4h_leg = svc._slice_conus_tile_sync(grid, z=7, x=tx, y=ty, advect_drow=0, advect_dcol=240, fut_mult=16.0)
    assert png_2h_leg == png_4h_leg


# ==============================================================================
# 3. ORTHOGONAL FREE MODELS, STREAM ALIGNMENT, NOWCAST & THERMODYNAMICS
# ==============================================================================


@pytest.mark.asyncio
async def test_v2_orthogonal_free_models_and_kill_switch(monkeypatch):
    """CONUS uses HRRR/ECMWF/AIFS/NBM/GFS_GLOBAL/ICON/GEM; Global excludes US models; v2=False restores legacy."""
    svc = WeatherProviderService()
    captured = []

    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"hourly": {"time": []}, "daily": {"time": []}}

    class FakeClient:
        async def get(self, url, params=None, timeout=None):
            captured.append((url, dict(params or {})))
            return FakeResp()

    client = FakeClient()

    monkeypatch.setattr(settings, "zus_accuracy_v2", True)
    await svc._fetch_open_meteo_ensemble(client, 40.7128, -74.0060)
    conus_params = [p for _, p in captured[-2:] if "models" in p][0]
    conus_models = conus_params["models"].split(",")
    assert conus_models == [
        "gfs_hrrr",
        "ecmwf_ifs025",
        "ecmwf_aifs025_single",
        "ncep_nbm_conus",
        "gfs_global",
        "icon_seamless",
        "gem_seamless",
    ]
    assert "best_match" not in conus_models and "gfs_seamless" not in conus_models

    await svc._fetch_open_meteo_ensemble(client, 51.5074, -0.1278)
    london_params = [p for _, p in captured[-2:] if "models" in p][0]
    london_models = london_params["models"].split(",")
    assert london_models == [
        "ecmwf_ifs025",
        "ecmwf_aifs025_single",
        "gfs_global",
        "icon_seamless",
        "gem_seamless",
    ]

    monkeypatch.setattr(settings, "zus_accuracy_v2", False)
    await svc._fetch_open_meteo_ensemble(client, 40.7128, -74.0060)
    legacy_params = [p for _, p in captured[-2:] if "models" in p][0]
    legacy_models = legacy_params["models"].split(",")
    assert "best_match" in legacy_models and "gfs_seamless" in legacy_models


@pytest.mark.asyncio
async def test_v2_stream_alignment_and_nullable_pop_no_drag(monkeypatch):
    """
    Missing hourly temp values (None) preserve exact hour index alignment instead of shifting later hours,
    and missing PoP (ecmwf_aifs025_single) does NOT drag consensus PoP down toward 0.
    """
    svc = WeatherProviderService()
    svc._cache.clear()
    monkeypatch.setattr(settings, "zus_accuracy_v2", True)

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    times = [(now + timedelta(hours=i)).strftime("%Y-%m-%dT%H:00") for i in range(48)]
    zeros = [0.0] * 48
    tens = [10.0] * 48
    hum_80 = [80] * 48
    codes_61 = [61] * 48
    pops_80 = [80] * 48
    # ECMWF misses hours 0 and 1 (temp=None) then jumps to 30.0°C at hour 2+
    ecmwf_temps = [None, None] + [30.0] * 46
    # GFS has constant 10.0°C across all 48h and 80% PoP
    gfs_temps = [10.0] * 48
    # AIFS has 10.0°C across all 48h and None PoP everywhere
    aifs_pops = [None] * 48

    hourly_raw = {
        "time": times,
        "temperature_2m_ecmwf_ifs025": ecmwf_temps,
        "temperature_2m_ecmwf_aifs025_single": gfs_temps,
        "temperature_2m_gfs_global": gfs_temps,
        "apparent_temperature_ecmwf_ifs025": ecmwf_temps,
        "apparent_temperature_ecmwf_aifs025_single": gfs_temps,
        "apparent_temperature_gfs_global": gfs_temps,
        "precipitation_probability_ecmwf_ifs025": pops_80,
        "precipitation_probability_ecmwf_aifs025_single": aifs_pops,
        "precipitation_probability_gfs_global": pops_80,
        "precipitation_ecmwf_ifs025": [1.0] * 48,
        "precipitation_ecmwf_aifs025_single": [1.0] * 48,
        "precipitation_gfs_global": [1.0] * 48,
        "wind_speed_10m_ecmwf_ifs025": tens,
        "wind_speed_10m_ecmwf_aifs025_single": tens,
        "wind_speed_10m_gfs_global": tens,
        "dew_point_2m_ecmwf_ifs025": zeros,
        "dew_point_2m_ecmwf_aifs025_single": zeros,
        "dew_point_2m_gfs_global": zeros,
        "relative_humidity_2m_ecmwf_ifs025": hum_80,
        "relative_humidity_2m_ecmwf_aifs025_single": hum_80,
        "relative_humidity_2m_gfs_global": hum_80,
        "cloud_cover_ecmwf_ifs025": [50] * 48,
        "cloud_cover_ecmwf_aifs025_single": [50] * 48,
        "cloud_cover_gfs_global": [50] * 48,
        "weather_code_ecmwf_ifs025": codes_61,
        "weather_code_ecmwf_aifs025_single": codes_61,
        "weather_code_gfs_global": codes_61,
    }
    daily_times = [(now + timedelta(days=d)).strftime("%Y-%m-%d") for d in range(3)]
    raw_ensemble = {
        "utc_offset_seconds": 0,
        "timezone": "UTC",
        "elevation": 10.0,
        "hourly": hourly_raw,
        "daily": {
            "time": daily_times,
            "weather_code_gfs_global": [61, 61, 61],
            "temperature_2m_max_gfs_global": [15.0, 15.0, 15.0],
            "temperature_2m_min_gfs_global": [5.0, 5.0, 5.0],
            "precipitation_probability_max_gfs_global": [80, 80, 80],
            "precipitation_sum_gfs_global": [5.0, 5.0, 5.0],
            "wind_speed_10m_max_gfs_global": [15.0, 15.0, 15.0],
        },
        "current": {"temperature_2m": 10.0, "weather_code": 61},
        "minutely_15": {"time": [], "precipitation": []},
    }

    with patch("app.services.weather_provider.http_pool.get_client", return_value=MagicMock()), \
         patch.object(svc, "_fetch_open_meteo_ensemble", new=AsyncMock(return_value=raw_ensemble)), \
         patch.object(svc, "_fetch_air_quality_and_tz", new=AsyncMock(return_value=(None, None, None, "UTC", 0, None, None, None))), \
         patch.object(svc, "_fetch_nws_polygon_alerts", new=AsyncMock(return_value=[])), \
         patch("app.services.weather_provider.station_service.fetch_nearby_stations", new=AsyncMock(return_value=([], None))), \
         patch("app.services.weather_provider.doppler_service.get_point_telemetry", new=AsyncMock(return_value=None)), \
         patch("app.services.weather_provider.db_manager.register_active_cell", new=AsyncMock(return_value=None)), \
         patch("app.services.weather_provider.db_manager.update_cell_elevation", new=AsyncMock(return_value=None)), \
         patch("app.services.weather_provider.db_manager.get_cell_rls_metadata", new=AsyncMock(return_value={})), \
         patch("app.services.weather_provider.db_manager.get_model_weights", new=AsyncMock(return_value=[])), \
         patch("app.services.weather_provider.db_manager.get_today_observed_temp_bounds", new=AsyncMock(return_value=(None, None))), \
         patch.object(svc, "_run_verification_pipeline", new=AsyncMock(return_value=None)):

        resp = await svc.get_forecast(51.5, -0.12, "London", bypass_cache=True)

    # Hours 0 and 1 only have AIFS & GFS (both 10.0°C) -> MUST stay ~10.0°C, NOT shifted to include ECMWF 30.0°C!
    assert 9.5 <= resp.hourly[0].temperature <= 10.5
    assert 9.5 <= resp.hourly[1].temperature <= 10.5
    # Hour 2 has ECMWF 30.0°C + AIFS 10.0°C + GFS 10.0°C -> jumps noticeably above 10°C (~16.7°C)
    assert resp.hourly[2].temperature > 15.0
    # PoP excludes AIFS None instead of averaging 0 -> stays 80%
    assert resp.hourly[0].precipitationChance == 80


def test_v2_freezing_rain_zero_snow_and_warm_slr_cap(monkeypatch):
    """Freezing rain (WMO 66) produces 0.0 cm snow accum and warm T >= 3C snow caps SLR at 0.0."""
    monkeypatch.setattr(settings, "zus_accuracy_v2", True)
    pt, snow_cm = blend_hourly_precip_and_snow(
        model_entries=[
            {
                "weight": 1.0,
                "temp_c": -1.0,
                "dew_c": -1.5,
                "rh_pct": 95.0,
                "precip_mm": 10.0,
                "precip_prob": 90,
                "wmo_code": 66,
            }
        ],
        total_w=1.0,
        blended_temp_c=-1.0,
        blended_rh=95.0,
        blended_pop=90,
        top_code=66,
        lead_hours=2.0,
    )
    assert pt == "sleet"
    assert snow_cm == 0.0
    assert calc_snow_liquid_ratio(3.5, "snow") == 0.0

    monkeypatch.setattr(settings, "zus_accuracy_v2", False)
    assert calc_snow_liquid_ratio(3.5, "snow") == 4.0


def test_v2_convex_nowcast_blend_no_nwp_max_floor(monkeypatch):
    """When MRMS is dry (0.0 mm/h) and NWP predicts 5.0 mm/h, convex blend is 0.50 (NOT floored at 5.0)."""
    svc = WeatherProviderService()
    now_min = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    raw = {
        "minutely_15": {
            "time": [(now_min + timedelta(minutes=15 * i)).strftime("%Y-%m-%dT%H:%M") for i in range(6)],
            "precipitation": [1.25] * 6,  # 1.25 mm/15m = 5.0 mm/hr
        }
    }
    dop = RadarLocalTelemetry(
        inConusGrid=True,
        currentDbz=0.0,
        precipRateMmHr=0.0,
        minuteByMinuteRates=[0.0] * 61,
        minuteProbabilities60m=[0.05] * 61,
    )

    monkeypatch.setattr(settings, "zus_accuracy_v2", True)
    nc_v2 = svc._build_nowcast(raw, doppler=dop, station_composite=None)
    # w_rad=0.90 at t=0 -> 0.90*0.0 + 0.10*5.0 = 0.50 mm/hr; w_rad=0.55 at t=60 -> 0.45*5.0 = 2.25 mm/hr
    assert dop.minuteByMinuteRates[0] == pytest.approx(0.50, abs=0.02)
    assert dop.minuteByMinuteRates[60] == pytest.approx(2.25, abs=0.02)
    assert nc_v2.points[0].precipProbability == 5

    monkeypatch.setattr(settings, "zus_accuracy_v2", False)
    dop.minuteByMinuteRates = [0.0] * 61
    svc._build_nowcast(raw, doppler=dop, station_composite=None)
    assert dop.minuteByMinuteRates[0] == pytest.approx(0.60, abs=0.02)
    assert dop.minuteByMinuteRates[60] == pytest.approx(1.40, abs=0.02)
    # When radar has tiny drizzle (0.1) and NWP has 5.0, legacy max(rad, blended) keeps blended; convex is strictly weighted
    dop_rad = RadarLocalTelemetry(inConusGrid=True, currentDbz=25.0, precipRateMmHr=2.0, minuteByMinuteRates=[2.0] * 61)
    monkeypatch.setattr(settings, "zus_accuracy_v2", True)
    nc_conv = svc._build_nowcast({"minutely_15": {"time": [], "precipitation": []}}, doppler=dop_rad, station_composite=None)
    assert nc_conv.points[0].precipIntensityMmHr == pytest.approx(1.80, abs=0.02)  # 0.90*2.0 + 0.10*0.0
    monkeypatch.setattr(settings, "zus_accuracy_v2", False)
    dop_rad.minuteByMinuteRates = [2.0] * 61
    nc_leg = svc._build_nowcast({"minutely_15": {"time": [], "precipitation": []}}, doppler=dop_rad, station_composite=None)
    assert nc_leg.points[0].precipIntensityMmHr == 2.0  # legacy max() floor pinned it at 2.0


def test_v2_apparent_temperature_regimes():
    """Wind chill (< -15°C at -10°C/30kmh), heat index (> 45°C at 35°C/70%), mild Steadman (~18°C)."""
    wc = calc_apparent_temp_c(-10.0, 60.0, 30.0)
    hi = calc_apparent_temp_c(35.0, 70.0, 5.0)
    mild = calc_apparent_temp_c(18.0, 50.0, 10.0)
    assert wc < -15.0
    assert hi > 45.0
    assert 15.0 <= mild <= 19.5


# ==============================================================================
# 4. STATION QC (N=3,4,5), STALENESS REJECTION, DEWPOINT LAPSE & WIND PARSER
# ==============================================================================


def test_v2_spatial_buddy_qc_small_samples(monkeypatch):
    """Leave-one-out Median+MAD rejects single extreme outliers even at N=3, 4, 5."""
    monkeypatch.setattr(settings, "zus_accuracy_v2", True)
    for n in (3, 4, 5):
        cands = [{"adj_temp_c": 20.0 + 0.1 * i} for i in range(n - 1)] + [{"adj_temp_c": 35.0}]
        out = run_spatial_buddy_qc(cands)
        assert all(c["qc_passed"] is True for c in out[:-1])
        assert out[-1]["qc_passed"] is False

    # Contrast: legacy inclusive pstdev capped z <= sqrt(2) = 1.414 at N=3 and failed to reject
    monkeypatch.setattr(settings, "zus_accuracy_v2", False)
    out_leg = run_spatial_buddy_qc([{"adj_temp_c": 20.0}, {"adj_temp_c": 20.2}, {"adj_temp_c": 35.0}])
    assert out_leg[-1]["qc_passed"] is True


@pytest.mark.asyncio
async def test_v2_station_staleness_rejection_and_utilities(monkeypatch):
    """Rejects >65m stale station obs; validates dewpoint lapse clamp and VRB wind dir."""
    monkeypatch.setattr(settings, "zus_accuracy_v2", True)
    monkeypatch.setattr(settings, "station_max_age_minutes", 65)

    assert _parse_wind_dir("VRB") is None
    assert _parse_wind_dir("") is None
    assert _parse_wind_dir(None) is None
    assert _parse_wind_dir("270") == 270
    assert _parse_wind_dir(360) == 0

    # 1000m higher station -> +2.0°C downslope dewpoint adjustment, strictly clamped <= air temp
    assert elevation_adjust_dewpoint(10.0, station_elev_m=1500.0, target_elev_m=500.0, max_temp_c=11.0) == 11.0
    assert elevation_adjust_dewpoint(10.0, station_elev_m=1500.0, target_elev_m=500.0, max_temp_c=15.0) == 12.0

    now = datetime.now(timezone.utc)
    fresh_iso = (now - timedelta(minutes=15)).isoformat()
    stale_iso = (now - timedelta(minutes=120)).isoformat()
    assert _parse_obs_timestamp(fresh_iso) is not None

    svc = HyperlocalStationService()
    metar_payload = [
        {
            "icaoId": "KSTL1",
            "name": "Stale Station",
            "lat": 40.71,
            "lon": -74.00,
            "elev": 10.0,
            "temp": 25.0,
            "dewp": 10.0,
            "wspd": 10.0,
            "wdir": "VRB",
            "obsTime": stale_iso,
        },
        {
            "icaoId": "KFRSH",
            "name": "Fresh Station",
            "lat": 40.72,
            "lon": -74.01,
            "elev": 10.0,
            "temp": 18.0,
            "dewp": 12.0,
            "wspd": 10.0,
            "wdir": "VRB",
            "obsTime": fresh_iso,
        },
    ]

    class FakeResp:
        def __init__(self, status_code, body):
            self.status_code = status_code
            self._body = body

        def json(self):
            return self._body

    class FakeClient:
        async def get(self, url, params=None, timeout=None):
            if "metar" in url:
                return FakeResp(200, metar_payload)
            return FakeResp(200, {"features": []})

    with patch("app.services.station_service.http_pool.get_client", return_value=FakeClient()), \
         patch("app.services.station_service.db_manager.batch_upsert_stations_and_observations", new=AsyncMock(return_value={})):
        readings, comp = await svc.fetch_nearby_stations(40.7128, -74.0060, target_elev_m=10.0)

    assert len(readings) == 1
    assert readings[0].stationId == "KFRSH"
    assert comp is not None and comp["temperature_c"] == pytest.approx(18.0, abs=0.05)


# ==============================================================================
# 5. VERIFICATION TRUTH RESOLUTION, RLS COVARIANCE & CELL ID NORMALIZATION
# ==============================================================================


def test_v2_verification_truth_rls_psd_and_cell_id(monkeypatch):
    """resolve_backfill_truth handles 0.0C + blocks NWP self-grading; RLS stays symmetric & PSD."""
    monkeypatch.setattr(settings, "zus_accuracy_v2", True)
    hr = datetime(2026, 11, 23, 12, 0, tzinfo=timezone.utc)

    # Explicit 0.0°C observation must NOT be treated as falsy!
    assert resolve_backfill_truth(hr, {hr: 0.0}, {hr: 5.0}, use_v2=True) == (0.0, True)
    # Missing station observation blocks NWP self-grading in v2, allows in legacy
    assert resolve_backfill_truth(hr, {}, {hr: 5.0}, use_v2=True) == (None, False)
    assert resolve_backfill_truth(hr, {}, {hr: 5.0}, use_v2=False) == (5.0, False)

    assert DatabaseManager.make_cell_id(-0.0, 0.0) == "0.0:0.0"
    assert DatabaseManager.make_cell_id(-0.001, -0.001) == "0.0:0.0"

    # Stress-test RLS covariance preservation over 50 collinear/extreme steps
    k = 6
    w = [1.0 / k] * k
    cov = [0.35 if i == j else 0.0 for i in range(k) for j in range(k)]
    for step in range(50):
        # First 3 models near-identical, others varying
        x = [20.0 + 0.01 * (step % 3), 20.01 + 0.01 * (step % 3), 19.99, 24.0, 15.0, 21.0]
        w, cov = _rls_ensemble_step(w, cov, x, y_true=20.0)

    p_mat = np.array(cov, dtype=np.float64).reshape(k, k)
    assert np.max(np.abs(p_mat - p_mat.T)) < 1e-4
    diag = np.diag(p_mat)
    assert np.all(diag >= 0.02) and np.all(diag <= 2.5)
    eigvals = np.linalg.eigvalsh(p_mat)
    assert np.min(eigvals) >= -1e-4
    assert sum(w) == pytest.approx(1.0, abs=1e-4)
    assert min(w) >= 0.03


# ==============================================================================
# 6. ENDPOINTS, LIGHTNING UTC INDEX & PUSH HYSTERESIS
# ==============================================================================


def test_v2_weather_zero_fahrenheit_and_zero_wind_not_falsy():
    """-17.7778°C converts to 0.0°F without falling back to -17.7778°C; 0.0 wind stays 0.0."""
    cur = WeatherDataModel(
        cityName="ZeroVille",
        temperature=-17.7778,
        feelsLike=-17.7778,
        condition=WeatherCondition.CLEAR,
        tempMin=-17.7778,
        tempMax=-17.7778,
        humidity=60,
        windSpeed=0.0,
        windDirection=0,
        uvIndex=0.0,
        airPressure=1013.0,
    )
    h0 = HourlyForecastModel(
        time=datetime.now(timezone.utc),
        temperature=-17.7778,
        condition=WeatherCondition.CLEAR,
        precipitationChance=0,
        windSpeed=0.0,
    )
    d0 = DailyForecastModel(
        date=datetime.now(timezone.utc),
        condition=WeatherCondition.CLEAR,
        minTemperature=-17.7778,
        maxTemperature=-17.7778,
        precipitationChance=0,
        windSpeedMax=0.0,
    )
    resp = WeatherResponse(current=cur, hourly=[h0], daily=[d0])
    us_resp = _apply_weather_transforms(resp, units="us", exclude=None, hours=None)
    assert us_resp.current.temperature == 0.0
    assert us_resp.current.feelsLike == 0.0
    assert us_resp.current.tempMin == 0.0
    assert us_resp.current.tempMax == 0.0
    assert us_resp.current.windSpeed == 0.0
    assert us_resp.hourly[0].temperature == 0.0
    assert us_resp.hourly[0].windSpeed == 0.0
    assert us_resp.daily[0].minTemperature == 0.0
    assert us_resp.daily[0].maxTemperature == 0.0
    assert us_resp.daily[0].windSpeedMax == 0.0


@pytest.mark.asyncio
async def test_v2_standard_endpoints_contracts_and_headers(monkeypatch):
    """Verifies Dark Sky cone pass-through, live UTC minute, zero precip fabrication, local-day grouping, solar wrap, and tile headers."""
    monkeypatch.setattr(settings, "zus_accuracy_v2", True)

    # Wrapped solar window (Tokyo: sunrise 21:00Z, sunset 08:00Z on same UTC date)
    sr_wrap = int(datetime(2026, 11, 23, 21, 0, tzinfo=timezone.utc).timestamp())
    ss_wrap = int(datetime(2026, 11, 23, 8, 0, tzinfo=timezone.utc).timestamp())
    t_day_tokyo = int(datetime(2026, 11, 24, 3, 0, tzinfo=timezone.utc).timestamp())
    t_night_tokyo = int(datetime(2026, 11, 24, 15, 0, tzinfo=timezone.utc).timestamp())
    assert _is_daytime_ts(t_day_tokyo, sr_wrap, ss_wrap, utc_offset_seconds=32400) is True
    assert _is_daytime_ts(t_night_tokyo, sr_wrap, ss_wrap, utc_offset_seconds=32400) is False

    # Construct fixture in America/New_York (UTC-5) with cur.timestamp 10 min stale
    tz_off = -18000
    old_cur_ts = datetime.now(timezone.utc) - timedelta(minutes=10)
    local_day0 = datetime(2026, 11, 24, 0, 0, tzinfo=timezone.utc)  # local midnight start in UTC-5 = 05:00Z
    hourly_list = []
    # Include 4 UTC hours belonging to previous local day (01Z..04Z) + 24 hours on Nov 24 local
    start_utc = datetime(2026, 11, 24, 1, 0, tzinfo=timezone.utc)
    for i in range(48):
        dt_h = start_utc + timedelta(hours=i)
        local_date = (dt_h + timedelta(seconds=tz_off)).date()
        is_target_day = local_date == datetime(2026, 11, 24).date()
        hourly_list.append(
            HourlyForecastModel(
                time=dt_h,
                temperature=22.0 if is_target_day else 3.0,
                condition=WeatherCondition.PARTLY_CLOUDY,
                precipitationChance=50,
                precipMm=0.0,  # Zero QPE despite 50% PoP -> must NOT fabricate precipIntensity!
                humidity=80 if is_target_day else 30,
            )
        )

    daily_list = [
        DailyForecastModel(
            date=datetime(2026, 11, 23, 12, 0, tzinfo=timezone.utc),
            condition=WeatherCondition.CLEAR,
            minTemperature=3.0,
            maxTemperature=5.0,
            precipitationChance=40,
            precipSumMm=0.0,
        ),
        DailyForecastModel(
            date=datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc),
            condition=WeatherCondition.PARTLY_CLOUDY,
            minTemperature=18.0,
            maxTemperature=24.0,
            precipitationChance=50,
            precipSumMm=0.0,
        ),
    ]
    cur = WeatherDataModel(
        cityName="Unknown Location",
        timestamp=old_cur_ts,
        temperature=15.0,
        feelsLike=15.0,
        condition=WeatherCondition.PARTLY_CLOUDY,
        tempMin=10.0,
        tempMax=20.0,
        humidity=50,
        windSpeed=10.0,
        windDirection=180,
        uvIndex=3.0,
        airPressure=1013.0,
        precipType="rain",
    )
    cone_probs = [0.42] * 61
    cone_errs = [0.77] * 61
    mock_resp = WeatherResponse(
        current=cur,
        hourly=hourly_list,
        daily=daily_list,
        dopplerTelemetry=RadarLocalTelemetry(
            inConusGrid=True,
            currentDbz=28.0,
            precipRateMmHr=1.5,
            minuteByMinuteRates=[1.5] * 61,
            minuteProbabilities60m=cone_probs,
            minuteIntensityErrors60m=cone_errs,
        ),
        timezone="America/New_York",
        utcOffsetSeconds=tz_off,
    )

    mock_gf = AsyncMock(return_value=mock_resp)
    live_min_ts = int(datetime.now(timezone.utc).replace(second=0, microsecond=0).timestamp())
    with patch("app.api.v1.endpoints.standard.weather_service.get_forecast", mock_gf):
        ds_payload = await _build_darksky_payload(40.7128, -74.0060, units="si")

    assert mock_gf.await_args.kwargs["city_name"] == "Unknown Location"
    assert abs(ds_payload["minutely"]["data"][0]["time"] - live_min_ts) <= 60
    assert ds_payload["minutely"]["data"][0]["precipProbability"] == 0.42
    assert ds_payload["minutely"]["data"][0]["precipIntensityError"] == 0.77
    assert ds_payload["hourly"]["data"][0]["precipIntensity"] == 0.0
    assert ds_payload["daily"]["data"][1]["humidity"] == 0.80

    # Tile cache headers & /data/2.5/weather city_name guard
    client = TestClient(app)
    with patch("app.api.v1.endpoints.standard.doppler_service.get_tile_png", new=AsyncMock(return_value=b"PNG")), \
         patch("app.api.v1.endpoints.radar.doppler_service.get_tile_png", new=AsyncMock(return_value=b"PNG")), \
         patch("app.api.v1.endpoints.standard.weather_service.get_forecast", mock_gf):
        r_rv_mut = client.get("/v2/radar/past_5m/256/5/10/12/2/1_1.png")
        r_rd_mut = client.get("/api/v1/radar/tiles/past_5m/5/10/12.png")
        r_rv_imm = client.get("/v2/radar/mrms_202611240300/256/5/10/12/2/1_1.png")
        r_owm = client.get("/data/2.5/weather?lat=40.7128&lon=-74.0060")

    assert "max-age=120" in r_rv_mut.headers["cache-control"]
    assert "immutable" not in r_rv_mut.headers["cache-control"]
    assert "max-age=120" in r_rd_mut.headers["cache-control"]
    assert "max-age=86400" in r_rv_imm.headers["cache-control"]
    assert "immutable" in r_rv_imm.headers["cache-control"]
    assert r_owm.json()["name"] == "40.713,-74.006"


def test_v2_lightning_selects_current_utc_hour(monkeypatch):
    """Fallback Open-Meteo query in /lightning must read current UTC hour, not index 0."""
    monkeypatch.setattr(settings, "zus_accuracy_v2", True)
    now_utc = datetime.now(timezone.utc)
    cur_hour = now_utc.hour
    base_day = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    times = [(base_day + timedelta(hours=h)).strftime("%Y-%m-%dT%H:00") for h in range(24)]
    capes = [9999.0] * 24
    capes[cur_hour] = 150.0
    lis = [-8.0] * 24
    lis[cur_hour] = 1.2

    class FakeResp:
        status_code = 200

        def json(self):
            return {
                "current": {"weather_code": 0},
                "hourly": {"time": times, "cape": capes, "lifted_index": lis},
            }

    class FakeClient:
        async def get(self, url, params=None, timeout=None):
            return FakeResp()

    client = TestClient(app)
    with patch("app.api.v1.endpoints.lightning.weather_service._cache", {}), \
         patch("app.api.v1.endpoints.lightning.http_pool.get_client", return_value=FakeClient()):
        res = client.get("/api/v1/lightning?lat=12.345&lon=67.890")

    assert res.status_code == 200
    body = res.json()
    assert body["capeJkg"] == 150.0
    assert body["liftedIndex"] == 1.2
    assert body["stormThreatLevel"] == "Low"


@pytest.mark.asyncio
async def test_v2_push_worker_blip_rejection_and_escalation(tmp_path, monkeypatch):
    """Single-minute radar blip stays IDLE; escalating APPROACHING storm notifies heavy tier without re-notifying any."""
    monkeypatch.setattr(settings, "zus_accuracy_v2", True)
    worker = PushWorker()
    worker._db_path = tmp_path / "test_v2_push.db"
    worker._init_sqlite()

    await worker.register_subscription(
        PushSubscriptionRegisterRequest(
            device_id="dev_any",
            expo_push_token="ExponentPushToken[any]",
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
            device_id="dev_heavy",
            expo_push_token="ExponentPushToken[heavy]",
            lat=40.71,
            lon=-74.00,
            timezone="America/New_York",
            notifications_enabled=True,
            dnd_enabled=False,
            precip_alerts_enabled=True,
            precip_sensitivity="heavy",
        )
    )

    # 1. Single-minute radar spike -> rejected (< 3 of 5 minutes), remains IDLE
    blip_rates = [0.0] * 61
    blip_rates[20] = 2.0
    t_blip = RadarLocalTelemetry(inConusGrid=True, currentDbz=0.0, precipRateMmHr=0.0, minuteByMinuteRates=blip_rates)
    with patch("app.services.push_worker.doppler_service.get_point_nowcast", new=AsyncMock(return_value=t_blip)), \
         patch("app.services.push_worker.weather_service._fetch_nws_polygon_alerts", new=AsyncMock(return_value=[])):
        res_blip = await worker.evaluate_and_dispatch()
    assert res_blip["dispatched"] == 0
    assert (await worker.get_stats()).cell_states.get("APPROACHING", 0) == 0

    # 2. Cycle 1: Persistent light rain (1.0 mm/h for 10 min) -> notifies dev_any ONLY
    light_rates = [0.0] * 61
    for m in range(15, 25):
        light_rates[m] = 1.0
    t_light = RadarLocalTelemetry(inConusGrid=True, currentDbz=0.0, precipRateMmHr=0.0, minuteByMinuteRates=light_rates)
    with patch("app.services.push_worker.doppler_service.get_point_nowcast", new=AsyncMock(return_value=t_light)), \
         patch("app.services.push_worker.weather_service._fetch_nws_polygon_alerts", new=AsyncMock(return_value=[])):
        res_light = await worker.evaluate_and_dispatch()
    assert res_light["dispatched"] == 1
    assert (await worker.get_stats()).cell_states.get("APPROACHING", 0) == 1

    # 3. Cycle 2: Storm intensifies to heavy rain (8.0 mm/h for 10 min) -> notifies dev_heavy without re-notifying dev_any
    heavy_rates = [0.0] * 61
    for m in range(10, 20):
        heavy_rates[m] = 8.0
    t_heavy = RadarLocalTelemetry(inConusGrid=True, currentDbz=0.0, precipRateMmHr=0.0, minuteByMinuteRates=heavy_rates)
    sent_payloads = []
    worker._dispatch_expo_batches = AsyncMock(side_effect=lambda client, p: sent_payloads.extend(p))
    with patch("app.services.push_worker.doppler_service.get_point_nowcast", new=AsyncMock(return_value=t_heavy)), \
         patch("app.services.push_worker.weather_service._fetch_nws_polygon_alerts", new=AsyncMock(return_value=[])):
        res_heavy = await worker.evaluate_and_dispatch()
    assert res_heavy["dispatched"] == 1
    assert [p["to"] for p in sent_payloads] == ["ExponentPushToken[heavy]"]
