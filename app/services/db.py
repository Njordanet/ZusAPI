import asyncpg
import math
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
from app.config import settings

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS postgis;

-- 1. Adaptive "Follow-the-User" Grid Cells
CREATE TABLE IF NOT EXISTS tracked_cells (
    cell_id TEXT PRIMARY KEY,
    city_name TEXT NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    elevation_m DOUBLE PRECISION DEFAULT 0.0,
    region_mode TEXT NOT NULL DEFAULT 'GLOBAL_ENSEMBLE', -- 'US_HYPERLOCAL' or 'GLOBAL_ENSEMBLE'
    is_coastal BOOLEAN DEFAULT FALSE,
    request_count INTEGER DEFAULT 1,
    last_requested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    geom GEOMETRY(Point, 4326)
);

CREATE INDEX IF NOT EXISTS idx_tracked_cells_geom ON tracked_cells USING GIST (geom);

-- 2. Personal Weather Stations (CWOP/APRS) & Official METAR/NWS Stations
CREATE TABLE IF NOT EXISTS weather_stations (
    station_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    network TEXT NOT NULL, -- 'CWOP_PWS', 'NWS_METAR', 'GLOBAL_METAR', 'SYNOPTIC'
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    elevation_m DOUBLE PRECISION DEFAULT 0.0,
    reliability_score DOUBLE PRECISION DEFAULT 1.0,
    residual_bias_c DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    residual_mae_c DOUBLE PRECISION NOT NULL DEFAULT 0.6,
    outlier_strikes INTEGER DEFAULT 0,
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    geom GEOMETRY(Point, 4326)
);

ALTER TABLE weather_stations ADD COLUMN IF NOT EXISTS residual_bias_c DOUBLE PRECISION NOT NULL DEFAULT 0.0;
ALTER TABLE weather_stations ADD COLUMN IF NOT EXISTS residual_mae_c DOUBLE PRECISION NOT NULL DEFAULT 0.6;
CREATE INDEX IF NOT EXISTS idx_weather_stations_geom ON weather_stations USING GIST (geom);

-- 3. Ground-Truth Station Observations (Time-Series)
CREATE TABLE IF NOT EXISTS station_observations (
    id BIGSERIAL PRIMARY KEY,
    station_id TEXT NOT NULL REFERENCES weather_stations(station_id) ON DELETE CASCADE,
    observed_at TIMESTAMPTZ NOT NULL,
    temperature_c DOUBLE PRECISION NOT NULL,
    dewpoint_c DOUBLE PRECISION,
    humidity INTEGER,
    wind_speed_kmh DOUBLE PRECISION,
    wind_dir INTEGER,
    pressure_hpa DOUBLE PRECISION,
    precip_rate_mm DOUBLE PRECISION DEFAULT 0.0,
    qc_passed BOOLEAN NOT NULL DEFAULT TRUE,
    qc_zscore DOUBLE PRECISION DEFAULT 0.0,
    spatial_residual_c DOUBLE PRECISION DEFAULT 0.0,
    UNIQUE (station_id, observed_at)
);

ALTER TABLE station_observations ADD COLUMN IF NOT EXISTS spatial_residual_c DOUBLE PRECISION DEFAULT 0.0;
CREATE INDEX IF NOT EXISTS idx_station_obs_time ON station_observations (observed_at DESC);
CREATE INDEX IF NOT EXISTS idx_station_obs_station ON station_observations (station_id, observed_at DESC);

-- 4. Multi-Model Forecast Snapshots (For Multi-Horizon RLS Verification & Bias Grading)
CREATE TABLE IF NOT EXISTS forecast_snapshots (
    id BIGSERIAL PRIMARY KEY,
    cell_id TEXT NOT NULL REFERENCES tracked_cells(cell_id) ON DELETE CASCADE,
    model_name TEXT NOT NULL, -- 'ECMWF', 'HRRR', 'GFS', 'ICON', 'GEM'
    issued_at TIMESTAMPTZ NOT NULL,
    target_hour TIMESTAMPTZ NOT NULL,
    lead_hours INTEGER NOT NULL,
    predicted_temp_c DOUBLE PRECISION NOT NULL,
    predicted_precip_prob INTEGER NOT NULL DEFAULT 0,
    predicted_wind_kmh DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    wmo_code INTEGER NOT NULL DEFAULT 0,
    verified BOOLEAN NOT NULL DEFAULT FALSE,
    actual_temp_c DOUBLE PRECISION,
    abs_error_c DOUBLE PRECISION,
    signed_bias_c DOUBLE PRECISION,
    actual_wind_kmh DOUBLE PRECISION,
    wind_abs_error_kmh DOUBLE PRECISION,
    brier_error DOUBLE PRECISION,
    UNIQUE (cell_id, model_name, target_hour, lead_hours)
);

ALTER TABLE forecast_snapshots ADD COLUMN IF NOT EXISTS actual_wind_kmh DOUBLE PRECISION;
ALTER TABLE forecast_snapshots ADD COLUMN IF NOT EXISTS wind_abs_error_kmh DOUBLE PRECISION;
ALTER TABLE forecast_snapshots ADD COLUMN IF NOT EXISTS brier_error DOUBLE PRECISION;
CREATE INDEX IF NOT EXISTS idx_forecast_snapshots_verify
    ON forecast_snapshots (cell_id, verified, target_hour);

-- 5. Dynamic Accuracy Weights, Horizon Buckets & Diurnal Harmonic MOS Bias Ledger
CREATE TABLE IF NOT EXISTS model_accuracy_weights (
    cell_id TEXT NOT NULL REFERENCES tracked_cells(cell_id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    mae_temp_c DOUBLE PRECISION NOT NULL DEFAULT 1.2,
    bias_temp_c DOUBLE PRECISION NOT NULL DEFAULT 0.0, -- Mean offset (beta_0)
    bias_sin_c DOUBLE PRECISION NOT NULL DEFAULT 0.0,  -- Diurnal sin(2*pi*h/24) harmonic
    bias_cos_c DOUBLE PRECISION NOT NULL DEFAULT 0.0,  -- Diurnal cos(2*pi*h/24) harmonic
    mae_wind_kmh DOUBLE PRECISION NOT NULL DEFAULT 3.5,
    bias_wind_kmh DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    brier_precip DOUBLE PRECISION NOT NULL DEFAULT 0.12,
    dynamic_weight DOUBLE PRECISION NOT NULL DEFAULT 0.20, -- Composite headline weight
    weight_short DOUBLE PRECISION NOT NULL DEFAULT 0.20,   -- 1-6h RLS weight (nowcast/mesoscale)
    weight_med DOUBLE PRECISION NOT NULL DEFAULT 0.20,     -- 7-24h RLS weight (diurnal)
    weight_ext DOUBLE PRECISION NOT NULL DEFAULT 0.20,     -- 25h+ RLS weight (synoptic)
    accuracy_pct DOUBLE PRECISION NOT NULL DEFAULT 92.0,
    samples_verified INTEGER NOT NULL DEFAULT 1,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (cell_id, model_name)
);

ALTER TABLE model_accuracy_weights ADD COLUMN IF NOT EXISTS bias_sin_c DOUBLE PRECISION NOT NULL DEFAULT 0.0;
ALTER TABLE model_accuracy_weights ADD COLUMN IF NOT EXISTS bias_cos_c DOUBLE PRECISION NOT NULL DEFAULT 0.0;
ALTER TABLE model_accuracy_weights ADD COLUMN IF NOT EXISTS mae_wind_kmh DOUBLE PRECISION NOT NULL DEFAULT 3.5;
ALTER TABLE model_accuracy_weights ADD COLUMN IF NOT EXISTS bias_wind_kmh DOUBLE PRECISION NOT NULL DEFAULT 0.0;
ALTER TABLE model_accuracy_weights ADD COLUMN IF NOT EXISTS weight_short DOUBLE PRECISION NOT NULL DEFAULT 0.20;
ALTER TABLE model_accuracy_weights ADD COLUMN IF NOT EXISTS weight_med DOUBLE PRECISION NOT NULL DEFAULT 0.20;
ALTER TABLE model_accuracy_weights ADD COLUMN IF NOT EXISTS weight_ext DOUBLE PRECISION NOT NULL DEFAULT 0.20;
UPDATE model_accuracy_weights
SET weight_short = dynamic_weight, weight_med = dynamic_weight, weight_ext = dynamic_weight
WHERE weight_short = 0.20 AND weight_med = 0.20 AND weight_ext = 0.20 AND dynamic_weight != 0.20;

-- 6. Cell-Level Constrained Recursive Least Squares (RLS) Covariance & Radar Z-R State
CREATE TABLE IF NOT EXISTS cell_rls_state (
    cell_id TEXT PRIMARY KEY REFERENCES tracked_cells(cell_id) ON DELETE CASCADE,
    model_order TEXT[] NOT NULL,
    cov_short DOUBLE PRECISION[] NOT NULL, -- Flattened KxK inverse covariance P_short
    cov_med DOUBLE PRECISION[] NOT NULL,   -- Flattened KxK inverse covariance P_med
    cov_ext DOUBLE PRECISION[] NOT NULL,   -- Flattened KxK inverse covariance P_ext
    radar_zr_factor DOUBLE PRECISION NOT NULL DEFAULT 1.0, -- Virga / Z-R calibration multiplier
    rls_updates INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 7. Züs CONUS 1km Doppler Radar Grid Snapshots Ledger
CREATE TABLE IF NOT EXISTS doppler_grid_frames (
    frame_id TEXT PRIMARY KEY,
    frame_time TIMESTAMPTZ NOT NULL,
    grid_shape TEXT NOT NULL DEFAULT '2700x6100',
    resolution_km DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    max_dbz DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    echo_coverage_pct DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    storm_cells_detected INTEGER NOT NULL DEFAULT 0,
    raw_bytes INTEGER NOT NULL DEFAULT 0,
    is_nowcast BOOLEAN NOT NULL DEFAULT FALSE,
    backfilled BOOLEAN NOT NULL DEFAULT FALSE,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE doppler_grid_frames ADD COLUMN IF NOT EXISTS backfilled BOOLEAN NOT NULL DEFAULT FALSE;
CREATE INDEX IF NOT EXISTS idx_doppler_frames_time ON doppler_grid_frames (frame_time DESC);
"""


def eval_diurnal_bias(beta0: float, beta_sin: float, beta_cos: float, utc_hour: int) -> float:
    """
    Evaluates the 3-parameter harmonic diurnal bias curve at a given UTC hour (0-23):
      Bias(h) = beta0 + beta_sin * sin(2*pi*h/24) + beta_cos * cos(2*pi*h/24)
    """
    rad = (2.0 * math.pi * (utc_hour % 24)) / 24.0
    val = beta0 + beta_sin * math.sin(rad) + beta_cos * math.cos(rad)
    return max(-6.0, min(6.0, val))


def _harmonic_bias_rls_step(
    beta0: float,
    beta_sin: float,
    beta_cos: float,
    utc_hour: int,
    observed_bias: float,
    lr: float = 0.08,
    l2_decay: float = 0.02,
) -> Tuple[float, float, float]:
    """
    O(1) Regularized Recursive Least Squares / NLMS step for the 3-parameter diurnal harmonic bias curve.
    L2 shrinkage on the harmonic terms prevents oscillation when observations only arrive at certain hours.
    """
    rad = (2.0 * math.pi * (utc_hour % 24)) / 24.0
    s_h = math.sin(rad)
    c_h = math.cos(rad)
    pred_bias = beta0 + beta_sin * s_h + beta_cos * c_h
    innovation = max(-8.0, min(8.0, observed_bias - pred_bias))

    # Feature norm squared ||[1, sin, cos]||^2 = 1 + sin^2 + cos^2 = 2.0
    step = lr * innovation / 2.0
    new_b0 = max(-5.0, min(5.0, beta0 + step * 1.2))
    new_bsin = max(-3.0, min(3.0, beta_sin * (1.0 - l2_decay) + step * s_h))
    new_bcos = max(-3.0, min(3.0, beta_cos * (1.0 - l2_decay) + step * c_h))
    return round(new_b0, 4), round(new_bsin, 4), round(new_bcos, 4)


def _project_capped_simplex(weights: List[float], min_w: float = 0.03) -> List[float]:
    """
    Projects an arbitrary K-dimensional weight vector onto the probability simplex
    with floor constraint w_i >= min_w and sum(w_i) = 1.0 in O(K log K) time.
    """
    k = len(weights)
    if k == 0:
        return []
    if k * min_w >= 1.0:
        return [round(1.0 / k, 4) for _ in range(k)]

    # Shift by min_w and project onto simplex of mass (1 - k * min_w)
    target_mass = 1.0 - k * min_w
    shifted = [w - min_w for w in weights]
    u = sorted(shifted, reverse=True)
    cssv = 0.0
    rho = 0
    for j in range(k):
        cssv += u[j]
        if u[j] - (cssv - target_mass) / (j + 1) > 0:
            rho = j
    theta = (sum(u[: rho + 1]) - target_mass) / (rho + 1)
    projected = [max(0.0, v - theta) + min_w for v in shifted]
    total = sum(projected) or 1.0
    return [round(v / total, 4) for v in projected]


def _rls_ensemble_step(
    weights: List[float],
    cov_flat: List[float],
    x_vec: List[float],
    y_true: float,
    forgetting_factor: float = 0.965,
    ridge_diag: float = 0.015,
) -> Tuple[List[float], List[float]]:
    """
    O(K^2) Constrained Recursive Least Squares (RLS) update for K ensemble models.
    Accounts for inter-model error covariance so collinear models don't gang up on independent models.
    - weights: length K prior weight vector
    - cov_flat: length K*K flattened inverse covariance matrix P_{t-1}
    - x_vec: length K vector of bias-corrected model predictions
    - y_true: ground-truth observation
    """
    k = len(weights)
    if k == 0 or len(cov_flat) != k * k or len(x_vec) != k:
        return weights, cov_flat

    # Center predictions around y_true to work directly in error space e_i = x_i - y_true
    # Minimizing (w^T x - y)^2 s.t. sum(w)=1 is equivalent to minimizing (w^T e)^2
    e_vec = [max(-12.0, min(12.0, x_i - y_true)) for x_i in x_vec]

    # 1. Compute P * e_vec (K operations per row -> K^2 ops)
    p_e = [0.0] * k
    for i in range(k):
        row_off = i * k
        s = 0.0
        for j in range(k):
            s += cov_flat[row_off + j] * e_vec[j]
        p_e[i] = s

    # 2. Innovation denominator: lambda + e^T P e
    e_p_e = sum(e_vec[i] * p_e[i] for i in range(k))
    denom = max(1e-4, forgetting_factor + e_p_e)

    # 3. Kalman gain vector k_t = (P * e) / denom
    gain = [val / denom for val in p_e]

    # 4. Prediction residual (0 - w^T e_vec = y_true - w^T x_vec)
    pred_err = -sum(weights[i] * e_vec[i] for i in range(k))

    # 5. Update weights & project onto probability simplex (w_i >= 0.03, sum = 1)
    raw_w = [weights[i] + gain[i] * pred_err for i in range(k)]
    new_weights = _project_capped_simplex(raw_w, min_w=0.03)

    # 6. Update inverse covariance matrix P_t = (1/lambda) * (P - k_t * (P*e)^T) + ridge_diag * I
    inv_lambda = 1.0 / forgetting_factor
    new_cov = [0.0] * (k * k)
    for i in range(k):
        row_off = i * k
        g_i = gain[i]
        for j in range(k):
            val = inv_lambda * (cov_flat[row_off + j] - g_i * p_e[j])
            if i == j:
                # Diagonal stabilization + cap trace so P never explodes or collapses
                val = max(0.02, min(2.5, val + ridge_diag))
            else:
                val = max(-1.5, min(1.5, val))
            new_cov[row_off + j] = round(val, 5)

    return new_weights, new_cov


def _init_identity_cov(k: int, diag_val: float = 0.35) -> List[float]:
    cov = [0.0] * (k * k)
    for i in range(k):
        cov[i * k + i] = diag_val
    return cov


class DatabaseManager:
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        if self.pool is not None:
            return
        try:
            self.pool = await asyncpg.create_pool(
                dsn=settings.database_url,
                min_size=2,
                max_size=10,
                command_timeout=15.0,
            )
            async with self.pool.acquire() as conn:
                await conn.execute(SCHEMA_SQL)
            print("⚡️ Connected to PostGIS Telemetry Database and verified RLS schema.")
        except Exception as e:
            print(f"⚠️ Database connection warning: {e}")
            self.pool = None

    async def disconnect(self):
        if self.pool:
            await self.pool.close()
            self.pool = None

    @staticmethod
    def make_cell_id(lat: float, lon: float) -> str:
        # ~10km grid cell resolution (0.1 degree)
        return f"{round(lat, 1):.1f}:{round(lon, 1):.1f}"

    @staticmethod
    def is_us_region(lat: float, lon: float) -> bool:
        # CONUS, Alaska, Hawaii, Puerto Rico bounding check
        in_conus = (24.0 <= lat <= 50.0) and (-125.5 <= lon <= -66.5)
        in_alaska = (51.0 <= lat <= 71.5) and (-170.0 <= lon <= -129.0)
        in_hawaii = (18.5 <= lat <= 22.5) and (-160.5 <= lon <= -154.5)
        return in_conus or in_alaska or in_hawaii

    async def register_active_cell(
        self,
        lat: float,
        lon: float,
        city_name: str,
        elevation_m: float = 0.0,
        is_coastal: bool = False,
    ) -> str:
        cell_id = self.make_cell_id(lat, lon)
        region_mode = "US_HYPERLOCAL" if self.is_us_region(lat, lon) else "GLOBAL_ENSEMBLE"

        if not self.pool:
            return cell_id

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO tracked_cells (cell_id, city_name, latitude, longitude, elevation_m, region_mode, is_coastal, request_count, last_requested_at, geom)
                VALUES ($1, $2, $3, $4, $5, $6, $7, 1, NOW(), ST_SetSRID(ST_MakePoint($4, $3), 4326))
                ON CONFLICT (cell_id) DO UPDATE SET
                    city_name = EXCLUDED.city_name,
                    elevation_m = CASE WHEN EXCLUDED.elevation_m > 0 THEN EXCLUDED.elevation_m ELSE tracked_cells.elevation_m END,
                    request_count = tracked_cells.request_count + 1,
                    last_requested_at = NOW();
                """,
                cell_id,
                city_name,
                lat,
                lon,
                elevation_m,
                region_mode,
                is_coastal,
            )

            # Seed baseline prior weights & horizon buckets if this is a newly discovered cell
            existing = await conn.fetchval(
                "SELECT COUNT(*) FROM model_accuracy_weights WHERE cell_id = $1",
                cell_id,
            )
            if existing == 0:
                if region_mode == "US_HYPERLOCAL":
                    # (model, mae_t, bias_t, brier, w_comp, w_short, w_med, w_ext, acc_pct)
                    priors = [
                        ("HRRR", 0.85, 0.0, 0.08, 0.34, 0.44, 0.32, 0.18, 96.4),
                        ("ECMWF", 0.92, -0.1, 0.09, 0.30, 0.24, 0.32, 0.42, 95.8),
                        ("GFS", 1.25, 0.2, 0.13, 0.16, 0.14, 0.16, 0.18, 92.5),
                        ("ICON", 1.15, -0.1, 0.11, 0.12, 0.10, 0.12, 0.14, 93.4),
                        ("GEM", 1.35, 0.1, 0.14, 0.08, 0.08, 0.08, 0.08, 91.2),
                    ]
                else:
                    priors = [
                        ("ECMWF", 0.88, 0.0, 0.08, 0.38, 0.36, 0.38, 0.42, 96.2),
                        ("ICON", 0.96, -0.1, 0.09, 0.28, 0.30, 0.28, 0.26, 95.1),
                        ("GFS", 1.22, 0.2, 0.12, 0.18, 0.18, 0.18, 0.18, 92.8),
                        ("GEM", 1.30, 0.0, 0.13, 0.16, 0.16, 0.16, 0.14, 91.9),
                    ]
                for m_name, mae, bias, brier, w_comp, w_s, w_m, w_e, acc in priors:
                    await conn.execute(
                        """
                        INSERT INTO model_accuracy_weights
                            (cell_id, model_name, mae_temp_c, bias_temp_c, brier_precip,
                             dynamic_weight, weight_short, weight_med, weight_ext, accuracy_pct, samples_verified)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, 12)
                        ON CONFLICT (cell_id, model_name) DO NOTHING;
                        """,
                        cell_id,
                        m_name,
                        mae,
                        bias,
                        brier,
                        w_comp,
                        w_s,
                        w_m,
                        w_e,
                        acc,
                    )

            # Ensure cell_rls_state row exists with initial diagonal covariance matrices
            rls_exists = await conn.fetchval(
                "SELECT 1 FROM cell_rls_state WHERE cell_id = $1",
                cell_id,
            )
            if not rls_exists:
                model_order = (
                    ["HRRR", "ECMWF", "GFS", "ICON", "GEM"]
                    if region_mode == "US_HYPERLOCAL"
                    else ["ECMWF", "ICON", "GFS", "GEM"]
                )
                init_cov = _init_identity_cov(len(model_order), diag_val=0.35)
                await conn.execute(
                    """
                    INSERT INTO cell_rls_state (cell_id, model_order, cov_short, cov_med, cov_ext, radar_zr_factor, rls_updates)
                    VALUES ($1, $2, $3, $4, $5, 1.0, 0)
                    ON CONFLICT (cell_id) DO NOTHING;
                    """,
                    cell_id,
                    model_order,
                    init_cov,
                    init_cov,
                    init_cov,
                )

        return cell_id

    async def get_model_weights(self, cell_id: str) -> List[Dict[str, Any]]:
        if not self.pool:
            return []
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT model_name, mae_temp_c, bias_temp_c, bias_sin_c, bias_cos_c,
                       mae_wind_kmh, bias_wind_kmh, brier_precip,
                       dynamic_weight, weight_short, weight_med, weight_ext,
                       accuracy_pct, samples_verified, updated_at
                FROM model_accuracy_weights
                WHERE cell_id = $1
                ORDER BY dynamic_weight DESC;
                """,
                cell_id,
            )
            return [dict(r) for r in rows]

    async def get_cell_rls_metadata(self, cell_id: str) -> Dict[str, Any]:
        if not self.pool:
            return {"radar_zr_factor": 1.0, "rls_updates": 0}
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT radar_zr_factor, rls_updates, updated_at FROM cell_rls_state WHERE cell_id = $1",
                cell_id,
            )
            return dict(row) if row else {"radar_zr_factor": 1.0, "rls_updates": 0}

    async def get_station_reliability_map(self, station_ids: List[str]) -> Dict[str, Dict[str, float]]:
        if not self.pool or not station_ids:
            return {}
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT station_id, reliability_score, residual_bias_c, residual_mae_c
                FROM weather_stations
                WHERE station_id = ANY($1::text[]);
                """,
                station_ids,
            )
            return {
                r["station_id"]: {
                    "reliability_score": float(r["reliability_score"] or 1.0),
                    "residual_bias_c": float(r["residual_bias_c"] or 0.0),
                    "residual_mae_c": float(r["residual_mae_c"] or 0.6),
                }
                for r in rows
            }

    async def upsert_station_and_observation(
        self,
        station_id: str,
        name: str,
        network: str,
        lat: float,
        lon: float,
        elevation_m: float,
        observed_at: datetime,
        temp_c: float,
        dewpoint_c: Optional[float],
        humidity: Optional[int],
        wind_kmh: Optional[float],
        wind_dir: Optional[int],
        pressure_hpa: Optional[float],
        qc_passed: bool,
        qc_zscore: float,
        precip_rate_mm: float = 0.0,
        spatial_residual_c: float = 0.0,
    ) -> Tuple[float, float]:
        """
        Upserts station metadata, recursively updates the station's micro-site residual bias
        and reliability score (1 / variance vs. local station consensus), and records the observation.
        Returns (updated_reliability_score, updated_residual_bias_c).
        """
        if not self.pool:
            return (1.0, 0.0)
        abs_res = abs(spatial_residual_c)
        clamped_res = max(-5.0, min(5.0, spatial_residual_c))

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO weather_stations
                    (station_id, name, network, latitude, longitude, elevation_m, residual_bias_c, residual_mae_c, last_seen_at, geom)
                VALUES ($1, $2, $3, $4, $5, $6, $8::double precision * 0.15, GREATEST(0.25, $9::double precision), NOW(), ST_SetSRID(ST_MakePoint($5, $4), 4326))
                ON CONFLICT (station_id) DO UPDATE SET
                    last_seen_at = NOW(),
                    outlier_strikes = CASE WHEN $7 = FALSE THEN weather_stations.outlier_strikes + 1 ELSE weather_stations.outlier_strikes END,
                    residual_bias_c = CASE
                        WHEN $7 = TRUE THEN GREATEST(-3.5, LEAST(3.5, 0.90 * weather_stations.residual_bias_c + 0.10 * $8::double precision))
                        ELSE weather_stations.residual_bias_c
                    END,
                    residual_mae_c = GREATEST(0.20, LEAST(4.0, 0.88 * weather_stations.residual_mae_c + 0.12 * $9::double precision)),
                    reliability_score = CASE
                        WHEN $7 = FALSE THEN GREATEST(0.15, weather_stations.reliability_score - 0.12)
                        ELSE GREATEST(0.20, LEAST(1.0, 1.0 / (1.0 + 0.35 * POW(0.88 * weather_stations.residual_mae_c + 0.12 * $9::double precision, 2))))
                    END
                RETURNING reliability_score, residual_bias_c;
                """,
                station_id,
                name,
                network,
                lat,
                lon,
                elevation_m,
                qc_passed,
                clamped_res,
                abs_res,
            )
            await conn.execute(
                """
                INSERT INTO station_observations
                    (station_id, observed_at, temperature_c, dewpoint_c, humidity, wind_speed_kmh, wind_dir, pressure_hpa, precip_rate_mm, qc_passed, qc_zscore, spatial_residual_c)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (station_id, observed_at) DO UPDATE SET
                    precip_rate_mm = GREATEST(station_observations.precip_rate_mm, EXCLUDED.precip_rate_mm),
                    pressure_hpa = COALESCE(EXCLUDED.pressure_hpa, station_observations.pressure_hpa),
                    spatial_residual_c = EXCLUDED.spatial_residual_c;
                """,
                station_id,
                observed_at,
                temp_c,
                dewpoint_c,
                humidity,
                wind_kmh,
                wind_dir,
                pressure_hpa,
                float(precip_rate_mm or 0.0),
                qc_passed,
                qc_zscore,
                clamped_res,
            )
            if row:
                return (round(float(row["reliability_score"]), 3), round(float(row["residual_bias_c"]), 2))
            return (1.0, 0.0)

    async def save_forecast_snapshots_and_verify(
        self,
        cell_id: str,
        model_forecasts: Dict[str, List[Dict[str, Any]]],
        ground_truth_temp_c: Optional[float],
        ground_truth_precip_active: Optional[bool] = None,
        ground_truth_wind_kmh: Optional[float] = None,
        ground_truth_precip_mmhr: Optional[float] = None,
        radar_raw_mmhr: Optional[float] = None,
    ):
        """
        Recursive Grading & Constrained RLS Weighting Engine:
        1. Grades past unverified snapshots against ground-truth station observations (Temp, Wind, Precip Brier).
        2. Updates 3-parameter Diurnal Harmonic Bias Regression (beta0, beta_sin, beta_cos) per model.
        3. Runs O(K^2) Constrained Recursive Least Squares (RLS) covariance updates across 3 horizon buckets:
           - short (1-6h lead), med (7-24h lead), ext (25h+ lead).
        4. Recursively calibrates the cell's Radar Z-R / Virga attenuation factor when radar & station gauges co-exist.
        5. Stores future forecast snapshots for 1h, 3h, 6h, 12h, 24h, 48h lead times and prunes verified snapshots >14d old.
        """
        if not self.pool:
            return
        now = datetime.now(timezone.utc)
        current_hour = now.replace(minute=0, second=0, microsecond=0)
        obs_wet = 1.0 if ground_truth_precip_active else 0.0

        async with self.pool.acquire() as conn:
            # Load current model weights & diurnal bias harmonics for this cell
            w_rows = await conn.fetch(
                """
                SELECT model_name, mae_temp_c, bias_temp_c, bias_sin_c, bias_cos_c,
                       mae_wind_kmh, bias_wind_kmh, brier_precip,
                       dynamic_weight, weight_short, weight_med, weight_ext, samples_verified
                FROM model_accuracy_weights
                WHERE cell_id = $1;
                """,
                cell_id,
            )
            model_state: Dict[str, Dict[str, Any]] = {r["model_name"]: dict(r) for r in w_rows}

            rls_row = await conn.fetchrow(
                "SELECT model_order, cov_short, cov_med, cov_ext, radar_zr_factor, rls_updates FROM cell_rls_state WHERE cell_id = $1",
                cell_id,
            )

            if ground_truth_temp_c is not None and model_state:
                # Step 1: Fetch unverified snapshots targeting the current verification window
                unverified = await conn.fetch(
                    """
                    SELECT id, model_name, target_hour, lead_hours, predicted_temp_c, predicted_precip_prob, predicted_wind_kmh, wmo_code
                    FROM forecast_snapshots
                    WHERE cell_id = $1
                      AND verified = FALSE
                      AND target_hour <= $2::timestamptz
                      AND target_hour >= $2::timestamptz - INTERVAL '3 hours';
                    """,
                    cell_id,
                    current_hour,
                )

                # Group verified predictions by lead_hours bucket so RLS can grade the full K-model vector together
                bucket_vectors: Dict[str, Dict[str, float]] = {"short": {}, "med": {}, "ext": {}}

                for row in unverified:
                    m_name = row["model_name"]
                    pred_t = float(row["predicted_temp_c"])
                    pred_w = float(row["predicted_wind_kmh"] or 0.0)
                    pred_prob = min(1.0, max(0.0, float(row["predicted_precip_prob"] or 0) / 100.0))
                    if int(row["wmo_code"] or 0) >= 51:
                        pred_prob = max(pred_prob, 0.75)

                    abs_err = abs(pred_t - ground_truth_temp_c)
                    signed_bias = pred_t - ground_truth_temp_c
                    wind_err = (
                        abs(pred_w - ground_truth_wind_kmh)
                        if ground_truth_wind_kmh is not None
                        else None
                    )
                    brier_err = (pred_prob - obs_wet) ** 2

                    await conn.execute(
                        """
                        UPDATE forecast_snapshots
                        SET verified = TRUE,
                            actual_temp_c = $2,
                            abs_error_c = $3,
                            signed_bias_c = $4,
                            actual_wind_kmh = $5,
                            wind_abs_error_kmh = $6,
                            brier_error = $7
                        WHERE id = $1;
                        """,
                        row["id"],
                        ground_truth_temp_c,
                        abs_err,
                        signed_bias,
                        ground_truth_wind_kmh,
                        wind_err,
                        brier_err,
                    )

                    if m_name in model_state:
                        st = model_state[m_name]
                        t_hour = row["target_hour"].hour
                        # Update 3-parameter diurnal harmonic bias regression
                        b0, b_sin, b_cos = _harmonic_bias_rls_step(
                            float(st["bias_temp_c"]),
                            float(st["bias_sin_c"]),
                            float(st["bias_cos_c"]),
                            t_hour,
                            signed_bias,
                        )
                        st["bias_temp_c"] = b0
                        st["bias_sin_c"] = b_sin
                        st["bias_cos_c"] = b_cos
                        st["mae_temp_c"] = max(0.25, 0.86 * float(st["mae_temp_c"]) + 0.14 * abs_err)
                        st["brier_precip"] = max(0.02, 0.88 * float(st["brier_precip"]) + 0.12 * brier_err)
                        if wind_err is not None and ground_truth_wind_kmh is not None:
                            st["mae_wind_kmh"] = max(0.5, 0.88 * float(st["mae_wind_kmh"]) + 0.12 * wind_err)
                            st["bias_wind_kmh"] = 0.88 * float(st["bias_wind_kmh"]) + 0.12 * (pred_w - ground_truth_wind_kmh)
                        st["samples_verified"] = int(st["samples_verified"]) + 1

                        # Record bias-corrected prediction for RLS horizon bucket
                        lead_h = int(row["lead_hours"])
                        b_corr_pred = pred_t - eval_diurnal_bias(b0, b_sin, b_cos, t_hour)
                        if lead_h <= 6:
                            bucket_vectors["short"][m_name] = b_corr_pred
                        elif lead_h <= 24:
                            bucket_vectors["med"][m_name] = b_corr_pred
                        else:
                            bucket_vectors["ext"][m_name] = b_corr_pred

                # Step 2: Also grade current-hour (0h) live stream against ground truth for immediate short-range responsiveness
                now_hour_utc = current_hour.hour
                for m_name, hours in model_forecasts.items():
                    if not hours or m_name not in model_state:
                        continue
                    st = model_state[m_name]
                    pred_now = float(hours[0]["temp_c"])
                    pred_w_now = float(hours[0].get("wind_kmh", 0.0))
                    abs_err = abs(pred_now - ground_truth_temp_c)
                    signed_bias = pred_now - ground_truth_temp_c
                    pred_prob = min(1.0, max(0.0, float(hours[0].get("precip_prob", 0)) / 100.0))
                    if int(hours[0].get("wmo_code", 0)) >= 51:
                        pred_prob = max(pred_prob, 0.75)
                    brier_err = (pred_prob - obs_wet) ** 2

                    b0, b_sin, b_cos = _harmonic_bias_rls_step(
                        float(st["bias_temp_c"]),
                        float(st["bias_sin_c"]),
                        float(st["bias_cos_c"]),
                        now_hour_utc,
                        signed_bias,
                        lr=0.06,
                    )
                    st["bias_temp_c"] = b0
                    st["bias_sin_c"] = b_sin
                    st["bias_cos_c"] = b_cos
                    st["mae_temp_c"] = max(0.25, 0.90 * float(st["mae_temp_c"]) + 0.10 * abs_err)
                    st["brier_precip"] = max(0.02, 0.90 * float(st["brier_precip"]) + 0.10 * brier_err)
                    if ground_truth_wind_kmh is not None:
                        w_err = abs(pred_w_now - ground_truth_wind_kmh)
                        st["mae_wind_kmh"] = max(0.5, 0.90 * float(st["mae_wind_kmh"]) + 0.10 * w_err)
                        st["bias_wind_kmh"] = 0.90 * float(st["bias_wind_kmh"]) + 0.10 * (pred_w_now - ground_truth_wind_kmh)
                    st["samples_verified"] = int(st["samples_verified"]) + 1

                    # Ensure 'short' bucket always has a complete K-model vector from hour 0 if snapshots weren't all present
                    if m_name not in bucket_vectors["short"]:
                        bucket_vectors["short"][m_name] = pred_now - eval_diurnal_bias(b0, b_sin, b_cos, now_hour_utc)

                # Step 3: Execute Constrained Recursive Least Squares (RLS) Covariance Updates per Horizon Bucket
                if rls_row:
                    model_order = [m for m in rls_row["model_order"] if m in model_state]
                    k = len(model_order)
                    cov_short = list(rls_row["cov_short"]) if len(rls_row["cov_short"]) == k * k else _init_identity_cov(k)
                    cov_med = list(rls_row["cov_med"]) if len(rls_row["cov_med"]) == k * k else _init_identity_cov(k)
                    cov_ext = list(rls_row["cov_ext"]) if len(rls_row["cov_ext"]) == k * k else _init_identity_cov(k)
                    zr_factor = float(rls_row["radar_zr_factor"] or 1.0)
                    rls_updates = int(rls_row["rls_updates"] or 0)

                    # Short-range (1-6h) RLS update
                    if all(m in bucket_vectors["short"] for m in model_order):
                        w_s = [float(model_state[m]["weight_short"]) for m in model_order]
                        x_s = [bucket_vectors["short"][m] for m in model_order]
                        new_w_s, cov_short = _rls_ensemble_step(w_s, cov_short, x_s, ground_truth_temp_c)
                        for idx, m in enumerate(model_order):
                            model_state[m]["weight_short"] = new_w_s[idx]
                        rls_updates += 1

                    # Medium-range (7-24h) RLS update
                    if all(m in bucket_vectors["med"] for m in model_order):
                        w_m = [float(model_state[m]["weight_med"]) for m in model_order]
                        x_m = [bucket_vectors["med"][m] for m in model_order]
                        new_w_m, cov_med = _rls_ensemble_step(w_m, cov_med, x_m, ground_truth_temp_c)
                        for idx, m in enumerate(model_order):
                            model_state[m]["weight_med"] = new_w_m[idx]
                    else:
                        # Gently nudge medium weights toward short + inverse-variance skill when waiting for 12-24h snapshots to mature
                        for m in model_order:
                            model_state[m]["weight_med"] = round(
                                0.85 * float(model_state[m]["weight_med"]) + 0.15 * float(model_state[m]["weight_short"]),
                                4,
                            )

                    # Extended-range (25h+) RLS update
                    if all(m in bucket_vectors["ext"] for m in model_order):
                        w_e = [float(model_state[m]["weight_ext"]) for m in model_order]
                        x_e = [bucket_vectors["ext"][m] for m in model_order]
                        new_w_e, cov_ext = _rls_ensemble_step(w_e, cov_ext, x_e, ground_truth_temp_c)
                        for idx, m in enumerate(model_order):
                            model_state[m]["weight_ext"] = new_w_e[idx]

                    # Calibrate Radar Z-R / Virga attenuation factor when radar echoes or station precip are present
                    if radar_raw_mmhr is not None and radar_raw_mmhr >= 0.25 and ground_truth_precip_mmhr is not None:
                        ratio = max(0.35, min(1.75, (ground_truth_precip_mmhr + 0.15) / (radar_raw_mmhr + 0.15)))
                        zr_factor = round(max(0.45, min(1.50, 0.90 * zr_factor + 0.10 * ratio)), 3)

                    await conn.execute(
                        """
                        UPDATE cell_rls_state
                        SET cov_short = $2,
                            cov_med = $3,
                            cov_ext = $4,
                            radar_zr_factor = $5,
                            rls_updates = $6,
                            updated_at = NOW()
                        WHERE cell_id = $1;
                        """,
                        cell_id,
                        cov_short,
                        cov_med,
                        cov_ext,
                        zr_factor,
                        rls_updates,
                    )

                # Step 4: Combine RLS horizon weights with multi-variable skill penalty (Precip Brier + Wind MAE) & persist
                raw_comp: Dict[str, float] = {}
                for m, st in model_state.items():
                    rls_blend = (
                        0.45 * float(st["weight_short"])
                        + 0.35 * float(st["weight_med"])
                        + 0.20 * float(st["weight_ext"])
                    )
                    # Multi-variable skill modifier (rewards models with low Brier precip error & low wind MAE)
                    brier_mod = 1.0 / (1.0 + 1.8 * max(0.02, float(st["brier_precip"])))
                    wind_mod = 1.0 / (1.0 + 0.04 * max(0.5, float(st["mae_wind_kmh"])))
                    raw_comp[m] = rls_blend * brier_mod * wind_mod

                total_comp = sum(raw_comp.values()) or 1.0
                for m, st in model_state.items():
                    comp_w = round(raw_comp[m] / total_comp, 4)
                    acc_pct = round(
                        max(
                            70.0,
                            min(
                                99.4,
                                100.0
                                - (float(st["mae_temp_c"]) * 3.1)
                                - (float(st["brier_precip"]) * 7.5)
                                - (max(0.0, float(st["mae_wind_kmh"]) - 2.5) * 0.25),
                            ),
                        ),
                        1,
                    )
                    await conn.execute(
                        """
                        UPDATE model_accuracy_weights
                        SET mae_temp_c = $3,
                            bias_temp_c = $4,
                            bias_sin_c = $5,
                            bias_cos_c = $6,
                            mae_wind_kmh = $7,
                            bias_wind_kmh = $8,
                            brier_precip = $9,
                            dynamic_weight = $10,
                            weight_short = $11,
                            weight_med = $12,
                            weight_ext = $13,
                            accuracy_pct = $14,
                            samples_verified = $15,
                            updated_at = NOW()
                        WHERE cell_id = $1 AND model_name = $2;
                        """,
                        cell_id,
                        m,
                        round(float(st["mae_temp_c"]), 3),
                        round(float(st["bias_temp_c"]), 4),
                        round(float(st["bias_sin_c"]), 4),
                        round(float(st["bias_cos_c"]), 4),
                        round(float(st["mae_wind_kmh"]), 2),
                        round(float(st["bias_wind_kmh"]), 2),
                        round(float(st["brier_precip"]), 4),
                        comp_w,
                        round(float(st["weight_short"]), 4),
                        round(float(st["weight_med"]), 4),
                        round(float(st["weight_ext"]), 4),
                        acc_pct,
                        int(st["samples_verified"]),
                    )

            # Step 5: Store future forecast snapshots across short, med, and ext horizons (1, 3, 6, 12, 24, 48h)
            for model_name, hours in model_forecasts.items():
                for lead in (1, 3, 6, 12, 24, 48):
                    if lead < len(hours):
                        h = hours[lead]
                        await conn.execute(
                            """
                            INSERT INTO forecast_snapshots
                                (cell_id, model_name, issued_at, target_hour, lead_hours, predicted_temp_c, predicted_precip_prob, predicted_wind_kmh, wmo_code)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                            ON CONFLICT (cell_id, model_name, target_hour, lead_hours) DO NOTHING;
                            """,
                            cell_id,
                            model_name,
                            now,
                            h["time"],
                            lead,
                            h["temp_c"],
                            h["precip_prob"],
                            h["wind_kmh"],
                            h["wmo_code"],
                        )

            # Lightweight housekeeping: prune verified snapshots older than 14 days so disk stays compact
            await conn.execute(
                """
                DELETE FROM forecast_snapshots
                WHERE cell_id = $1
                  AND verified = TRUE
                  AND target_hour < NOW() - INTERVAL '14 days';
                """,
                cell_id,
            )

    async def record_doppler_frame(
        self,
        frame_id: str,
        frame_time: datetime,
        grid_shape: str,
        resolution_km: float,
        max_dbz: float,
        echo_coverage_pct: float,
        storm_cells_detected: int,
        raw_bytes: int,
        is_nowcast: bool = False,
        backfilled: bool = False,
    ):
        if not self.pool:
            return
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO doppler_grid_frames
                    (frame_id, frame_time, grid_shape, resolution_km, max_dbz, echo_coverage_pct, storm_cells_detected, raw_bytes, is_nowcast, backfilled, ingested_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW())
                ON CONFLICT (frame_id) DO UPDATE SET
                    max_dbz = EXCLUDED.max_dbz,
                    echo_coverage_pct = EXCLUDED.echo_coverage_pct,
                    storm_cells_detected = EXCLUDED.storm_cells_detected,
                    raw_bytes = EXCLUDED.raw_bytes,
                    ingested_at = NOW();
                """,
                frame_id,
                frame_time,
                grid_shape,
                resolution_km,
                max_dbz,
                echo_coverage_pct,
                storm_cells_detected,
                raw_bytes,
                is_nowcast,
                backfilled,
            )

    async def get_existing_doppler_frame_ids(self, since_dt: datetime) -> set[str]:
        if not self.pool:
            return set()
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT frame_id FROM doppler_grid_frames
                WHERE frame_time >= $1 AND is_nowcast = FALSE;
                """,
                since_dt,
            )
            return {str(r["frame_id"]) for r in rows}

    async def backfill_expired_snapshots(
        self, cell_id: str, past_hourly_temps: Dict[datetime, float]
    ) -> int:
        """
        Grades unverified forecast_snapshots whose target_hour passed during an offline window.
        Prioritizes real PostGIS station_observations within 45km of the cell; only falls back to
        reanalysis with reduced weight (0.04) if no physical station reading exists for that hour.
        """
        if not self.pool:
            return 0
        now = datetime.now(timezone.utc)
        graded_count = 0
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, model_name, target_hour, predicted_temp_c
                FROM forecast_snapshots
                WHERE cell_id = $1
                  AND verified = FALSE
                  AND target_hour < $2::timestamptz - INTERVAL '3 hours';
                """,
                cell_id,
                now,
            )
            if not rows:
                return 0

            # Lookup any real QC-passed station observations within 45km (0.45 deg bbox) for this cell
            station_obs_rows = await conn.fetch(
                """
                SELECT date_trunc('hour', o.observed_at) AS obs_hour,
                       AVG(o.temperature_c) AS avg_temp_c
                FROM station_observations o
                JOIN weather_stations s ON s.station_id = o.station_id
                JOIN tracked_cells c ON c.cell_id = $1
                WHERE o.qc_passed = TRUE
                  AND o.observed_at >= $2::timestamptz - INTERVAL '48 hours'
                  AND ST_DWithin(s.geom::geography, c.geom::geography, 45000)
                GROUP BY obs_hour;
                """,
                cell_id,
                now,
            )
            real_station_hourly = {
                r["obs_hour"].replace(minute=0, second=0, microsecond=0): float(r["avg_temp_c"])
                for r in station_obs_rows
                if r["obs_hour"] is not None and r["avg_temp_c"] is not None
            }

            for row in rows:
                th = row["target_hour"].replace(minute=0, second=0, microsecond=0)
                is_real_obs = th in real_station_hourly
                truth_c = real_station_hourly.get(th) or past_hourly_temps.get(th)
                if truth_c is not None:
                    pred_c = float(row["predicted_temp_c"])
                    abs_err = abs(pred_c - truth_c)
                    signed_bias = pred_c - truth_c
                    alpha = 0.10 if is_real_obs else 0.03
                    await conn.execute(
                        """
                        UPDATE forecast_snapshots
                        SET verified = TRUE,
                            actual_temp_c = $2,
                            abs_error_c = $3,
                            signed_bias_c = $4
                        WHERE id = $1;
                        """,
                        row["id"],
                        truth_c,
                        abs_err,
                        signed_bias,
                    )
                    await conn.execute(
                        """
                        UPDATE model_accuracy_weights
                        SET mae_temp_c = GREATEST(0.25, (1.0 - $5::double precision) * mae_temp_c + $5::double precision * $3::double precision),
                            bias_temp_c = (1.0 - $5::double precision) * bias_temp_c + $5::double precision * $4::double precision,
                            samples_verified = samples_verified + 1,
                            updated_at = NOW()
                        WHERE cell_id = $1 AND model_name = $2;
                        """,
                        cell_id,
                        row["model_name"],
                        abs_err,
                        signed_bias,
                        alpha,
                    )
                    graded_count += 1
        return graded_count

    async def get_database_stats(self) -> Dict[str, Any]:
        if not self.pool:
            return {"connected": False}
        async with self.pool.acquire() as conn:
            cells_count = await conn.fetchval("SELECT COUNT(*) FROM tracked_cells;")
            stations_count = await conn.fetchval("SELECT COUNT(*) FROM weather_stations;")
            obs_count = await conn.fetchval("SELECT COUNT(*) FROM station_observations;")
            snapshots_count = await conn.fetchval("SELECT COUNT(*) FROM forecast_snapshots;")
            verified_count = await conn.fetchval("SELECT COUNT(*) FROM forecast_snapshots WHERE verified = TRUE;")
            rls_updates_total = await conn.fetchval("SELECT COALESCE(SUM(rls_updates), 0) FROM cell_rls_state;")
            doppler_count = await conn.fetchval("SELECT COUNT(*) FROM doppler_grid_frames;")
            backfilled_count = await conn.fetchval("SELECT COUNT(*) FROM doppler_grid_frames WHERE backfilled = TRUE;")
            db_size = await conn.fetchval("SELECT pg_size_pretty(pg_database_size(current_database()));")
            postgis_ver = await conn.fetchval("SELECT PostGIS_Lib_Version();")

            cells = await conn.fetch(
                """
                SELECT c.cell_id, c.city_name, c.region_mode, c.elevation_m, c.request_count, c.last_requested_at,
                       COALESCE(r.rls_updates, 0) AS rls_updates, COALESCE(r.radar_zr_factor, 1.0) AS radar_zr_factor
                FROM tracked_cells c
                LEFT JOIN cell_rls_state r ON r.cell_id = c.cell_id
                ORDER BY c.last_requested_at DESC
                LIMIT 10;
                """
            )
            latest_doppler = await conn.fetchrow(
                """
                SELECT frame_id, frame_time, grid_shape, max_dbz, echo_coverage_pct, storm_cells_detected
                FROM doppler_grid_frames
                WHERE is_nowcast = FALSE
                ORDER BY frame_time DESC
                LIMIT 1;
                """
            )
            return {
                "connected": True,
                "engine": f"PostgreSQL 16 + PostGIS {postgis_ver} (Constrained RLS + Harmonic MOS)",
                "database_size": db_size,
                "tracked_cells_count": cells_count,
                "stations_indexed": stations_count,
                "station_observations_stored": obs_count,
                "forecast_snapshots_stored": snapshots_count,
                "verified_predictions_count": verified_count,
                "rls_covariance_updates": int(rls_updates_total or 0),
                "doppler_frames_indexed": doppler_count,
                "doppler_backfilled_frames": backfilled_count,
                "latest_doppler_frame": dict(latest_doppler) if latest_doppler else None,
                "active_cells": [dict(c) for c in cells],
            }


db_manager = DatabaseManager()
