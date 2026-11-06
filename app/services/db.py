import asyncpg
import math
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
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
    outlier_strikes INTEGER DEFAULT 0,
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    geom GEOMETRY(Point, 4326)
);

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
    UNIQUE (station_id, observed_at)
);

CREATE INDEX IF NOT EXISTS idx_station_obs_time ON station_observations (observed_at DESC);
CREATE INDEX IF NOT EXISTS idx_station_obs_station ON station_observations (station_id, observed_at DESC);

-- 4. Multi-Model Forecast Snapshots (For Verification & Bias Grading)
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
    UNIQUE (cell_id, model_name, target_hour, lead_hours)
);

CREATE INDEX IF NOT EXISTS idx_forecast_snapshots_verify
    ON forecast_snapshots (cell_id, verified, target_hour);

-- 5. Dynamic Accuracy Weights & MOS Bias Correction Ledger
CREATE TABLE IF NOT EXISTS model_accuracy_weights (
    cell_id TEXT NOT NULL REFERENCES tracked_cells(cell_id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    mae_temp_c DOUBLE PRECISION NOT NULL DEFAULT 1.2,
    bias_temp_c DOUBLE PRECISION NOT NULL DEFAULT 0.0, -- Positive = model runs hot, Negative = runs cold
    brier_precip DOUBLE PRECISION NOT NULL DEFAULT 0.12,
    dynamic_weight DOUBLE PRECISION NOT NULL DEFAULT 0.20,
    accuracy_pct DOUBLE PRECISION NOT NULL DEFAULT 92.0,
    samples_verified INTEGER NOT NULL DEFAULT 1,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (cell_id, model_name)
);

-- 6. Züs CONUS 1km Doppler Radar Grid Snapshots Ledger
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
            print("⚡️ Connected to PostGIS Telemetry Database and verified schema.")
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

            # Seed baseline prior weights if this is a newly discovered cell
            existing = await conn.fetchval(
                "SELECT COUNT(*) FROM model_accuracy_weights WHERE cell_id = $1",
                cell_id,
            )
            if existing == 0:
                if region_mode == "US_HYPERLOCAL":
                    priors = [
                        ("HRRR", 0.85, 0.0, 0.08, 0.34, 96.4),
                        ("ECMWF", 0.92, -0.1, 0.09, 0.30, 95.8),
                        ("GFS", 1.25, 0.2, 0.13, 0.16, 92.5),
                        ("ICON", 1.15, -0.1, 0.11, 0.12, 93.4),
                        ("GEM", 1.35, 0.1, 0.14, 0.08, 91.2),
                    ]
                else:
                    priors = [
                        ("ECMWF", 0.88, 0.0, 0.08, 0.38, 96.2),
                        ("ICON", 0.96, -0.1, 0.09, 0.28, 95.1),
                        ("GFS", 1.22, 0.2, 0.12, 0.18, 92.8),
                        ("GEM", 1.30, 0.0, 0.13, 0.16, 91.9),
                    ]
                for model_name, mae, bias, brier, weight, acc in priors:
                    await conn.execute(
                        """
                        INSERT INTO model_accuracy_weights
                            (cell_id, model_name, mae_temp_c, bias_temp_c, brier_precip, dynamic_weight, accuracy_pct, samples_verified)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, 12)
                        ON CONFLICT (cell_id, model_name) DO NOTHING;
                        """,
                        cell_id,
                        model_name,
                        mae,
                        bias,
                        brier,
                        weight,
                        acc,
                    )

        return cell_id

    async def get_model_weights(self, cell_id: str) -> List[Dict[str, Any]]:
        if not self.pool:
            return []
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT model_name, mae_temp_c, bias_temp_c, brier_precip, dynamic_weight, accuracy_pct, samples_verified, updated_at
                FROM model_accuracy_weights
                WHERE cell_id = $1
                ORDER BY dynamic_weight DESC;
                """,
                cell_id,
            )
            return [dict(r) for r in rows]

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
    ):
        if not self.pool:
            return
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO weather_stations (station_id, name, network, latitude, longitude, elevation_m, last_seen_at, geom)
                VALUES ($1, $2, $3, $4, $5, $6, NOW(), ST_SetSRID(ST_MakePoint($5, $4), 4326))
                ON CONFLICT (station_id) DO UPDATE SET
                    last_seen_at = NOW(),
                    outlier_strikes = CASE WHEN $7 = FALSE THEN weather_stations.outlier_strikes + 1 ELSE weather_stations.outlier_strikes END,
                    reliability_score = CASE
                        WHEN $7 = FALSE THEN GREATEST(0.1, weather_stations.reliability_score - 0.1)
                        ELSE LEAST(1.0, weather_stations.reliability_score + 0.02)
                    END;
                """,
                station_id,
                name,
                network,
                lat,
                lon,
                elevation_m,
                qc_passed,
            )
            await conn.execute(
                """
                INSERT INTO station_observations
                    (station_id, observed_at, temperature_c, dewpoint_c, humidity, wind_speed_kmh, wind_dir, pressure_hpa, precip_rate_mm, qc_passed, qc_zscore)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (station_id, observed_at) DO UPDATE SET
                    precip_rate_mm = GREATEST(station_observations.precip_rate_mm, EXCLUDED.precip_rate_mm),
                    pressure_hpa = COALESCE(EXCLUDED.pressure_hpa, station_observations.pressure_hpa);
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
            )

    async def save_forecast_snapshots_and_verify(
        self,
        cell_id: str,
        model_forecasts: Dict[str, List[Dict[str, Any]]],
        ground_truth_temp_c: Optional[float],
        ground_truth_precip_active: Optional[bool] = None,
    ):
        """
        1. Grades past unverified snapshots for the current hour against ground_truth_temp_c.
        2. Updates dynamic weights (Inverse-Variance 1/MAE^2) and MOS bias offsets (exponential moving average).
        3. Stores new snapshots for 0h, 1h, 3h, 6h, 12h, 24h lead times.
        """
        if not self.pool:
            return
        now = datetime.now(timezone.utc)
        current_hour = now.replace(minute=0, second=0, microsecond=0)

        async with self.pool.acquire() as conn:
            # Step 1: Verify any predictions targeting the recent window against ground truth
            if ground_truth_temp_c is not None:
                unverified = await conn.fetch(
                    """
                    SELECT id, model_name, predicted_temp_c
                    FROM forecast_snapshots
                    WHERE cell_id = $1
                      AND verified = FALSE
                      AND target_hour <= $2
                      AND target_hour >= $2 - INTERVAL '3 hours';
                    """,
                    cell_id,
                    current_hour,
                )
                for row in unverified:
                    pred = row["predicted_temp_c"]
                    abs_err = abs(pred - ground_truth_temp_c)
                    signed_bias = pred - ground_truth_temp_c
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
                        ground_truth_temp_c,
                        abs_err,
                        signed_bias,
                    )

                    # Exponentially decaying MAE & MOS bias update (alpha = 0.15)
                    await conn.execute(
                        """
                        UPDATE model_accuracy_weights
                        SET mae_temp_c = GREATEST(0.25, 0.85 * mae_temp_c + 0.15 * $3),
                            bias_temp_c = 0.85 * bias_temp_c + 0.15 * $4,
                            samples_verified = samples_verified + 1,
                            updated_at = NOW()
                        WHERE cell_id = $1 AND model_name = $2;
                        """,
                        cell_id,
                        row["model_name"],
                        abs_err,
                        signed_bias,
                    )

                # Step 2: Also compare current hour model values directly to ground truth for immediate calibration
                obs_wet = 1.0 if ground_truth_precip_active else 0.0
                for model_name, hours in model_forecasts.items():
                    if not hours:
                        continue
                    pred_now = hours[0]["temp_c"]
                    abs_err = abs(pred_now - ground_truth_temp_c)
                    signed_bias = pred_now - ground_truth_temp_c
                    # Brier skill score component for precipitation & wet WMO code (51+)
                    pred_prob = min(1.0, max(0.0, float(hours[0].get("precip_prob", 0)) / 100.0))
                    if int(hours[0].get("wmo_code", 0)) >= 51:
                        pred_prob = max(pred_prob, 0.75)
                    brier_err = (pred_prob - obs_wet) ** 2

                    await conn.execute(
                        """
                        UPDATE model_accuracy_weights
                        SET mae_temp_c = GREATEST(0.25, 0.90 * mae_temp_c + 0.10 * $3),
                            bias_temp_c = 0.90 * bias_temp_c + 0.10 * $4,
                            brier_precip = GREATEST(0.02, 0.88 * brier_precip + 0.12 * $5),
                            samples_verified = samples_verified + 1,
                            updated_at = NOW()
                        WHERE cell_id = $1 AND model_name = $2;
                        """,
                        cell_id,
                        model_name,
                        abs_err,
                        signed_bias,
                        brier_err,
                    )

                # Step 3: Normalize Inverse-Variance dynamic weights across all models in this cell
                weight_rows = await conn.fetch(
                    "SELECT model_name, mae_temp_c, brier_precip FROM model_accuracy_weights WHERE cell_id = $1",
                    cell_id,
                )
                if weight_rows:
                    inv_vars = {
                        r["model_name"]: 1.0
                        / ((max(0.3, r["mae_temp_c"]) ** 2) * (1.0 + max(0.02, float(r["brier_precip"] or 0.1)) * 2.5))
                        for r in weight_rows
                    }
                    total_inv = sum(inv_vars.values()) or 1.0
                    for r in weight_rows:
                        m = r["model_name"]
                        w = round(inv_vars[m] / total_inv, 4)
                        # Accuracy percentage mapped from MAE + Brier skill
                        acc_pct = round(
                            max(
                                70.0,
                                min(
                                    99.4,
                                    100.0 - (r["mae_temp_c"] * 3.4) - (float(r["brier_precip"] or 0.08) * 8.0),
                                ),
                            ),
                            1,
                        )
                        await conn.execute(
                            """
                            UPDATE model_accuracy_weights
                            SET dynamic_weight = $3, accuracy_pct = $4
                            WHERE cell_id = $1 AND model_name = $2;
                            """,
                            cell_id,
                            m,
                            w,
                            acc_pct,
                        )

            # Step 4: Store future forecast snapshots (lead hours: 1, 3, 6, 12, 24)
            for model_name, hours in model_forecasts.items():
                for lead in (1, 3, 6, 12, 24):
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
        Grades any unverified forecast_snapshots whose target_hour passed during an offline outage
        using historical hourly reanalysis/observations.
        """
        if not self.pool or not past_hourly_temps:
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
                  AND target_hour < $2;
                """,
                cell_id,
                now,
            )
            for row in rows:
                th = row["target_hour"].replace(minute=0, second=0, microsecond=0)
                if th in past_hourly_temps:
                    truth_c = past_hourly_temps[th]
                    pred_c = float(row["predicted_temp_c"])
                    abs_err = abs(pred_c - truth_c)
                    signed_bias = pred_c - truth_c
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
                        SET mae_temp_c = GREATEST(0.25, 0.88 * mae_temp_c + 0.12 * $3),
                            bias_temp_c = 0.88 * bias_temp_c + 0.12 * $4,
                            samples_verified = samples_verified + 1,
                            updated_at = NOW()
                        WHERE cell_id = $1 AND model_name = $2;
                        """,
                        cell_id,
                        row["model_name"],
                        abs_err,
                        signed_bias,
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
            doppler_count = await conn.fetchval("SELECT COUNT(*) FROM doppler_grid_frames;")
            backfilled_count = await conn.fetchval("SELECT COUNT(*) FROM doppler_grid_frames WHERE backfilled = TRUE;")
            db_size = await conn.fetchval("SELECT pg_size_pretty(pg_database_size(current_database()));")
            postgis_ver = await conn.fetchval("SELECT PostGIS_Lib_Version();")

            cells = await conn.fetch(
                """
                SELECT cell_id, city_name, region_mode, elevation_m, request_count, last_requested_at
                FROM tracked_cells
                ORDER BY last_requested_at DESC
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
                "engine": f"PostgreSQL 16 + PostGIS {postgis_ver} (NVMe SSD)",
                "database_size": db_size,
                "tracked_cells_count": cells_count,
                "stations_indexed": stations_count,
                "station_observations_stored": obs_count,
                "forecast_snapshots_stored": snapshots_count,
                "verified_predictions_count": verified_count,
                "doppler_frames_indexed": doppler_count,
                "doppler_backfilled_frames": backfilled_count,
                "latest_doppler_frame": dict(latest_doppler) if latest_doppler else None,
                "active_cells": [dict(c) for c in cells],
            }


db_manager = DatabaseManager()
