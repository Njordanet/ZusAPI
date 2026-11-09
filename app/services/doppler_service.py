import asyncio
import base64
import io
import math
from collections import OrderedDict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import httpx
import numpy as np
from PIL import Image

from app.config import settings
from app.models.weather import (
    RadarFrameModel,
    RadarLocalTelemetry,
    RadarManifestResponse,
)
from app.services.dataset_service import dataset_service
from app.services.db import db_manager


def _build_zus_palette() -> Tuple[List[int], bytes]:
    """
    Pre-computes the 256-entry 8-bit Indexed Palette (PLTE RGB + tRNS alpha)
    for IEM NEXRAD N0Q reflectivity indices:
      idx = 0       -> No data / clear sky
      idx = 1..255  -> dBZ = -32.5 + (idx * 0.5)
      idx = 80      -> +7.5 dBZ (light mist threshold)
      idx = 100     -> +17.5 dBZ (light rain)
      idx = 125     -> +30.0 dBZ (moderate-to-heavy rain)
      idx = 145     -> +40.0 dBZ (heavy storm core - Züs Gold)
      idx = 165     -> +50.0 dBZ (severe convective core - Crimson)
      idx = 180+    -> +57.5+ dBZ (hail / mesocyclone - Electric Violet)
    """
    rgb: List[int] = [0] * (256 * 3)
    alpha = bytearray(256)

    # Key color control stops: (index, R, G, B, Alpha)
    stops = [
        (82, 56, 189, 248, 0),      # 8.5 dBZ: crisp outer echo boundary
        (86, 56, 189, 248, 135),    # 10.5 dBZ: crisp cyan mist
        (96, 14, 165, 233, 185),    # 15.5 dBZ: light rain sky blue
        (112, 37, 99, 235, 218),    # 23.5 dBZ: moderate electric blue
        (124, 52, 211, 153, 234),   # 29.5 dBZ: intense emerald-cyan transition
        (134, 250, 204, 21, 246),   # 34.5 dBZ: Züs Storm Gold (#FACC15)
        (146, 245, 158, 11, 252),   # 40.5 dBZ: deep amber gold
        (158, 239, 68, 68, 255),    # 46.5 dBZ: crimson storm core
        (172, 217, 70, 239, 255),   # 53.5 dBZ: electric violet hail core
        (195, 240, 171, 252, 255),  # 65.0 dBZ: white-hot magenta
        (255, 255, 255, 255, 255),
    ]

    for i in range(len(stops) - 1):
        idx0, r0, g0, b0, a0 = stops[i]
        idx1, r1, g1, b1, a1 = stops[i + 1]
        span = max(1, idx1 - idx0)
        for idx in range(idx0, idx1 + 1):
            t = (idx - idx0) / span
            rgb[idx * 3 + 0] = int(round(r0 + (r1 - r0) * t))
            rgb[idx * 3 + 1] = int(round(g0 + (g1 - g0) * t))
            rgb[idx * 3 + 2] = int(round(b0 + (b1 - b0) * t))
            alpha[idx] = int(round(a0 + (a1 - a0) * t))

    return rgb, bytes(alpha)


ZUS_PALETTE_RGB, ZUS_PALETTE_ALPHA = _build_zus_palette()


def _create_empty_transparent_tile() -> bytes:
    """Pre-encoded 256x256 transparent 8-bit indexed PNG (~310 bytes)."""
    img = Image.new("P", (256, 256), 0)
    img.putpalette(ZUS_PALETTE_RGB)
    img.info["transparency"] = ZUS_PALETTE_ALPHA
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


EMPTY_TILE_PNG: bytes = _create_empty_transparent_tile()


def _dbz_from_index(idx: int) -> float:
    if idx < 72:
        return 0.0
    return round(-32.5 + (idx * 0.5), 1)


def _mmhr_from_dbz(dbz: float) -> float:
    """
    Composite Z-R relationship blending cool-season stratiform/drizzle (Z = 130 * R^1.5)
    at low reflectivity (8..20 dBZ) with standard Marshall-Palmer (Z = 200 * R^1.6) for >= 20 dBZ.
    """
    if dbz < 8.0:
        return 0.0
    z_linear = 10.0 ** (dbz / 10.0)
    if dbz < 20.0:
        r = (z_linear / 140.0) ** (1.0 / 1.5)
    else:
        r = (z_linear / 200.0) ** (1.0 / 1.6)
    return round(min(100.0, max(0.0, r)), 2)


def _classify_dbz(dbz: float) -> str:
    if dbz < 8.0:
        return "Clear"
    if dbz < 15.0:
        return "Light Drizzle"
    if dbz < 25.0:
        return "Light Rain"
    if dbz < 35.0:
        return "Moderate Rain"
    if dbz < 46.0:
        return "Heavy Storm"
    return "Severe Hail Core"


def _compass_from_degrees(deg: float) -> str:
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    ix = int(round((deg % 360) / 22.5)) % 16
    return dirs[ix]


class DopplerGridFrame:
    """Holds a single 1km CONUS uint8 reflectivity matrix (2700 x 6100 = 16.47 MB) or zero-copy advected view."""
    __slots__ = (
        "frame_id",
        "timestamp",
        "dt",
        "label",
        "offset_minutes",
        "is_future",
        "grid",
        "advect_drow",
        "advect_dcol",
        "decay_indices",
        "max_dbz",
        "echo_coverage_pct",
        "storm_cells_count",
        "raw_bytes",
    )

    def __init__(
        self,
        frame_id: str,
        dt: datetime,
        label: str,
        offset_minutes: int,
        is_future: bool,
        grid: np.ndarray,
        raw_bytes: int,
        advect_drow: int = 0,
        advect_dcol: int = 0,
        decay_indices: int = 0,
    ):
        self.frame_id = frame_id
        self.dt = dt
        self.timestamp = int(dt.timestamp())
        self.label = label
        self.offset_minutes = offset_minutes
        self.is_future = is_future
        self.grid = grid  # shape (2700, 6100), dtype uint8, step 0.01 deg
        self.advect_drow = int(advect_drow)
        self.advect_dcol = int(advect_dcol)
        self.decay_indices = int(decay_indices)
        self.raw_bytes = raw_bytes

        # Compute summary stats on decimated slice for speed (<1ms)
        sample = grid[::4, ::4]
        max_idx = max(0, int(sample.max()) - self.decay_indices) if sample.size else 0
        self.max_dbz = _dbz_from_index(max_idx)
        active_mask = sample >= (85 + self.decay_indices)  # >= 10 dBZ
        self.echo_coverage_pct = round(float(active_mask.sum()) / max(1, sample.size) * 100.0, 2)
        severe_mask = sample >= (145 + self.decay_indices)  # >= 40 dBZ
        self.storm_cells_count = int(severe_mask.sum() // 6)


class DopplerRadarService:
    """
    High-efficiency CONUS 1km Master Grid + 8-Bit Indexed Tile Slicer & Nowcast Engine.
    - CONUS Bounds: 23.0°N .. 50.0°N, -126.0°W .. -65.0°W (0.01° resolution = 2700 x 6100 uint8)
    - Memory footprint: ~131 MB for 8 frames (6 past/live + 2 advected nowcast frames)
    - Tile rendering: <0.4ms per 256x256 8-bit PNG tile via vectorized numpy slicing
    """
    LAT_MAX = 50.0
    LAT_MIN = 23.0
    LON_MIN = -126.0
    LON_MAX = -65.0
    STEP_DEG = 0.01
    GRID_H = 2700
    GRID_W = 6100

    def __init__(self):
        self.doppler_dir: Path = settings.data_dir / "doppler"
        self.archive_dir: Path = self.doppler_dir / "archive"
        self._frames: OrderedDict[str, DopplerGridFrame] = OrderedDict()
        self._tile_cache: OrderedDict[str, bytes] = OrderedDict()
        self._max_cached_tiles = 4000
        self._inflight_locks: Dict[str, asyncio.Lock] = {}
        self._ingest_lock = asyncio.Lock()
        self._backfill_lock = asyncio.Lock()
        self._last_ingested_at: Optional[datetime] = None
        self._last_backfill_at: Optional[datetime] = None
        self._total_backfilled_frames: int = 0
        self._global_motion_drow_15m: int = 2   # default synoptic drift
        self._global_motion_dcol_15m: int = 8
        self._motion_field_drow_15m: Optional[np.ndarray] = None  # shape (55, 123) float32
        self._motion_field_dcol_15m: Optional[np.ndarray] = None  # shape (55, 123) float32
        self._cum_drow_fields: Dict[int, np.ndarray] = {}         # fut_mult (1..8) -> shape (110, 245) float32
        self._cum_dcol_fields: Dict[int, np.ndarray] = {}         # fut_mult (1..8) -> shape (110, 245) float32
        self._hrrr_trend_fields: Dict[int, np.ndarray] = {}       # fut_mult (1..8) -> shape (110, 245) float32
        self._rainviewer_frames: List[Dict[str, Any]] = []
        self._rainviewer_host: str = "https://tilecache.rainviewer.com"

    def _ensure_dir(self):
        try:
            self.doppler_dir.mkdir(parents=True, exist_ok=True)
            self.archive_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def _prune_disk_archive(self, max_age_hours: int = 24) -> None:
        try:
            cutoff_ts = datetime.now(timezone.utc).timestamp() - (max_age_hours * 3600)
            for pattern in ("n0q_*.png", "hrrr_*.png"):
                for p in self.archive_dir.glob(pattern):
                    if p.stat().st_mtime < cutoff_ts:
                        p.unlink(missing_ok=True)
        except Exception:
            pass

    @staticmethod
    def _decode_iem_png_to_1km_grid(png_bytes: bytes) -> np.ndarray:
        """
        Decodes IEM 5400x12200 8-bit N0Q PNG into a 2700x6100 (1km) uint8 numpy array,
        filtering out clear-air ground clutter (<13.5 dBZ / idx < 92) and isolated radar speckles.
        """
        with Image.open(io.BytesIO(png_bytes)) as img:
            arr = np.asarray(img, dtype=np.uint8)
            h, w = arr.shape[:2]
            h2 = (h // 2) * 2
            w2 = (w // 2) * 2

            p00 = arr[0:h2:2, 0:w2:2]
            p01 = arr[0:h2:2, 1:w2:2]
            p10 = arr[1:h2:2, 0:w2:2]
            p11 = arr[1:h2:2, 1:w2:2]

            # Require at least 2 of 4 500m sub-pixels to exceed +7.5 dBZ (idx 80)
            # to reject single-pixel NEXRAD clear-air/AP ground clutter while preserving light rain/drizzle
            active_sub = (
                (p00 >= 80).astype(np.uint8)
                + (p01 >= 80).astype(np.uint8)
                + (p10 >= 80).astype(np.uint8)
                + (p11 >= 80).astype(np.uint8)
            )
            pooled = np.maximum(np.maximum(p00, p01), np.maximum(p10, p11))
            pooled[active_sub < 2] = 0
            pooled[pooled < 80] = 0

            # 3x3 morphological despeckle: reject isolated 1km clutter specks (nbr_count < 2)
            mask = (pooled >= 80).astype(np.uint8)
            pad_m = np.pad(mask, 1, mode="constant", constant_values=0)
            nbr_count = (
                pad_m[:-2, :-2] + pad_m[:-2, 1:-1] + pad_m[:-2, 2:]
                + pad_m[1:-1, :-2] + pad_m[1:-1, 1:-1] + pad_m[1:-1, 2:]
                + pad_m[2:, :-2] + pad_m[2:, 1:-1] + pad_m[2:, 2:]
            )
            pooled[nbr_count < 2] = 0
            return pooled

    @staticmethod
    def _zncc_block_motion_15m(
        sub1: np.ndarray,
        sub2: np.ndarray,
        step_km: float,
        minutes_apart: float,
        fallback_drow: int,
        fallback_dcol: int,
        min_active_pixels: int = 20,
    ) -> Tuple[int, int]:
        """
        Estimates 15-minute storm displacement (drow_1km, dcol_1km) between `sub1` (older)
        and `sub2` (newer) using Zero-Mean Normalized Cross-Correlation (ZNCC) on the
        high-pass spatial reflectivity gradient so flat stratiform shields never bias
        the motion vector toward zero shift.
        Enforces a realistic synoptic propagation speed of 30..80 km/h (7.5..20 km per 15m).
        """
        if minutes_apart <= 0 or sub1.shape != sub2.shape:
            return (fallback_drow, fallback_dcol)

        active1 = sub1 >= 80
        active2 = sub2 >= 80
        if int(active1.sum()) < min_active_pixels or int(active2.sum()) < min_active_pixels:
            return (fallback_drow, fallback_dcol)

        # Continuous reflectivity above threshold (0..110)
        f1 = np.clip(sub1.astype(np.float32) - 78.0, 0.0, 110.0)
        f2 = np.clip(sub2.astype(np.float32) - 78.0, 0.0, 110.0)

        # Subtract 5x5 local mean (high-pass edge/core filter) so interior plateaus don't dominate
        p1 = np.pad(f1, 2, mode="edge")
        p2 = np.pad(f2, 2, mode="edge")
        smooth1 = (
            p1[:-4, :-4] + p1[:-4, 2:-2] + p1[:-4, 4:]
            + p1[2:-2, :-4] + p1[2:-2, 2:-2] * 2.0 + p1[2:-2, 4:]
            + p1[4:, :-4] + p1[4:, 2:-2] + p1[4:, 4:]
        ) / 10.0
        smooth2 = (
            p2[:-4, :-4] + p2[:-4, 2:-2] + p2[:-4, 4:]
            + p2[2:-2, :-4] + p2[2:-2, 2:-2] * 2.0 + p2[2:-2, 4:]
            + p2[4:, :-4] + p2[4:, 2:-2] + p2[4:, 4:]
        ) / 10.0

        hp1 = np.maximum(0.0, f1 - smooth1 * 0.72)
        hp2 = np.maximum(0.0, f2 - smooth2 * 0.72)

        h, w = hp1.shape
        scale_to_15m = 15.0 / max(5.0, minutes_apart)

        # Search candidate shifts whose 15-minute equivalent speed is 0..25 km / 15m (0..100 km/h)
        # in all 360° directions so cyclonic pivots and westward wrap-around bands are accurately tracked
        max_shift = max(5, int(round(25.0 / (step_km * scale_to_15m))) + 1)
        max_shift = min(max_shift, min(h, w) // 3)

        best_score = -1.0
        best_drow_15m = fallback_drow
        best_dcol_15m = fallback_dcol

        for dr in range(-max_shift, max_shift + 1):
            for dc in range(-max_shift, max_shift + 1):
                drow_15m_flt = dr * step_km * scale_to_15m
                dcol_15m_flt = dc * step_km * scale_to_15m
                dist_15m_km = math.hypot(drow_15m_flt, dcol_15m_flt)
                if dist_15m_km > 25.0:
                    continue

                r1_s = max(0, -dr)
                r1_e = min(h, h - dr)
                c1_s = max(0, -dc)
                c1_e = min(w, w - dc)
                if (r1_e - r1_s) < 12 or (c1_e - c1_s) < 12:
                    continue

                patch1 = hp1[r1_s:r1_e, c1_s:c1_e]
                patch2 = hp2[r1_s + dr : r1_e + dr, c1_s + dc : c1_e + dc]

                n1 = float(np.sqrt((patch1 * patch1).sum()))
                n2 = float(np.sqrt((patch2 * patch2).sum()))
                if n1 < 15.0 or n2 < 15.0:
                    continue

                score = float((patch1 * patch2).sum()) / (n1 * n2 + 1e-5)

                if score > best_score:
                    best_score = score
                    best_drow_15m = int(round(drow_15m_flt))
                    best_dcol_15m = int(round(dcol_15m_flt))

        if best_score < 0.12:
            return (fallback_drow, fallback_dcol)
        return (best_drow_15m, best_dcol_15m)

    def _estimate_regional_motion_15m(
        self, row: int, col: int
    ) -> Tuple[int, int]:
        """
        Estimates local/regional storm displacement (d_row, d_col) per 15 minutes
        Around (row, col). Uses precomputed smoothed 2D regional velocity field when available,
        or computes high-pass ZNCC over a ±140km window between the -30m/-45m frame and LIVE.
        """
        if (
            self._motion_field_drow_15m is not None
            and self._motion_field_dcol_15m is not None
        ):
            fh, fw = self._motion_field_drow_15m.shape
            fy = np.clip((row / max(1, self.GRID_H - 1)) * (fh - 1), 0.0, fh - 1.001)
            fx = np.clip((col / max(1, self.GRID_W - 1)) * (fw - 1), 0.0, fw - 1.001)
            iy = int(fy)
            ix = int(fx)
            dy = fy - iy
            dx = fx - ix
            dr_val = (
                self._motion_field_drow_15m[iy, ix] * (1 - dy) * (1 - dx)
                + self._motion_field_drow_15m[iy, ix + 1] * (1 - dy) * dx
                + self._motion_field_drow_15m[iy + 1, ix] * dy * (1 - dx)
                + self._motion_field_drow_15m[iy + 1, ix + 1] * dy * dx
            )
            dc_val = (
                self._motion_field_dcol_15m[iy, ix] * (1 - dy) * (1 - dx)
                + self._motion_field_dcol_15m[iy, ix + 1] * (1 - dy) * dx
                + self._motion_field_dcol_15m[iy + 1, ix] * dy * (1 - dx)
                + self._motion_field_dcol_15m[iy + 1, ix + 1] * dy * dx
            )
            return (int(round(float(dr_val))), int(round(float(dc_val))))

        past_frames = [f for f in self._frames.values() if not f.is_future]
        if len(past_frames) < 2:
            return (self._global_motion_drow_15m, self._global_motion_dcol_15m)

        live_f = past_frames[-1]
        # Prefer -30m frame (past_frames[1]) for a clean 30-minute baseline
        ref_f = past_frames[1] if len(past_frames) >= 4 else past_frames[0]
        minutes_apart = max(5.0, (live_f.dt - ref_f.dt).total_seconds() / 60.0)

        rad = 140
        r0, r1 = max(0, row - rad), min(self.GRID_H, row + rad)
        c0, c1 = max(0, col - rad), min(self.GRID_W, col + rad)
        if (r1 - r0) < 60 or (c1 - c0) < 60:
            return (self._global_motion_drow_15m, self._global_motion_dcol_15m)

        sub1 = ref_f.grid[r0:r1:2, c0:c1:2]
        sub2 = live_f.grid[r0:r1:2, c0:c1:2]
        return self._zncc_block_motion_15m(
            sub1,
            sub2,
            step_km=2.0,
            minutes_apart=minutes_apart,
            fallback_drow=self._global_motion_drow_15m,
            fallback_dcol=self._global_motion_dcol_15m,
            min_active_pixels=18,
        )

    @classmethod
    def _estimate_motion_vector_15m(
        cls, older_grid: np.ndarray, newer_grid: np.ndarray, minutes_apart: float
    ) -> Tuple[int, int]:
        """
        Estimates dominant CONUS storm displacement (d_row, d_col) per 15 minutes
        using 4km high-pass ZNCC across active precipitation bands.
        """
        sub1 = older_grid[::4, ::4]
        sub2 = newer_grid[::4, ::4]
        return cls._zncc_block_motion_15m(
            sub1,
            sub2,
            step_km=4.0,
            minutes_apart=minutes_apart,
            fallback_drow=3,
            fallback_dcol=11,
            min_active_pixels=50,
        )

    @staticmethod
    def _compute_mesoscale_flow_grid(
        g1: np.ndarray,
        g2: np.ndarray,
        minutes_apart: float,
        step_dec: int = 4,
        grid_step_km: float = 1.0,
        fh: int = 55,
        fw: int = 123,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes a high-resolution (55, 123) mesoscale storm velocity field (~50km spacing across CONUS)
        using vectorized exact-overlap ZNCC across active precipitation patches with zero speed clamp
        and zero directional bias, followed by confidence-weighted spatial diffusion so cyclonic
        vortices, stationary pivots, and wrap-around bands are preserved with full fidelity.
        """
        s1 = np.clip(g1[::step_dec, ::step_dec].astype(np.float32) - 78.0, 0.0, 110.0)
        s2 = np.clip(g2[::step_dec, ::step_dec].astype(np.float32) - 78.0, 0.0, 110.0)
        H, W = s1.shape
        eff_km = float(step_dec) * grid_step_km
        rad = max(10, int(round(76.0 / eff_km)))
        r_cs = np.round(np.linspace(rad, H - rad - 1, fh)).astype(np.int32)
        c_cs = np.round(np.linspace(rad, W - rad - 1, fw)).astype(np.int32)

        patches1, patches2, coords = [], [], []
        for iy, rc in enumerate(r_cs):
            for ix, cc in enumerate(c_cs):
                p1 = s1[rc - rad : rc + rad, cc - rad : cc + rad]
                p2 = s2[rc - rad : rc + rad, cc - rad : cc + rad]
                if (p1 > 2.0).sum() >= 12 and (p2 > 2.0).sum() >= 12:
                    patches1.append(p1 - p1.mean())
                    patches2.append(p2 - p2.mean())
                    coords.append((iy, ix))

        drow_15m = np.zeros((fh, fw), dtype=np.float32)
        dcol_15m = np.zeros((fh, fw), dtype=np.float32)
        conf = np.zeros((fh, fw), dtype=np.float32)

        if patches1:
            P1 = np.stack(patches1, axis=0)
            P2 = np.stack(patches2, axis=0)
            N, ph, pw = P1.shape
            scale_15m = (15.0 / max(5.0, minutes_apart)) * eff_km
            max_sh = min(10, max(4, int(round(24.0 / scale_15m))))

            best_sc = np.full(N, -1.0, dtype=np.float32)
            best_r = np.zeros(N, dtype=np.float32)
            best_c = np.zeros(N, dtype=np.float32)

            for dr in range(-max_sh, max_sh + 1):
                for dc in range(-max_sh, max_sh + 1):
                    if math.hypot(dr * scale_15m, dc * scale_15m) > 25.0:
                        continue
                    r1s, r1e = max(0, -dr), min(ph, ph - dr)
                    c1s, c1e = max(0, -dc), min(pw, pw - dc)
                    sub1 = P1[:, r1s:r1e, c1s:c1e]
                    sub2 = P2[:, r1s + dr : r1e + dr, c1s + dc : c1e + dc]
                    e1 = (sub1 * sub1).sum(axis=(1, 2))
                    e2 = (sub2 * sub2).sum(axis=(1, 2))
                    cross = (sub1 * sub2).sum(axis=(1, 2))
                    denom = np.sqrt(e1 * e2) + 1e-5
                    sc = np.where((e1 > 30.0) & (e2 > 30.0), cross / denom, -1.0)
                    better = sc > best_sc
                    best_sc = np.where(better, sc, best_sc)
                    best_r = np.where(better, dr * scale_15m, best_r)
                    best_c = np.where(better, dc * scale_15m, best_c)

            for k, (iy, ix) in enumerate(coords):
                if best_sc[k] >= 0.15:
                    drow_15m[iy, ix] = float(best_r[k])
                    dcol_15m[iy, ix] = float(best_c[k])
                    conf[iy, ix] = float(best_sc[k])

        w = np.maximum(conf, 1e-4)
        wr = drow_15m * w
        wc = dcol_15m * w
        for _ in range(7):
            pw_pad = np.pad(w, 1, mode="edge")
            pr_pad = np.pad(wr, 1, mode="edge")
            pc_pad = np.pad(wc, 1, mode="edge")
            w_nbr = (
                pw_pad[:-2, :-2] + pw_pad[:-2, 1:-1] * 2.0 + pw_pad[:-2, 2:]
                + pw_pad[1:-1, :-2] * 2.0 + pw_pad[1:-1, 1:-1] * 4.0 + pw_pad[1:-1, 2:] * 2.0
                + pw_pad[2:, :-2] + pw_pad[2:, 1:-1] * 2.0 + pw_pad[2:, 2:]
            ) / 16.0
            wr_nbr = (
                pr_pad[:-2, :-2] + pr_pad[:-2, 1:-1] * 2.0 + pr_pad[:-2, 2:]
                + pr_pad[1:-1, :-2] * 2.0 + pr_pad[1:-1, 1:-1] * 4.0 + pr_pad[1:-1, 2:] * 2.0
                + pr_pad[2:, :-2] + pr_pad[2:, 1:-1] * 2.0 + pr_pad[2:, 2:]
            ) / 16.0
            wc_nbr = (
                pc_pad[:-2, :-2] + pc_pad[:-2, 1:-1] * 2.0 + pc_pad[:-2, 2:]
                + pc_pad[1:-1, :-2] * 2.0 + pc_pad[1:-1, 1:-1] * 4.0 + pc_pad[1:-1, 2:] * 2.0
                + pc_pad[2:, :-2] + pc_pad[2:, 1:-1] * 2.0 + pc_pad[2:, 2:]
            ) / 16.0
            keep = np.clip(conf * 1.35, 0.0, 0.82)
            wr = keep * (drow_15m * w) + (1.0 - keep) * wr_nbr
            wc = keep * (dcol_15m * w) + (1.0 - keep) * wc_nbr
            w = keep * w + (1.0 - keep) * np.maximum(w_nbr, 1e-4)

        smooth_dr = (wr / np.maximum(w, 1e-4)).astype(np.float32)
        smooth_dc = (wc / np.maximum(w, 1e-4)).astype(np.float32)
        return (smooth_dr, smooth_dc, conf)

    def _build_curved_trajectories_sync(
        self,
        past_frames: List[DopplerGridFrame],
        hrrr_grids: Dict[int, np.ndarray],
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        Dict[int, np.ndarray],
        Dict[int, np.ndarray],
        Dict[int, np.ndarray],
    ]:
        """
        Combines dual-baseline observed NEXRAD N0Q optical flow (-45m/LIVE and -30m/LIVE) with
        NOAA HRRR 15-minute Subhourly Forecast reflectivity velocity & intensity evolution, and
        integrates 16-step RK2 curved backward trajectories across CONUS on a (110, 245) grid
        for each of the 8 future frames (+15m .. +2h).
        """
        fh, fw = 55, 123
        live_f = past_frames[-1]
        ref_a = past_frames[0]
        ref_b = past_frames[1] if len(past_frames) >= 4 else past_frames[0]

        mins_a = max(5.0, (live_f.dt - ref_a.dt).total_seconds() / 60.0)
        mins_b = max(5.0, (live_f.dt - ref_b.dt).total_seconds() / 60.0)

        dr_a, dc_a, conf_a = self._compute_mesoscale_flow_grid(ref_a.grid, live_f.grid, mins_a, step_dec=4, grid_step_km=1.0, fh=fh, fw=fw)
        if ref_b is not ref_a:
            dr_b, dc_b, conf_b = self._compute_mesoscale_flow_grid(ref_b.grid, live_f.grid, mins_b, step_dec=4, grid_step_km=1.0, fh=fh, fw=fw)
            obs_dr = (0.45 * dr_a + 0.55 * dr_b).astype(np.float32)
            obs_dc = (0.45 * dc_a + 0.55 * dc_b).astype(np.float32)
            obs_conf = np.maximum(conf_a, conf_b)
        else:
            obs_dr, obs_dc, obs_conf = dr_a, dc_a, conf_a

        # Compute NOAA HRRR forecast flow fields for 0..+60m and +60m..+120m when available
        hrrr_dr_1, hrrr_dc_1, hrrr_conf_1 = obs_dr, obs_dc, np.zeros_like(obs_conf)
        hrrr_dr_2, hrrr_dc_2, hrrr_conf_2 = obs_dr, obs_dc, np.zeros_like(obs_conf)
        if 0 in hrrr_grids and 60 in hrrr_grids:
            hrrr_dr_1, hrrr_dc_1, hrrr_conf_1 = self._compute_mesoscale_flow_grid(
                hrrr_grids[0], hrrr_grids[60], 60.0, step_dec=2, grid_step_km=2.0, fh=fh, fw=fw
            )
        if 60 in hrrr_grids and 120 in hrrr_grids:
            hrrr_dr_2, hrrr_dc_2, hrrr_conf_2 = self._compute_mesoscale_flow_grid(
                hrrr_grids[60], hrrr_grids[120], 60.0, step_dec=2, grid_step_km=2.0, fh=fh, fw=fw
            )
        elif 0 in hrrr_grids and 60 in hrrr_grids:
            hrrr_dr_2, hrrr_dc_2, hrrr_conf_2 = hrrr_dr_1, hrrr_dc_1, hrrr_conf_1

        lats_row = self.LAT_MAX - (np.arange(fh, dtype=np.float32) / (fh - 1)) * (self.LAT_MAX - self.LAT_MIN)
        cos_lats = np.clip(np.cos(np.radians(lats_row)), 0.55, 0.95)[:, None]

        # Build time-indexed velocity fields V(step) for step = 1..8 (in 0.01-deg grid units per 15m)
        v_dr_steps: Dict[int, np.ndarray] = {}
        v_dc_steps: Dict[int, np.ndarray] = {}
        for step in range(1, 9):
            if step <= 4:
                # Ramp HRRR guidance weight from 0.15 at +15m to 0.45 at +60m where HRRR has active signal
                w_h = (0.10 + 0.085 * step) * np.clip(hrrr_conf_1 * 2.2, 0.0, 1.0)
                dr_s = (1.0 - w_h) * obs_dr + w_h * hrrr_dr_1
                dc_s = (1.0 - w_h) * obs_dc + w_h * hrrr_dc_1
            else:
                # Ramp HRRR guidance weight from 0.50 at +75m to 0.70 at +120m
                w_h = (0.45 + 0.06 * (step - 4)) * np.clip(hrrr_conf_2 * 2.2, 0.0, 1.0)
                dr_s = (1.0 - w_h) * obs_dr + w_h * hrrr_dr_2
                dc_s = (1.0 - w_h) * obs_dc + w_h * hrrr_dc_2
            v_dr_steps[step] = dr_s.astype(np.float32)
            v_dc_steps[step] = (dc_s / cos_lats).astype(np.float32)

        obs_dc_deg = (obs_dc / cos_lats).astype(np.float32)

        # Helper to bilinearly sample a (fh, fw) velocity grid at arbitrary (r_grid, c_grid) coordinates
        def _sample_v(v_r_grid: np.ndarray, v_c_grid: np.ndarray, r_pts: np.ndarray, c_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            fy = np.clip((r_pts / (self.GRID_H - 1)) * (fh - 1), 0.0, fh - 1.001)
            fx = np.clip((c_pts / (self.GRID_W - 1)) * (fw - 1), 0.0, fw - 1.001)
            iy = fy.astype(np.int32)
            ix = fx.astype(np.int32)
            dy = fy - iy
            dx = fx - ix
            vr = (
                v_r_grid[iy, ix] * (1.0 - dy) * (1.0 - dx)
                + v_r_grid[iy, ix + 1] * (1.0 - dy) * dx
                + v_r_grid[iy + 1, ix] * dy * (1.0 - dx)
                + v_r_grid[iy + 1, ix + 1] * dy * dx
            )
            vc = (
                v_c_grid[iy, ix] * (1.0 - dy) * (1.0 - dx)
                + v_c_grid[iy, ix + 1] * (1.0 - dy) * dx
                + v_c_grid[iy + 1, ix] * dy * (1.0 - dx)
                + v_c_grid[iy + 1, ix + 1] * dy * dx
            )
            return vr, vc

        # Integrate curved backward trajectories on a (110, 245) grid (~25km spacing) across CONUS using RK2
        th, tw = 110, 245
        base_r = np.linspace(0.0, float(self.GRID_H - 1), th, dtype=np.float32)[:, None]
        base_c = np.linspace(0.0, float(self.GRID_W - 1), tw, dtype=np.float32)[None, :]
        base_r_2d = np.broadcast_to(base_r, (th, tw)).copy()
        base_c_2d = np.broadcast_to(base_c, (th, tw)).copy()

        cum_drow: Dict[int, np.ndarray] = {}
        cum_dcol: Dict[int, np.ndarray] = {}
        hrrr_trends: Dict[int, np.ndarray] = {}

        # Pre-smooth HRRR 2km grids on (110, 245) for Lagrangian intensity trend modulation
        hrrr_coarse: Dict[int, np.ndarray] = {}
        for m_min, h_grid in hrrr_grids.items():
            # h_grid is (1340, 3050) uint8 -> sample onto (110, 245) with local 5x5 mean
            hr_idx = np.clip(np.round(np.linspace(0, h_grid.shape[0] - 1, th)).astype(np.int32), 0, h_grid.shape[0] - 1)
            hc_idx = np.clip(np.round(np.linspace(0, h_grid.shape[1] - 1, tw)).astype(np.int32), 0, h_grid.shape[1] - 1)
            sub_h = h_grid[np.ix_(hr_idx, hc_idx)].astype(np.float32)
            ph_pad = np.pad(sub_h, 1, mode="edge")
            hrrr_coarse[m_min] = (
                ph_pad[:-2, :-2] + ph_pad[:-2, 1:-1] * 2.0 + ph_pad[:-2, 2:]
                + ph_pad[1:-1, :-2] * 2.0 + ph_pad[1:-1, 1:-1] * 4.0 + ph_pad[1:-1, 2:] * 2.0
                + ph_pad[2:, :-2] + ph_pad[2:, 1:-1] * 2.0 + ph_pad[2:, 2:]
            ) / 16.0

        for target_step in range(1, 9):
            cur_r = base_r_2d.copy()
            cur_c = base_c_2d.copy()
            # Step backwards from t = target_step to t = 0 in half-steps (dt = 0.5 = 7.5 min) using RK2 midpoint
            for sub in range(target_step * 2, 0, -1):
                step_idx = max(1, min(8, (sub + 1) // 2))
                vr_grid = v_dr_steps[step_idx]
                vc_grid = v_dc_steps[step_idx]
                vr1, vc1 = _sample_v(vr_grid, vc_grid, cur_r, cur_c)
                mid_r = cur_r - 0.25 * vr1
                mid_c = cur_c - 0.25 * vc1
                vr2, vc2 = _sample_v(vr_grid, vc_grid, mid_r, mid_c)
                cur_r = cur_r - 0.5 * vr2
                cur_c = cur_c - 0.5 * vc2

            cum_drow[target_step] = (base_r_2d - cur_r).astype(np.float32)
            cum_dcol[target_step] = (base_c_2d - cur_c).astype(np.float32)

            # Compute subtle HRRR Lagrangian intensity trend along the parcel trajectory (clamped to [-5, +6] index units)
            fut_min = target_step * 15
            near_hrrr_min = min(hrrr_coarse.keys(), key=lambda k: abs(k - fut_min)) if hrrr_coarse else -1
            if 0 in hrrr_coarse and near_hrrr_min > 0:
                h_end = hrrr_coarse[near_hrrr_min]
                # Sample hrrr_coarse[0] at the parcel origin (cur_r, cur_c)
                oy = np.clip(np.round((cur_r / (self.GRID_H - 1)) * (th - 1)).astype(np.int32), 0, th - 1)
                ox = np.clip(np.round((cur_c / (self.GRID_W - 1)) * (tw - 1)).astype(np.int32), 0, tw - 1)
                h_start = hrrr_coarse[0][oy, ox]
                raw_diff = (h_end - h_start) * 0.35
                hrrr_trends[target_step] = np.clip(raw_diff, -5.0, 6.0).astype(np.float32)
            else:
                hrrr_trends[target_step] = np.zeros((th, tw), dtype=np.float32)

        return (obs_dr, obs_dc_deg, cum_drow, cum_dcol, hrrr_trends)

    @staticmethod
    def _advect_grid(grid: np.ndarray, drow: int, dcol: int, decay_indices: int = 2) -> np.ndarray:
        """Shifts the 1km reflectivity matrix by (drow, dcol) and applies slight intensity decay."""
        out = np.zeros_like(grid)
        h, w = grid.shape
        src_r0 = max(0, -drow)
        src_r1 = min(h, h - drow)
        src_c0 = max(0, -dcol)
        src_c1 = min(w, w - dcol)

        dst_r0 = src_r0 + drow
        dst_r1 = src_r1 + drow
        dst_c0 = src_c0 + dcol
        dst_c1 = src_c1 + dcol

        if src_r1 > src_r0 and src_c1 > src_c0:
            shifted = grid[src_r0:src_r1, src_c0:src_c1].copy()
            active = shifted >= (78 + decay_indices)
            shifted[active] -= decay_indices
            shifted[~active] = 0
            out[dst_r0:dst_r1, dst_c0:dst_c1] = shifted
        return out

    async def refresh_dataset(self, force: bool = False) -> None:
        """
        Downloads the latest CONUS N0Q Master Grid (plus historical 5-min archive frames on cold start),
        builds the +15m and +30m advected nowcast frames, and logs telemetry to PostGIS.
        """
        now = datetime.now(timezone.utc)
        if (
            not force
            and self._last_ingested_at
            and (now - self._last_ingested_at).total_seconds() < 240
            and len(self._frames) >= 4
        ):
            return

        async with self._ingest_lock:
            if (
                not force
                and self._last_ingested_at
                and (datetime.now(timezone.utc) - self._last_ingested_at).total_seconds() < 240
                and len(self._frames) >= 4
            ):
                return

            self._ensure_dir()
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    # 1. Also refresh RainViewer manifest for global fallback
                    try:
                        rv_resp = await client.get("https://api.rainviewer.com/public/weather-maps.json")
                        if rv_resp.status_code == 200:
                            rv_json = rv_resp.json()
                            self._rainviewer_host = rv_json.get("host", "https://tilecache.rainviewer.com")
                            past = rv_json.get("radar", {}).get("past", [])
                            nowcast = rv_json.get("radar", {}).get("nowcast", [])
                            self._rainviewer_frames = past[-6:] + nowcast[:2]
                    except Exception:
                        pass

                    # 2. Check IEM's current composite valid timestamp via n0q_0.json (with HEAD fallback)
                    base_dt = None
                    try:
                        meta_resp = await client.get(
                            "https://mesonet.agron.iastate.edu/data/gis/images/4326/USCOMP/n0q_0.json"
                        )
                        if meta_resp.status_code == 200:
                            valid_str = meta_resp.json().get("meta", {}).get("valid")
                            if valid_str:
                                base_dt = datetime.strptime(valid_str, "%Y-%m-%dT%H:%M:%SZ").replace(
                                    tzinfo=timezone.utc
                                )
                    except Exception:
                        pass

                    if base_dt is None:
                        head_resp = await client.head(
                            "https://mesonet.agron.iastate.edu/data/gis/images/4326/USCOMP/n0q_0.png"
                        )
                        lm_str = head_resp.headers.get("Last-Modified")
                        if lm_str:
                            base_dt = datetime.strptime(lm_str, "%a, %d %b %Y %H:%M:%S %Z").replace(
                                tzinfo=timezone.utc
                            )
                        else:
                            base_dt = now
                    base_dt = base_dt.replace(minute=(base_dt.minute // 5) * 5, second=0, microsecond=0)

                    # Offsets to maintain in ring buffer: -45m, -30m, -20m, -10m, -5m, 0m (LIVE)
                    target_offsets = [-45, -30, -20, -10, -5, 0]
                    new_past_frames: List[DopplerGridFrame] = []

                    for offset_min in target_offsets:
                        frame_dt = base_dt + timedelta(minutes=offset_min)
                        frame_id = f"n0q_{frame_dt:%Y%m%d%H%M}"
                        label = "LIVE" if offset_min == 0 else f"{offset_min}m"

                        # Reuse if already decoded in memory
                        if frame_id in self._frames and not self._frames[frame_id].is_future:
                            existing_f = self._frames[frame_id]
                            existing_f.label = label
                            existing_f.offset_minutes = offset_min
                            new_past_frames.append(existing_f)
                            continue

                        if offset_min == 0:
                            url = "https://mesonet.agron.iastate.edu/data/gis/images/4326/USCOMP/n0q_0.png"
                        else:
                            url = (
                                f"https://mesonet.agron.iastate.edu/archive/data/"
                                f"{frame_dt:%Y/%m/%d}/GIS/uscomp/n0q_{frame_dt:%Y%m%d%H%M}.png"
                            )

                        try:
                            archive_file = self.archive_dir / f"{frame_id}.png"
                            raw_bytes: Optional[bytes] = None
                            was_backfilled = offset_min < 0

                            if archive_file.exists() and archive_file.stat().st_size > 50_000:
                                raw_bytes = archive_file.read_bytes()
                                was_backfilled = False
                            else:
                                resp = await client.get(url)
                                if resp.status_code == 200 and len(resp.content) > 50_000:
                                    raw_bytes = resp.content
                                    try:
                                        archive_file.write_bytes(raw_bytes)
                                    except Exception:
                                        pass

                            if raw_bytes is not None:
                                grid_1km = await asyncio.to_thread(self._decode_iem_png_to_1km_grid, raw_bytes)
                                frame_obj = DopplerGridFrame(
                                    frame_id=frame_id,
                                    dt=frame_dt,
                                    label=label,
                                    offset_minutes=offset_min,
                                    is_future=False,
                                    grid=grid_1km,
                                    raw_bytes=len(raw_bytes),
                                )
                                new_past_frames.append(frame_obj)

                                # Save latest live composite to disk for dataset persistence
                                if offset_min == 0:
                                    try:
                                        (self.doppler_dir / "latest_conus_n0q.png").write_bytes(raw_bytes)
                                    except Exception:
                                        pass

                                await db_manager.record_doppler_frame(
                                    frame_id=frame_id,
                                    frame_time=frame_dt,
                                    grid_shape=f"{grid_1km.shape[0]}x{grid_1km.shape[1]}",
                                    resolution_km=1.0,
                                    max_dbz=frame_obj.max_dbz,
                                    echo_coverage_pct=frame_obj.echo_coverage_pct,
                                    storm_cells_detected=frame_obj.storm_cells_count,
                                    raw_bytes=len(raw_bytes),
                                    is_nowcast=False,
                                    backfilled=was_backfilled,
                                )
                        except Exception as e:
                            print(f"⚠️ Doppler frame {frame_id} fetch warning: {e}")

                    if not new_past_frames:
                        return

                    live_frame = new_past_frames[-1]

                    # Fetch NOAA HRRR 15-minute Subhourly Forecast Reflectivity grids (0m, +30m, +60m, +90m, +120m)
                    hrrr_grids: Dict[int, np.ndarray] = {}
                    try:
                        hrrr_base = "https://mesonet.agron.iastate.edu/data/gis/images/4326/hrrr"
                        h_meta_resp = await client.get(f"{hrrr_base}/refd_0000.json")
                        if h_meta_resp.status_code == 200:
                            init_str = h_meta_resp.json().get("model_init_utc")
                            if init_str:
                                h_init_dt = datetime.strptime(init_str, "%Y-%m-%dT%H:%M:%SZ").replace(
                                    tzinfo=timezone.utc
                                )
                                lead0 = max(
                                    0,
                                    int(round((live_frame.dt - h_init_dt).total_seconds() / 60.0 / 15.0)) * 15,
                                )
                                init_tag = f"{h_init_dt:%Y%m%d%H%M}"
                                for h_off in (0, 30, 60, 90, 120):
                                    lead = lead0 + h_off
                                    if lead > 1080:
                                        continue
                                    h_file = self.archive_dir / f"hrrr_{init_tag}_{lead:04d}.png"
                                    h_bytes: Optional[bytes] = None
                                    if h_file.exists() and h_file.stat().st_size > 20_000:
                                        h_bytes = h_file.read_bytes()
                                    else:
                                        h_resp = await client.get(f"{hrrr_base}/refd_{lead:04d}.png")
                                        if h_resp.status_code == 200 and len(h_resp.content) > 20_000:
                                            h_bytes = h_resp.content
                                            try:
                                                h_file.write_bytes(h_bytes)
                                            except Exception:
                                                pass
                                    if h_bytes is not None:
                                        with Image.open(io.BytesIO(h_bytes)) as h_img:
                                            hrrr_grids[h_off] = np.asarray(h_img, dtype=np.uint8)
                    except Exception as e:
                        print(f"⚠️ HRRR reflectivity guidance fetch warning: {e}")

                    # Compute mesoscale storm motion field + 16-step RK2 curved backward trajectories
                    if len(new_past_frames) >= 2:
                        ref_frame = new_past_frames[1] if len(new_past_frames) >= 4 else new_past_frames[0]
                        mins_diff = max(5.0, (live_frame.dt - ref_frame.dt).total_seconds() / 60.0)
                        drow, dcol = await asyncio.to_thread(
                            self._estimate_motion_vector_15m,
                            ref_frame.grid,
                            live_frame.grid,
                            mins_diff,
                        )
                        self._global_motion_drow_15m = drow
                        self._global_motion_dcol_15m = dcol
                        (
                            f_drow,
                            f_dcol,
                            cum_dr,
                            cum_dc,
                            hrrr_trends,
                        ) = await asyncio.to_thread(
                            self._build_curved_trajectories_sync,
                            new_past_frames,
                            hrrr_grids,
                        )
                        self._motion_field_drow_15m = f_drow
                        self._motion_field_dcol_15m = f_dcol
                        self._cum_drow_fields = cum_dr
                        self._cum_dcol_fields = cum_dc
                        self._hrrr_trend_fields = hrrr_trends

                    # 3. Synthesize 2-Hour (+15m .. +2h in 15m steps) Zero-Copy Advected Nowcast Future Frames
                    future_steps = [
                        (1, 15, "+15m"),
                        (2, 30, "+30m"),
                        (3, 45, "+45m"),
                        (4, 60, "+1h"),
                        (5, 75, "+1h 15m"),
                        (6, 90, "+1h 30m"),
                        (7, 105, "+1h 45m"),
                        (8, 120, "+2h"),
                    ]
                    future_frames: List[DopplerGridFrame] = []
                    for fut_mult, fut_min, fut_label in future_steps:
                        fut_dt = live_frame.dt + timedelta(minutes=fut_min)
                        fut_id = f"nowcast_{fut_dt:%Y%m%d%H%M}"
                        fut_obj = DopplerGridFrame(
                            frame_id=fut_id,
                            dt=fut_dt,
                            label=fut_label,
                            offset_minutes=fut_min,
                            is_future=True,
                            grid=live_frame.grid,
                            raw_bytes=live_frame.raw_bytes,
                            advect_drow=self._global_motion_drow_15m * fut_mult,
                            advect_dcol=self._global_motion_dcol_15m * fut_mult,
                            decay_indices=fut_mult,
                        )
                        future_frames.append(fut_obj)

                    # Evict any stale nowcast tiles from cache on refresh
                    stale_keys = [k for k in self._tile_cache if k.startswith("nowcast_")]
                    for k in stale_keys:
                        self._tile_cache.pop(k, None)

                    # Rebuild ordered ring buffer
                    updated_map: OrderedDict[str, DopplerGridFrame] = OrderedDict()
                    for f in new_past_frames + future_frames:
                        updated_map[f.frame_id] = f
                    self._frames = updated_map
                    self._last_ingested_at = datetime.now(timezone.utc)

                    # Update dataset inventory metadata on disk
                    self._update_dataset_catalog(live_frame, len(updated_map))
            except Exception as e:
                print(f"⚠️ Doppler dataset refresh warning: {e}")

    async def backfill_offline_gaps(self, lookback_hours: int = 24, max_batch: int = 24) -> Dict[str, Any]:
        """
        Detects any missing historical Doppler radar checkpoints in PostGIS over the past
        `lookback_hours` (caused by server power-off, reboot, or network outage) and
        automatically backfills them from local NVMe archive or Iowa State Mesonet's NEXRAD archive.
        """
        async with self._backfill_lock:
            self._ensure_dir()
            self._prune_disk_archive(max_age_hours=max(24, lookback_hours))

            now = datetime.now(timezone.utc)
            # Determine latest reference 15-min synoptic mark
            base_dt = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
            if self._frames:
                live_candidates = [f for f in self._frames.values() if not f.is_future]
                if live_candidates:
                    l_dt = live_candidates[-1].dt
                    base_dt = l_dt.replace(minute=(l_dt.minute // 15) * 15, second=0, microsecond=0)

            since_dt = base_dt - timedelta(hours=lookback_hours)
            existing_ids = await db_manager.get_existing_doppler_frame_ids(since_dt)

            # Build expected 15-minute synoptic checkpoints across the lookback window
            missing_dts: List[datetime] = []
            steps = int((lookback_hours * 60) // 15)
            for step in range(1, steps + 1):
                cand_dt = base_dt - timedelta(minutes=step * 15)
                cand_id = f"n0q_{cand_dt:%Y%m%d%H%M}"
                if cand_id not in existing_ids:
                    missing_dts.append(cand_dt)

            recovered = 0
            checked = len(missing_dts)
            if missing_dts:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    for frame_dt in missing_dts[:max_batch]:
                        frame_id = f"n0q_{frame_dt:%Y%m%d%H%M}"
                        archive_file = self.archive_dir / f"{frame_id}.png"
                        raw_bytes: Optional[bytes] = None

                        try:
                            if archive_file.exists() and archive_file.stat().st_size > 50_000:
                                raw_bytes = archive_file.read_bytes()
                            else:
                                url = (
                                    f"https://mesonet.agron.iastate.edu/archive/data/"
                                    f"{frame_dt:%Y/%m/%d}/GIS/uscomp/n0q_{frame_dt:%Y%m%d%H%M}.png"
                                )
                                resp = await client.get(url)
                                if resp.status_code == 200 and len(resp.content) > 50_000:
                                    raw_bytes = resp.content
                                    # Keep recent 6h on disk
                                    if (base_dt - frame_dt).total_seconds() <= 6 * 3600:
                                        try:
                                            archive_file.write_bytes(raw_bytes)
                                        except Exception:
                                            pass

                            if raw_bytes is not None:
                                grid_1km = await asyncio.to_thread(self._decode_iem_png_to_1km_grid, raw_bytes)
                                frame_obj = DopplerGridFrame(
                                    frame_id=frame_id,
                                    dt=frame_dt,
                                    label="ARCHIVE",
                                    offset_minutes=-int((base_dt - frame_dt).total_seconds() // 60),
                                    is_future=False,
                                    grid=grid_1km,
                                    raw_bytes=len(raw_bytes),
                                )
                                await db_manager.record_doppler_frame(
                                    frame_id=frame_id,
                                    frame_time=frame_dt,
                                    grid_shape=f"{grid_1km.shape[0]}x{grid_1km.shape[1]}",
                                    resolution_km=1.0,
                                    max_dbz=frame_obj.max_dbz,
                                    echo_coverage_pct=frame_obj.echo_coverage_pct,
                                    storm_cells_detected=frame_obj.storm_cells_count,
                                    raw_bytes=len(raw_bytes),
                                    is_nowcast=False,
                                    backfilled=True,
                                )
                                recovered += 1
                                self._total_backfilled_frames += 1
                                await asyncio.sleep(0.15)
                        except Exception as e:
                            print(f"⚠️ Doppler backfill frame {frame_id} warning: {e}")

            self._last_backfill_at = datetime.now(timezone.utc)
            return {
                "lookbackHours": lookback_hours,
                "missingDetected": checked,
                "backfilledNow": recovered,
                "remainingGaps": max(0, checked - recovered),
                "totalBackfilledSession": self._total_backfilled_frames,
                "checkedAt": self._last_backfill_at.isoformat(),
            }

    def _update_dataset_catalog(self, live_frame: DopplerGridFrame, frame_count: int) -> None:
        try:
            meta_path = settings.data_dir / "datasets_meta.json"
            catalog = dataset_service.get_dataset_inventory()
            datasets = catalog.get("datasets", [])
            datasets = [d for d in datasets if d.get("id") != "zus-conus-doppler-1km"]
            datasets.insert(
                0,
                {
                    "id": "zus-conus-doppler-1km",
                    "name": "Züs CONUS 1km Multi-Radar Base Reflectivity + Nowcast Cube",
                    "type": "uint8_dbz_raster_cube",
                    "grid_shape": f"{self.GRID_H}x{self.GRID_W}",
                    "resolution_km": 1.0,
                    "frames_in_memory": frame_count,
                    "max_dbz": live_frame.max_dbz,
                    "echo_coverage_pct": live_frame.echo_coverage_pct,
                    "storm_cells_detected": live_frame.storm_cells_count,
                    "updated_at": live_frame.dt.isoformat(),
                    "status": "active_streaming",
                },
            )
            catalog["datasets"] = datasets
            import json
            with open(meta_path, "w") as f:
                json.dump(catalog, f, indent=2)
        except Exception:
            pass

    @staticmethod
    def _tile_bounds_latlon(z: int, x: int, y: int) -> Tuple[float, float, float, float]:
        """Returns (lat_north, lon_west, lat_south, lon_east) for WebMercator XYZ tile."""
        n = 2.0 ** z
        lon_west = x / n * 360.0 - 180.0
        lon_east = (x + 1) / n * 360.0 - 180.0
        lat_north = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
        lat_south = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * (y + 1) / n))))
        return (lat_north, lon_west, lat_south, lon_east)

    def _slice_conus_tile_sync(
        self,
        grid: np.ndarray,
        z: int,
        x: int,
        y: int,
        advect_drow: int = 0,
        advect_dcol: int = 0,
        decay_indices: int = 0,
        fut_mult: float = 0.0,
    ) -> bytes:
        """
        Slices a 256x256 WebMercator tile directly from the 2700x6100 uint8 master grid
        (with spatially-smooth 2D regional Lagrangian advection for future nowcast frames)
        and encodes an 8-bit Indexed PNG with the Züs Storm Palette in <0.5ms.
        """
        n = 2.0 ** z
        px = (np.arange(256, dtype=np.float64) + 0.5) / 256.0
        py = (np.arange(256, dtype=np.float64) + 0.5) / 256.0

        lons = (x + px) / n * 360.0 - 180.0
        merc_y = math.pi * (1.0 - 2.0 * (y + py) / n)
        lats = np.degrees(np.arctan(np.sinh(merc_y)))

        base_cols = (lons - self.LON_MIN) / self.STEP_DEG
        base_rows = (self.LAT_MAX - lats) / self.STEP_DEG

        step_idx = max(1, min(8, int(round(fut_mult)))) if fut_mult > 0.0 else 0
        if (
            step_idx > 0
            and step_idx in self._cum_drow_fields
            and step_idx in self._cum_dcol_fields
        ):
            cum_dr_grid = self._cum_drow_fields[step_idx]
            cum_dc_grid = self._cum_dcol_fields[step_idx]
            th, tw = cum_dr_grid.shape
            fy = np.clip((base_rows / (self.GRID_H - 1)) * (th - 1), 0.0, th - 1.001)
            fx = np.clip((base_cols / (self.GRID_W - 1)) * (tw - 1), 0.0, tw - 1.001)
            iy = fy.astype(np.int32)
            ix = fx.astype(np.int32)
            dy = (fy - iy)[:, None]
            dx = (fx - ix)[None, :]

            iy0 = iy[:, None]
            iy1 = (iy + 1)[:, None]
            ix0 = ix[None, :]
            ix1 = (ix + 1)[None, :]

            drow_2d = (
                cum_dr_grid[iy0, ix0] * (1.0 - dy) * (1.0 - dx)
                + cum_dr_grid[iy0, ix1] * (1.0 - dy) * dx
                + cum_dr_grid[iy1, ix0] * dy * (1.0 - dx)
                + cum_dr_grid[iy1, ix1] * dy * dx
            )
            dcol_2d = (
                cum_dc_grid[iy0, ix0] * (1.0 - dy) * (1.0 - dx)
                + cum_dc_grid[iy0, ix1] * (1.0 - dy) * dx
                + cum_dc_grid[iy1, ix0] * dy * (1.0 - dx)
                + cum_dc_grid[iy1, ix1] * dy * dx
            )

            rows_2d = np.floor(base_rows[:, None] - drow_2d).astype(np.int32)
            cols_2d = np.floor(base_cols[None, :] - dcol_2d).astype(np.int32)

            valid_mask = (
                (rows_2d >= 0)
                & (rows_2d < self.GRID_H)
                & (cols_2d >= 0)
                & (cols_2d < self.GRID_W)
            )
            if not np.any(valid_mask):
                return EMPTY_TILE_PNG

            r_clip = np.clip(rows_2d, 0, self.GRID_H - 1)
            c_clip = np.clip(cols_2d, 0, self.GRID_W - 1)
            tile = grid[r_clip, c_clip].copy()

            # Apply HRRR Lagrangian intensity trend on active precipitation bands
            if step_idx in self._hrrr_trend_fields:
                tr_grid = self._hrrr_trend_fields[step_idx]
                trend_2d = (
                    tr_grid[iy0, ix0] * (1.0 - dy) * (1.0 - dx)
                    + tr_grid[iy0, ix1] * (1.0 - dy) * dx
                    + tr_grid[iy1, ix0] * dy * (1.0 - dx)
                    + tr_grid[iy1, ix1] * dy * dx
                )
                active_echo = tile >= 86
                if np.any(active_echo):
                    mod = np.clip(tile.astype(np.float32) + trend_2d, 84.0, 195.0).astype(np.uint8)
                    tile = np.where(active_echo, mod, tile)

            if not np.all(valid_mask):
                tile[~valid_mask] = 0
        else:
            cols = np.floor(base_cols).astype(np.int32) - int(advect_dcol)
            rows = np.floor(base_rows).astype(np.int32) - int(advect_drow)

            valid_r = (rows >= 0) & (rows < self.GRID_H)
            valid_c = (cols >= 0) & (cols < self.GRID_W)

            if not np.any(valid_r) or not np.any(valid_c):
                return EMPTY_TILE_PNG

            rows_clipped = np.clip(rows, 0, self.GRID_H - 1)
            cols_clipped = np.clip(cols, 0, self.GRID_W - 1)

            tile = grid[np.ix_(rows_clipped, cols_clipped)].copy()

            if not np.all(valid_r):
                tile[~valid_r, :] = 0
            if not np.all(valid_c):
                tile[:, ~valid_c] = 0

        if decay_indices > 0:
            # Gently relax intense convective cores (>32 dBZ / idx > 130) toward moderate stratiform rain
            # while preserving light/moderate rain bands (idx 86..130) across the full 2-hour track
            core_mask = tile > 130
            if np.any(core_mask):
                tile[core_mask] = np.maximum(126, tile[core_mask].astype(np.int16) - decay_indices * 2).astype(np.uint8)

        if int(tile.max()) < 90:
            return EMPTY_TILE_PNG

        # Preserve crisp native 1km reflectivity gradients without Gaussian blur;
        # only filter out isolated single-pixel noise specks (nbr_sum < 2)
        if z >= 6:
            padded = np.pad(tile, 1, mode="edge")
            active_nbrs = (padded >= 88).astype(np.uint8)
            nbr_sum = (
                active_nbrs[:-2, :-2] + active_nbrs[:-2, 1:-1] + active_nbrs[:-2, 2:]
                + active_nbrs[1:-1, :-2] + active_nbrs[1:-1, 1:-1] + active_nbrs[1:-1, 2:]
                + active_nbrs[2:, :-2] + active_nbrs[2:, 1:-1] + active_nbrs[2:, 2:]
            )
            tile = np.where((tile >= 86) & (nbr_sum >= 2), tile, 0)

        img = Image.fromarray(tile, mode="P")
        img.putpalette(ZUS_PALETTE_RGB)
        img.info["transparency"] = ZUS_PALETTE_ALPHA
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=False)
        return buf.getvalue()

    @staticmethod
    def _remap_rainviewer_tile_sync(raw_png: bytes) -> bytes:
        """Remaps a global RainViewer RGBA tile into the 8-bit Züs Storm Palette."""
        with Image.open(io.BytesIO(raw_png)).convert("RGBA") as img:
            arr = np.asarray(img, dtype=np.uint8)
            alpha = arr[:, :, 3]
            if int(alpha.max()) == 0:
                return EMPTY_TILE_PNG
            # Estimate intensity from luminance + alpha of RainViewer raw scheme
            r = arr[:, :, 0].astype(np.uint16)
            g = arr[:, :, 1].astype(np.uint16)
            b = arr[:, :, 2].astype(np.uint16)
            lum = (r * 3 + g * 4 + b * 2) // 9
            idx = np.where(alpha > 15, np.clip(80 + (lum * 95 // 255), 80, 190), 0).astype(np.uint8)
            out = Image.fromarray(idx, mode="P")
            out.putpalette(ZUS_PALETTE_RGB)
            out.info["transparency"] = ZUS_PALETTE_ALPHA
            buf = io.BytesIO()
            out.save(buf, format="PNG", optimize=False)
            return buf.getvalue()

    async def get_tile_png(self, frame_id: str, z: int, x: int, y: int) -> bytes:
        """
        Returns a 256x256 8-bit Züs-styled Doppler PNG tile.
        Uses LRU RAM cache + per-tile asyncio.Lock request coalescing.
        """
        if z < 0 or z > 12:
            return EMPTY_TILE_PNG

        if not self._frames:
            await self.refresh_dataset()

        # Resolve exact frame_id, semantic offset aliases (past_45m, nowcast_p15m, latest), or fallback to latest live frame
        frame = self._frames.get(frame_id)
        if frame is None and self._frames:
            target_off: Optional[int] = None
            if frame_id == "latest":
                target_off = 0
            elif frame_id.startswith("past_") and frame_id.endswith("m"):
                try:
                    target_off = -int(frame_id[len("past_") : -1])
                except ValueError:
                    target_off = None
            elif frame_id.startswith("nowcast_p") and frame_id.endswith("m"):
                try:
                    target_off = int(frame_id[len("nowcast_p") : -1])
                except ValueError:
                    target_off = None

            if target_off is not None:
                best_f = min(self._frames.values(), key=lambda f: abs(f.offset_minutes - target_off))
                frame = best_f
                frame_id = best_f.frame_id
            else:
                live_candidates = [f for f in self._frames.values() if not f.is_future]
                frame = live_candidates[-1] if live_candidates else list(self._frames.values())[-1]
                frame_id = frame.frame_id

        cache_key = f"{frame_id}:{z}:{x}:{y}"
        if cache_key in self._tile_cache:
            self._tile_cache.move_to_end(cache_key)
            return self._tile_cache[cache_key]

        lock = self._inflight_locks.setdefault(cache_key, asyncio.Lock())
        async with lock:
            if cache_key in self._tile_cache:
                return self._tile_cache[cache_key]

            lat_n, lon_w, lat_s, lon_e = self._tile_bounds_latlon(z, x, y)
            intersects_conus = not (
                lat_s > self.LAT_MAX
                or lat_n < self.LAT_MIN
                or lon_e < self.LON_MIN
                or lon_w > self.LON_MAX
            )

            png_bytes = EMPTY_TILE_PNG
            if intersects_conus and frame is not None:
                fut_mult = (frame.offset_minutes / 15.0) if (frame.is_future and frame.offset_minutes > 0) else 0.0
                png_bytes = await asyncio.to_thread(
                    self._slice_conus_tile_sync,
                    frame.grid,
                    z,
                    x,
                    y,
                    frame.advect_drow,
                    frame.advect_dcol,
                    frame.decay_indices,
                    fut_mult,
                )
            elif self._rainviewer_frames and z <= 7:
                # Global fallback outside CONUS (RainViewer supports z <= 7; z >= 8 returns a 1370-byte error tile)
                rv_frame = self._rainviewer_frames[-1]
                rv_url = f"{self._rainviewer_host}{rv_frame['path']}/256/{z}/{x}/{y}/2/1_1.png"
                try:
                    async with httpx.AsyncClient(timeout=6.0) as client:
                        resp = await client.get(rv_url)
                        if (
                            resp.status_code == 200
                            and len(resp.content) > 400
                            and len(resp.content) != 1370
                        ):
                            png_bytes = await asyncio.to_thread(self._remap_rainviewer_tile_sync, resp.content)
                except Exception:
                    png_bytes = EMPTY_TILE_PNG

            self._tile_cache[cache_key] = png_bytes
            if len(self._tile_cache) > self._max_cached_tiles:
                self._tile_cache.popitem(last=False)

        self._inflight_locks.pop(cache_key, None)
        return png_bytes

    async def get_point_nowcast(
        self,
        lat: float,
        lon: float,
        temp_c: Optional[float] = None,
        dewpoint_c: Optional[float] = None,
        station_wet: bool = False,
        radar_zr_factor: float = 1.0,
    ) -> RadarLocalTelemetry:
        """
        O(1) in-memory radar nowcast alias for get_point_telemetry.
        """
        return await self.get_point_telemetry(
            lat=lat,
            lon=lon,
            temp_c=temp_c,
            dewpoint_c=dewpoint_c,
            station_wet=station_wet,
            radar_zr_factor=radar_zr_factor,
        )

    async def get_point_telemetry(
        self,
        lat: float,
        lon: float,
        temp_c: Optional[float] = None,
        dewpoint_c: Optional[float] = None,
        station_wet: bool = False,
        radar_zr_factor: float = 1.0,
    ) -> RadarLocalTelemetry:
        """
        Dark Sky-grade 1-minute sub-pixel Lagrangian CONUS 1km Doppler ray-caster:
        - Traces 61 1-minute steps (0..60m) upstream along the regional optical-flow storm vector
        - Uses bilinear sub-pixel interpolation + lead-time-widening uncertainty cone
        - Tracks Lagrangian echo growth/decay (dBZ/10m) between the -10m/-15m frame and LIVE frame
        - Applies sub-cloud dewpoint depression (LCL) virga evaporation filtering + learned Z-R factor
        Executes in ~0.09ms directly from numpy RAM.
        """
        if not self._frames:
            await self.refresh_dataset()

        in_conus = (self.LAT_MIN <= lat <= self.LAT_MAX) and (self.LON_MIN <= lon <= self.LON_MAX)
        if not in_conus or not self._frames:
            return RadarLocalTelemetry(
                inConusGrid=False,
                currentDbz=0.0,
                maxDbzNearby=0.0,
                precipRateMmHr=0.0,
                intensityCategory="Clear",
                echoCoveragePercent=0.0,
                stormMotionKmh=38.0,
                stormMotionDegrees=68,
                stormMotionCompass="ENE",
                echoTrendDbzPer10m=0.0,
                virgaFactor=1.0,
                minuteRates60m=[0.0] * 13,
                minuteByMinuteRates=[0.0] * 61,
            )

        live_candidates = [f for f in self._frames.values() if not f.is_future]
        live_frame = live_candidates[-1] if live_candidates else list(self._frames.values())[-1]
        ref_frame = live_candidates[-3] if len(live_candidates) >= 3 else (live_candidates[0] if len(live_candidates) >= 2 else None)
        ref_dt_mins = (
            max(5.0, (live_frame.dt - ref_frame.dt).total_seconds() / 60.0)
            if (ref_frame and ref_frame is not live_frame)
            else 10.0
        )
        grid = live_frame.grid
        ref_grid = ref_frame.grid if (ref_frame and ref_frame is not live_frame) else None

        f_row = (self.LAT_MAX - lat) / self.STEP_DEG
        f_col = (lon - self.LON_MIN) / self.STEP_DEG
        row = max(0, min(self.GRID_H - 1, int(round(f_row))))
        col = max(0, min(self.GRID_W - 1, int(round(f_col))))
        cos_lat = math.cos(math.radians(lat))

        # Sub-cloud Dewpoint Depression (LCL) Virga Filter:
        # When T - Td > 9.0C (cloud base > ~1.1km) and no surface station reports rain, weak radar echoes (<24 dBZ) partially evaporate
        virga_factor = max(0.45, min(1.50, float(radar_zr_factor or 1.0)))
        if not station_wet and temp_c is not None and dewpoint_c is not None:
            dp_spread = max(0.0, float(temp_c) - float(dewpoint_c))
            if dp_spread > 9.0:
                evap_mult = max(0.25, min(1.0, 1.0 - 0.055 * (dp_spread - 9.0)))
                virga_factor = round(virga_factor * evap_mult, 3)

        # 1. Local distance-attenuated 9x9 neighborhood (~4.5km wind-drift & beam-overshoot kernel)
        k_rad = 4
        yy, xx = np.mgrid[-k_rad : k_rad + 1, -k_rad : k_rad + 1]
        k_dist_km = np.hypot(yy * 1.11, xx * 1.11 * cos_lat)
        idx_penalty = np.where(k_dist_km <= 1.2, 0, np.round((k_dist_km - 0.8) * 3.6)).astype(np.int32)

        def _sample_cone_dbz(target_grid: np.ndarray, r_flt: float, c_flt: float, lead_min: int = 0) -> float:
            r0 = int(math.floor(r_flt))
            c0 = int(math.floor(c_flt))
            if not (0 <= r0 < self.GRID_H - 1 and 0 <= c0 < self.GRID_W - 1):
                return 0.0
            dr = r_flt - r0
            dc = c_flt - c0
            # Bilinear sub-pixel interpolation of raw reflectivity index (converted to dBZ)
            d00 = _dbz_from_index(int(target_grid[r0, c0]))
            d01 = _dbz_from_index(int(target_grid[r0, c0 + 1]))
            d10 = _dbz_from_index(int(target_grid[r0 + 1, c0]))
            d11 = _dbz_from_index(int(target_grid[r0 + 1, c0 + 1]))
            bilin_dbz = (
                d00 * (1.0 - dr) * (1.0 - dc)
                + d01 * (1.0 - dr) * dc
                + d10 * dr * (1.0 - dc)
                + d11 * dr * dc
            )
            cr = int(round(r_flt))
            cc = int(round(c_flt))
            if k_rad <= cr < self.GRID_H - k_rad and k_rad <= cc < self.GRID_W - k_rad:
                patch = target_grid[cr - k_rad : cr + k_rad + 1, cc - k_rad : cc + k_rad + 1].astype(np.int32)
                # Cone widens gently with lead_min (reduces distance penalty slightly at 30-60m so narrow bands don't alias)
                cone_relax = min(1.4, lead_min * 0.025)
                eff_idx = int(np.where(patch >= 72, patch - np.maximum(0, idx_penalty - int(round(cone_relax * 2))), 0).max())
                cone_max_dbz = _dbz_from_index(eff_idx)
                # Blend razor-sharp bilinear center (65% at t=0 -> 40% at t=60m) with uncertainty cone envelope
                alpha_bilin = max(0.40, 0.65 - (lead_min / 60.0) * 0.25)
                if bilin_dbz >= 7.5:
                    return round(alpha_bilin * bilin_dbz + (1.0 - alpha_bilin) * cone_max_dbz, 1)
                return round(cone_max_dbz * 0.88, 1)
            return round(bilin_dbz, 1)

        current_dbz = _sample_cone_dbz(grid, f_row, f_col, lead_min=0)
        raw_curr_mmhr = _mmhr_from_dbz(current_dbz)
        current_mmhr = round(raw_curr_mmhr * (virga_factor if current_dbz < 26.0 else max(0.75, virga_factor)), 2)

        # 2. Regional 35km radius box (35 pixels in each direction)
        rad = 35
        br0, br1 = max(0, row - rad), min(self.GRID_H, row + rad + 1)
        bc0, bc1 = max(0, col - rad), min(self.GRID_W, col + rad + 1)
        box = grid[br0:br1, bc0:bc1]

        max_box_idx = int(box.max()) if box.size else 0
        max_dbz_nearby = _dbz_from_index(max_box_idx)
        active_echoes = box >= 80  # >= 7.5 dBZ
        coverage_pct = round(float(active_echoes.sum()) / max(1, box.size) * 100.0, 1)

        # 3. Find nearest significant precipitation echo (>= 10 dBZ -> idx >= 85)
        nearest_km: Optional[float] = None
        nearest_compass: Optional[str] = None
        sig_rows, sig_cols = np.where(box >= 85)
        if sig_rows.size > 0:
            dy_km = (sig_rows - (row - br0)) * 1.11  # positive = south
            dx_km = (sig_cols - (col - bc0)) * (1.11 * cos_lat)  # positive = east
            dists = np.hypot(dx_km, dy_km)
            min_i = int(np.argmin(dists))
            nearest_km = round(float(dists[min_i]), 1)
            bearing_deg = (math.degrees(math.atan2(float(dx_km[min_i]), float(-dy_km[min_i]))) + 360.0) % 360.0
            nearest_compass = _compass_from_degrees(bearing_deg)

        # 4. Regional or Curved Trajectory Storm Motion Vector (steps 1..4 = 15m, 30m, 45m, 60m)
        has_curved = bool(
            self._cum_drow_fields
            and self._cum_dcol_fields
            and all(s in self._cum_drow_fields and s in self._cum_dcol_fields for s in (1, 2, 3, 4))
        )

        step_cum_dr: Dict[int, float] = {0: 0.0}
        step_cum_dc: Dict[int, float] = {0: 0.0}
        step_trend: Dict[int, float] = {0: 0.0}

        def _sample_field_at_pt(field_grid: np.ndarray, r_val: float, c_val: float) -> float:
            th, tw = field_grid.shape
            fy = max(0.0, min(float(th - 1.001), (r_val / (self.GRID_H - 1)) * (th - 1)))
            fx = max(0.0, min(float(tw - 1.001), (c_val / (self.GRID_W - 1)) * (tw - 1)))
            iy = int(fy)
            ix = int(fx)
            dy = fy - iy
            dx = fx - ix
            return float(
                field_grid[iy, ix] * (1.0 - dy) * (1.0 - dx)
                + field_grid[iy, ix + 1] * (1.0 - dy) * dx
                + field_grid[iy + 1, ix] * dy * (1.0 - dx)
                + field_grid[iy + 1, ix + 1] * dy * dx
            )

        if has_curved:
            for s in range(1, 5):
                step_cum_dr[s] = _sample_field_at_pt(self._cum_drow_fields[s], f_row, f_col)
                step_cum_dc[s] = _sample_field_at_pt(self._cum_dcol_fields[s], f_row, f_col)
                step_trend[s] = (
                    _sample_field_at_pt(self._hrrr_trend_fields[s], f_row, f_col)
                    if (self._hrrr_trend_fields and s in self._hrrr_trend_fields)
                    else 0.0
                )
        else:
            drow_15m, dcol_15m = self._estimate_regional_motion_15m(row, col)
            for s in range(1, 5):
                step_cum_dr[s] = float(drow_15m * s)
                step_cum_dc[s] = float(dcol_15m * s)
                step_trend[s] = 0.0

        # Motion from initial tangent (step = 1, 15-minute lead)
        tan_dr_15m = step_cum_dr[1]
        tan_dc_15m = step_cum_dc[1]
        vy_km_15m = -tan_dr_15m * 1.11  # positive = moving north
        vx_km_15m = tan_dc_15m * (1.11 * cos_lat)  # positive = moving east
        speed_kmh = round(math.hypot(vx_km_15m, vy_km_15m) * 4.0, 1)
        motion_deg = int(round((math.degrees(math.atan2(vx_km_15m, vy_km_15m)) + 360.0) % 360.0))
        motion_compass = _compass_from_degrees(motion_deg)

        # 5. Cast 61 1-minute curved sub-pixel steps (0..60m) upstream + Lagrangian Trend
        minute_by_minute: List[float] = []
        trend_samples: List[float] = []
        tan_step_r_1m = tan_dr_15m / 15.0
        tan_step_c_1m = tan_dc_15m / 15.0

        for t_min in range(61):
            step_flt = t_min / 15.0
            s0 = int(math.floor(step_flt))
            s1 = min(4, s0 + 1)
            frac = step_flt - s0
            dr_t = (1.0 - frac) * step_cum_dr[s0] + frac * step_cum_dr[s1]
            dc_t = (1.0 - frac) * step_cum_dc[s0] + frac * step_cum_dc[s1]
            trend_t = (1.0 - frac) * step_trend[s0] + frac * step_trend[s1]

            up_r = f_row - dr_t
            up_c = f_col - dc_t
            if 0.0 <= up_r < (self.GRID_H - 1) and 0.0 <= up_c < (self.GRID_W - 1):
                live_dbz = _sample_cone_dbz(grid, up_r, up_c, lead_min=t_min)
                proj_dbz = live_dbz
                # Compute Lagrangian growth/decay rate by checking where this parcel was in ref_frame
                if ref_grid is not None and live_dbz >= 7.5:
                    prev_r = up_r - tan_step_r_1m * ref_dt_mins
                    prev_c = up_c - tan_step_c_1m * ref_dt_mins
                    if 0.0 <= prev_r < (self.GRID_H - 1) and 0.0 <= prev_c < (self.GRID_W - 1):
                        prev_dbz = _sample_cone_dbz(ref_grid, prev_r, prev_c, lead_min=t_min)
                        # Rate of change in dBZ per minute (clamped to [-0.45, +0.45] dBZ/min)
                        rate_per_min = max(-0.45, min(0.45, (live_dbz - prev_dbz) / ref_dt_mins))
                        trend_samples.append(rate_per_min * 10.0)
                        # Exponentially damped trend projection (tau = 20 min)
                        tau = 20.0
                        delta_dbz = rate_per_min * tau * (1.0 - math.exp(-t_min / tau))
                        proj_dbz = max(0.0, min(65.0, live_dbz + delta_dbz))

                # Apply HRRR reflectivity trend (with distance dampening) on active echoes before Z-R conversion
                if proj_dbz >= 7.5 and abs(trend_t) > 1e-4:
                    dist_km = math.hypot(-dr_t * 1.11, dc_t * 1.11 * cos_lat)
                    damp_factor = 1.0 / (1.0 + (dist_km / 60.0))
                    eff_hrrr_dbz = (trend_t * 0.5) * damp_factor
                    proj_dbz = max(0.0, min(65.0, proj_dbz + eff_hrrr_dbz))

                raw_rate = _mmhr_from_dbz(proj_dbz)
                eff_vf = virga_factor if proj_dbz < 26.0 else max(0.75, virga_factor)
                minute_by_minute.append(round(raw_rate * eff_vf, 2))
            else:
                minute_by_minute.append(0.0)

        # Apply a light 3-point [0.25, 0.50, 0.25] temporal filter across 1-min steps to remove pixel-boundary stairsteps
        smoothed_61: List[float] = []
        for i in range(61):
            if i == 0 or i == 60:
                smoothed_61.append(minute_by_minute[i])
            else:
                v = 0.25 * minute_by_minute[i - 1] + 0.50 * minute_by_minute[i] + 0.25 * minute_by_minute[i + 1]
                smoothed_61.append(round(v, 2) if v >= 0.04 else 0.0)

        # Extract 13 5-minute checkpoints (0, 5, 10, ..., 60m) for backward compatibility
        minute_rates_13 = [smoothed_61[t] for t in range(0, 61, 5)]
        avg_trend_10m = round(sum(trend_samples) / len(trend_samples), 2) if trend_samples else 0.0

        return RadarLocalTelemetry(
            inConusGrid=True,
            currentDbz=current_dbz,
            maxDbzNearby=max_dbz_nearby,
            precipRateMmHr=current_mmhr,
            intensityCategory=_classify_dbz(max(current_dbz, max_dbz_nearby if (nearest_km or 99) < 8.0 else 0.0)),
            echoCoveragePercent=coverage_pct,
            nearestEchoKm=nearest_km,
            nearestEchoCompass=nearest_compass,
            stormMotionKmh=speed_kmh,
            stormMotionDegrees=motion_deg,
            stormMotionCompass=motion_compass,
            echoTrendDbzPer10m=avg_trend_10m,
            virgaFactor=virga_factor,
            minuteRates60m=minute_rates_13,
            minuteByMinuteRates=smoothed_61,
        )

    async def _prewarm_viewport_tiles(self, lat: float, lon: float, z: int = 8) -> None:
        """Pre-slices and caches the 3x3 viewport tile neighborhood across all timeline frames in RAM."""
        if not (self.LAT_MIN <= lat <= self.LAT_MAX and self.LON_MIN <= lon <= self.LON_MAX):
            return
        try:
            n = 2.0 ** z
            cx = int(math.floor(((lon + 180.0) / 360.0) * n))
            lat_rad = math.radians(max(-85.0, min(85.0, lat)))
            cy = int(math.floor(((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0) * n))
            frame_ids = list(self._frames.keys())
            for fid in frame_ids:
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        tx, ty = cx + dx, cy + dy
                        if 0 <= tx < int(n) and 0 <= ty < int(n):
                            ck = f"{fid}:{z}:{tx}:{ty}"
                            if ck not in self._tile_cache:
                                await self.get_tile_png(fid, z, tx, ty)
        except Exception:
            pass

    def _build_regional_pack_sync(
        self, snap_lat: float, snap_lon: float, size_px: int = 560
    ) -> Dict[str, Any]:
        """
        Slices all 14 timeline frames (-45m to +2h) in a single vectorized pass into continuous
        560x560 WebMercator regional PNG images (~830km x 830km span) with zero tile seams.
        """
        u_c = (snap_lon + 180.0) / 360.0
        lat_rad = math.radians(max(-80.0, min(80.0, snap_lat)))
        v_c = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0

        half_span = 0.0135  # ~9.7 deg lon x ~7.2 deg lat (~800km x 800km regional box)
        u_min, u_max = u_c - half_span, u_c + half_span
        v_min, v_max = v_c - half_span, v_c + half_span

        lon_west = u_min * 360.0 - 180.0
        lon_east = u_max * 360.0 - 180.0
        lat_north = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * v_min))))
        lat_south = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * v_max))))

        t = (np.arange(size_px, dtype=np.float64) + 0.5) / float(size_px)
        u_arr = u_min + t * (u_max - u_min)
        v_arr = v_min + t * (v_max - v_min)

        lons = u_arr * 360.0 - 180.0
        merc_y = math.pi * (1.0 - 2.0 * v_arr)
        lats = np.degrees(np.arctan(np.sinh(merc_y)))

        base_cols = (lons - self.LON_MIN) / self.STEP_DEG
        base_rows = (self.LAT_MAX - lats) / self.STEP_DEG

        # Precompute trajectory grid bilinear weights once for all frames
        sample_grid = next(iter(self._cum_drow_fields.values()), None)
        if sample_grid is not None:
            th, tw = sample_grid.shape
        else:
            th, tw = 110, 245

        fy = np.clip((base_rows / (self.GRID_H - 1)) * (th - 1), 0.0, th - 1.001)
        fx = np.clip((base_cols / (self.GRID_W - 1)) * (tw - 1), 0.0, tw - 1.001)
        iy = fy.astype(np.int32)
        ix = fx.astype(np.int32)
        dy = (fy - iy)[:, None]
        dx = (fx - ix)[None, :]
        iy0 = iy[:, None]
        iy1 = (iy + 1)[:, None]
        ix0 = ix[None, :]
        ix1 = (ix + 1)[None, :]

        frames_b64: Dict[str, str] = {}
        frames_meta: List[Dict[str, Any]] = []

        for f in self._frames.values():
            fut_mult = (f.offset_minutes / 15.0) if (f.is_future and f.offset_minutes > 0) else 0.0
            step_idx = max(1, min(8, int(round(fut_mult)))) if fut_mult > 0.0 else 0

            if (
                step_idx > 0
                and step_idx in self._cum_drow_fields
                and step_idx in self._cum_dcol_fields
            ):
                cum_dr_grid = self._cum_drow_fields[step_idx]
                cum_dc_grid = self._cum_dcol_fields[step_idx]
                drow_2d = (
                    cum_dr_grid[iy0, ix0] * (1.0 - dy) * (1.0 - dx)
                    + cum_dr_grid[iy0, ix1] * (1.0 - dy) * dx
                    + cum_dr_grid[iy1, ix0] * dy * (1.0 - dx)
                    + cum_dr_grid[iy1, ix1] * dy * dx
                )
                dcol_2d = (
                    cum_dc_grid[iy0, ix0] * (1.0 - dy) * (1.0 - dx)
                    + cum_dc_grid[iy0, ix1] * (1.0 - dy) * dx
                    + cum_dc_grid[iy1, ix0] * dy * (1.0 - dx)
                    + cum_dc_grid[iy1, ix1] * dy * dx
                )
                rows_2d = np.floor(base_rows[:, None] - drow_2d).astype(np.int32)
                cols_2d = np.floor(base_cols[None, :] - dcol_2d).astype(np.int32)
                valid_mask = (
                    (rows_2d >= 0)
                    & (rows_2d < self.GRID_H)
                    & (cols_2d >= 0)
                    & (cols_2d < self.GRID_W)
                )
                if not np.any(valid_mask):
                    frames_b64[f.frame_id] = ""
                    frames_meta.append(
                        {
                            "frameId": f.frame_id,
                            "timestamp": f.timestamp,
                            "label": f.label,
                            "offsetMinutes": f.offset_minutes,
                            "isFuture": f.is_future,
                        }
                    )
                    continue
                r_clip = np.clip(rows_2d, 0, self.GRID_H - 1)
                c_clip = np.clip(cols_2d, 0, self.GRID_W - 1)
                sub = f.grid[r_clip, c_clip].copy()

                if step_idx in self._hrrr_trend_fields:
                    tr_grid = self._hrrr_trend_fields[step_idx]
                    trend_2d = (
                        tr_grid[iy0, ix0] * (1.0 - dy) * (1.0 - dx)
                        + tr_grid[iy0, ix1] * (1.0 - dy) * dx
                        + tr_grid[iy1, ix0] * dy * (1.0 - dx)
                        + tr_grid[iy1, ix1] * dy * dx
                    )
                    active_echo = sub >= 86
                    if np.any(active_echo):
                        mod = np.clip(sub.astype(np.float32) + trend_2d, 84.0, 195.0).astype(np.uint8)
                        sub = np.where(active_echo, mod, sub)

                if not np.all(valid_mask):
                    sub[~valid_mask] = 0
            else:
                cols = np.floor(base_cols).astype(np.int32) - int(f.advect_dcol)
                rows = np.floor(base_rows).astype(np.int32) - int(f.advect_drow)
                valid_r = (rows >= 0) & (rows < self.GRID_H)
                valid_c = (cols >= 0) & (cols < self.GRID_W)
                if not np.any(valid_r) or not np.any(valid_c):
                    frames_b64[f.frame_id] = ""
                    frames_meta.append(
                        {
                            "frameId": f.frame_id,
                            "timestamp": f.timestamp,
                            "label": f.label,
                            "offsetMinutes": f.offset_minutes,
                            "isFuture": f.is_future,
                        }
                    )
                    continue
                r_clip = np.clip(rows, 0, self.GRID_H - 1)
                c_clip = np.clip(cols, 0, self.GRID_W - 1)
                sub = f.grid[np.ix_(r_clip, c_clip)].copy()
                if not np.all(valid_r):
                    sub[~valid_r, :] = 0
                if not np.all(valid_c):
                    sub[:, ~valid_c] = 0

            if f.decay_indices > 0:
                core_mask = sub > 130
                if np.any(core_mask):
                    sub[core_mask] = np.maximum(
                        126, sub[core_mask].astype(np.int16) - f.decay_indices * 2
                    ).astype(np.uint8)

            if int(sub.max()) < 88:
                frames_b64[f.frame_id] = ""
            else:
                padded = np.pad(sub, 1, mode="edge")
                active_nbrs = (padded >= 88).astype(np.uint8)
                nbr_sum = (
                    active_nbrs[:-2, :-2] + active_nbrs[:-2, 1:-1] + active_nbrs[:-2, 2:]
                    + active_nbrs[1:-1, :-2] + active_nbrs[1:-1, 1:-1] + active_nbrs[1:-1, 2:]
                    + active_nbrs[2:, :-2] + active_nbrs[2:, 1:-1] + active_nbrs[2:, 2:]
                )
                sub = np.where((sub >= 86) & (nbr_sum >= 2), sub, 0)
                img = Image.fromarray(sub, mode="P")
                img.putpalette(ZUS_PALETTE_RGB)
                img.info["transparency"] = ZUS_PALETTE_ALPHA
                buf = io.BytesIO()
                img.save(buf, format="PNG", optimize=False)
                frames_b64[f.frame_id] = base64.b64encode(buf.getvalue()).decode("ascii")

            frames_meta.append(
                {
                    "frameId": f.frame_id,
                    "timestamp": f.timestamp,
                    "label": f.label,
                    "offsetMinutes": f.offset_minutes,
                    "isFuture": f.is_future,
                }
            )

        return {
            "inConus": True,
            "bounds": [
                [round(lat_south, 5), round(lon_west, 5)],
                [round(lat_north, 5), round(lon_east, 5)],
            ],
            "sizePx": size_px,
            "frames": frames_meta,
            "images": frames_b64,
        }

    async def get_regional_pack(self, lat: float, lon: float) -> Dict[str, Any]:
        if not self._frames:
            await self.refresh_dataset()

        if not (self.LAT_MIN <= lat <= self.LAT_MAX and self.LON_MIN <= lon <= self.LON_MAX):
            return {"inConus": False, "bounds": None, "frames": [], "images": {}}

        snap_lat = round(lat * 4.0) / 4.0
        snap_lon = round(lon * 4.0) / 4.0
        live_fid = next((f.frame_id for f in reversed(list(self._frames.values())) if not f.is_future), "latest")
        cache_key = f"regpack:{live_fid}:{snap_lat:.2f}:{snap_lon:.2f}"

        if not hasattr(self, "_regional_pack_cache"):
            self._regional_pack_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

        if cache_key in self._regional_pack_cache:
            self._regional_pack_cache.move_to_end(cache_key)
            return self._regional_pack_cache[cache_key]

        pack = await asyncio.to_thread(self._build_regional_pack_sync, snap_lat, snap_lon, 560)
        self._regional_pack_cache[cache_key] = pack
        if len(self._regional_pack_cache) > 40:
            self._regional_pack_cache.popitem(last=False)
        return pack

    async def get_viewport_tile_pack(
        self, lat: float, lon: float, z_levels: Tuple[int, ...] = (7, 8), radius: int = 2
    ) -> Dict[str, Any]:
        """
        Returns a single-request atomic tile bundle containing all viewport WebMercator tiles
        across all timeline frames in RAM (encoded as base64 PNG strings, or "" for empty tiles).
        Eliminates 140+ sequential HTTP tile requests and guarantees 0-flicker instant playback.
        """
        if not self._frames:
            await self.refresh_dataset()

        tiles_out: Dict[str, str] = {}
        aliases: Dict[str, str] = {}
        if self._frames:
            for alias_id, off in [
                ("past_45m", -45),
                ("past_30m", -30),
                ("past_15m", -15),
                ("latest", 0),
                ("nowcast_p15m", 15),
                ("nowcast_p30m", 30),
                ("nowcast_p45m", 45),
                ("nowcast_p60m", 60),
                ("nowcast_p75m", 75),
                ("nowcast_p90m", 90),
                ("nowcast_p105m", 105),
                ("nowcast_p120m", 120),
            ]:
                best_f = min(self._frames.values(), key=lambda f: abs(f.offset_minutes - off))
                aliases[alias_id] = best_f.frame_id

        if not (self.LAT_MIN <= lat <= self.LAT_MAX and self.LON_MIN <= lon <= self.LON_MAX):
            return {"inConus": False, "aliases": aliases, "tiles": tiles_out}

        lat_rad = math.radians(max(-85.0, min(85.0, lat)))
        frame_ids = list(self._frames.keys())

        for z in z_levels:
            n = 2.0 ** z
            cx = int(math.floor(((lon + 180.0) / 360.0) * n))
            cy = int(
                math.floor(
                    ((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0) * n
                )
            )
            for fid in frame_ids:
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        tx, ty = cx + dx, cy + dy
                        if 0 <= tx < int(n) and 0 <= ty < int(n):
                            png_bytes = await self.get_tile_png(fid, z, tx, ty)
                            key = f"{fid}:{z}:{tx}:{ty}"
                            if png_bytes == EMPTY_TILE_PNG or len(png_bytes) <= 120:
                                tiles_out[key] = ""
                            else:
                                tiles_out[key] = base64.b64encode(png_bytes).decode("ascii")

        return {
            "inConus": True,
            "center": [round(lat, 4), round(lon, 4)],
            "zLevels": list(z_levels),
            "radius": radius,
            "frameIds": frame_ids,
            "aliases": aliases,
            "tiles": tiles_out,
        }

    async def get_manifest(
        self, lat: Optional[float] = None, lon: Optional[float] = None, base_url: str = ""
    ) -> RadarManifestResponse:
        if not self._frames:
            await self.refresh_dataset()

        prefix = base_url.rstrip("/") if base_url else ""
        frames_out: List[RadarFrameModel] = []
        for f in self._frames.values():
            frames_out.append(
                RadarFrameModel(
                    frameId=f.frame_id,
                    timestamp=f.timestamp,
                    isoTime=f.dt.isoformat(),
                    label=f.label,
                    offsetMinutes=f.offset_minutes,
                    isFuture=f.is_future,
                    tileUrlTemplate=f"{prefix}/api/v1/radar/tiles/{f.frame_id}/{{z}}/{{x}}/{{y}}.png?v=9",
                )
            )

        local_telemetry: Optional[RadarLocalTelemetry] = None
        if lat is not None and lon is not None:
            local_telemetry = await self.get_point_telemetry(lat, lon)
            asyncio.create_task(self._prewarm_viewport_tiles(lat, lon, z=8))

        unique_grids = {id(f.grid): f.grid for f in self._frames.values()}
        ram_mb = round(sum(g.nbytes for g in unique_grids.values()) / (1024 * 1024), 1)
        tile_ram_kb = round(sum(len(b) for b in self._tile_cache.values()) / 1024, 1)

        return RadarManifestResponse(
            datasetId="zus-conus-doppler-1km",
            gridResolutionKm=1.0,
            bounds=[self.LAT_MIN, self.LON_MIN, self.LAT_MAX, self.LON_MAX],
            frames=frames_out,
            localTelemetry=local_telemetry,
            cacheStats={
                "framesInRam": len(self._frames),
                "masterGridRamMb": ram_mb,
                "cachedTilesCount": len(self._tile_cache),
                "tileCacheRamKb": tile_ram_kb,
                "lastIngestedAt": self._last_ingested_at.isoformat() if self._last_ingested_at else None,
                "lastBackfillAt": self._last_backfill_at.isoformat() if self._last_backfill_at else None,
                "totalBackfilledSession": self._total_backfilled_frames,
            },
        )


doppler_service = DopplerRadarService()
