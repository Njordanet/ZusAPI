from typing import Optional
from fastapi import APIRouter, Query, Request, Response
from app.models.weather import RadarManifestResponse, RadarLocalTelemetry
from app.services.doppler_service import doppler_service

router = APIRouter()


@router.get(
    "/radar/manifest",
    response_model=RadarManifestResponse,
    summary="Get Züs CONUS 1km Doppler Radar timeline frames, tile URLs & local telemetry",
)
async def get_radar_manifest(
    request: Request,
    lat: Optional[float] = Query(None, ge=-90.0, le=90.0, description="Optional latitude for instant point telemetry"),
    lon: Optional[float] = Query(None, ge=-180.0, le=180.0, description="Optional longitude for instant point telemetry"),
):
    base_url = str(request.base_url).rstrip("/")
    return await doppler_service.get_manifest(lat=lat, lon=lon, base_url=base_url)


@router.get(
    "/radar/point",
    response_model=RadarLocalTelemetry,
    summary="Instant O(1) in-memory CONUS 1km Doppler point reflectivity & 60-min Marshall-Palmer nowcast",
)
async def get_radar_point(
    lat: float = Query(..., ge=-90.0, le=90.0, description="Latitude"),
    lon: float = Query(..., ge=-180.0, le=180.0, description="Longitude"),
):
    return await doppler_service.get_point_telemetry(lat=lat, lon=lon)


@router.post(
    "/radar/backfill",
    summary="Trigger historical offline gap detection & IEM NEXRAD archive backfill",
)
async def trigger_radar_backfill(
    lookback_hours: int = Query(24, ge=1, le=72, description="Hours to scan for offline gaps"),
    max_batch: int = Query(24, ge=1, le=96, description="Max missing frames to backfill in this pass"),
):
    await doppler_service.refresh_dataset()
    result = await doppler_service.backfill_offline_gaps(
        lookback_hours=lookback_hours, max_batch=max_batch
    )
    return result


@router.get(
    "/radar/tiles/{frame_id}/{z}/{x}/{y}.png",
    summary="Serve 8-bit Züs Storm Palette WebMercator Doppler tile (<0.4ms from RAM)",
    responses={200: {"content": {"image/png": {}}}},
)
async def get_radar_tile(
    frame_id: str,
    z: int,
    x: int,
    y: int,
):
    png_bytes = await doppler_service.get_tile_png(frame_id=frame_id, z=z, x=x, y=y)
    # Content-addressed frame_ids are immutable once recorded; "latest" refreshes every 300s
    cache_header = (
        "public, max-age=300"
        if frame_id == "latest"
        else "public, max-age=86400, immutable"
    )
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "Cache-Control": cache_header,
            "Access-Control-Allow-Origin": "*",
            "X-Zus-Radar-Engine": "8bit-Indexed-Numpy-ZeroCopy",
        },
    )
