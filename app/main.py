import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager
from app.config import settings
from app.api.v1.router import api_router
from app.api.v1.endpoints import standard
from app.services.db import db_manager
from app.services.weather_provider import weather_service
from app.services.doppler_service import doppler_service
from app.services.http_client import http_pool


async def _background_doppler_loop():
    """
    Refreshes the CONUS 1km Master Doppler Reflectivity Grid & +30m Nowcast Cube every 5 minutes,
    and automatically detects & backfills any historical gaps from when the server was offline.
    """
    await asyncio.sleep(1)
    while True:
        try:
            await doppler_service.refresh_dataset()
            await doppler_service.backfill_offline_gaps(lookback_hours=24, max_batch=16)
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"⚠️ Doppler ingestion/backfill loop warning: {e}")
        await asyncio.sleep(300)


async def _background_verification_loop():
    """
    Continuously refreshes active "Follow-the-User" grid cells every 30 minutes
    so panda captures forecast snapshots and grades models against live PWS/METAR stations 24/7.
    """
    await asyncio.sleep(5)
    while True:
        try:
            if db_manager.pool:
                async with db_manager.pool.acquire() as conn:
                    active_cells = await conn.fetch(
                        """
                        SELECT city_name, latitude, longitude
                        FROM tracked_cells
                        WHERE last_requested_at >= NOW() - INTERVAL '7 days'
                        ORDER BY last_requested_at DESC
                        LIMIT 15;
                        """
                    )
                for c in active_cells:
                    try:
                        await weather_service.get_forecast(
                            latitude=float(c["latitude"]),
                            longitude=float(c["longitude"]),
                            city_name=str(c["city_name"]),
                            sync_verify=True,
                        )
                        await asyncio.sleep(2)
                    except Exception:
                        pass
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"⚠️ Background verification loop warning: {e}")

        await asyncio.sleep(settings.verification_interval_seconds)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"⚡️ {settings.app_name} v{settings.app_version} starting up on {settings.host}:{settings.port}...")
    http_pool.get_client()
    await db_manager.connect()
    doppler_task = asyncio.create_task(_background_doppler_loop())
    bg_task = asyncio.create_task(_background_verification_loop())
    yield
    doppler_task.cancel()
    bg_task.cancel()
    await http_pool.close()
    await db_manager.disconnect()
    print(f"⚡️ {settings.app_name} shutting down...")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Foundational Multi-Model Consensus, Dynamic Accuracy Weighting, "
        "& Hyperlocal Station Assimilation Engine for Züs Weather & Universal Weather Clients."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_private_network_access_header(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    response.headers["Access-Control-Allow-Origin"] = "*"
    if request.url.path.startswith("/api/v1/radar/tiles/") or request.url.path.startswith("/v2/radar/"):
        if "vary" in response.headers:
            del response.headers["vary"]
    return response


app.include_router(api_router, prefix="/api/v1")
app.include_router(standard.router)


@app.get("/", summary="Root Index")
async def root():
    return {
        "message": f"Welcome to {settings.app_name} v{settings.app_version} - Universal Multi-Model Consensus & Hyperlocal Core",
        "docs": "/docs",
        "health": "/api/v1/health",
        "inventory": "/api/v1/dataset/inventory",
        "standard_endpoints": {
            "zus_native": "/api/v1/weather?lat={lat}&lon={lon}&units={metric|imperial|si}&exclude={blocks}",
            "darksky_pirateweather": "/forecast/{api_key}/{lat},{lon}",
            "darksky_keyless": "/api/v1/darksky/{lat},{lon}",
            "open_meteo_v1": "/v1/forecast?latitude={lat}&longitude={lon}",
            "owm_onecall_3_0": "/data/3.0/onecall?lat={lat}&lon={lon}&units={metric|imperial|standard}",
            "owm_current_2_5": "/data/2.5/weather?lat={lat}&lon={lon}&units={metric|imperial|standard}",
            "rainviewer_manifest": "/public/weather-maps.json",
            "tilejson_3_0": "/api/v1/radar/tilejson.json",
            "xyz_radar_tiles": "/api/v1/radar/tiles/{frame_id}/{z}/{x}/{y}.png",
        },
    }
