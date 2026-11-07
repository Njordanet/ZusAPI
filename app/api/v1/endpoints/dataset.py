from fastapi import APIRouter, Query
from app.services.dataset_service import dataset_service
from app.services.db import db_manager

router = APIRouter()


@router.get("/dataset/inventory", summary="Get PostGIS database stats, active cells, and dataset catalog")
async def get_datasets():
    """
    Returns live PostGIS database statistics (tracked cells, indexed PWS/METAR stations,
    forecast snapshots, verified observations) along with dataset catalog info.
    """
    db_stats = await db_manager.get_database_stats()
    file_inventory = dataset_service.get_dataset_inventory()
    return {
        "database": db_stats,
        "catalog": file_inventory,
    }


@router.get("/dataset/leaderboard", summary="Get accuracy leaderboard, RLS horizon weights & learned harmonic MOS bias for a grid cell")
async def get_model_leaderboard(
    lat: float = Query(40.0856, ge=-90.0, le=90.0),
    lon: float = Query(22.3585, ge=-180.0, le=180.0),
):
    cell_id = db_manager.make_cell_id(lat, lon)
    weights = await db_manager.get_model_weights(cell_id)
    rls_meta = await db_manager.get_cell_rls_metadata(cell_id)
    return {
        "cell_id": cell_id,
        "region_mode": "US_HYPERLOCAL" if db_manager.is_us_region(lat, lon) else "GLOBAL_ENSEMBLE",
        "rls_state": rls_meta,
        "models": weights,
    }
