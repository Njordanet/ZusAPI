import os
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime, timezone
from app.config import settings

class DatasetService:
    """
    Manages custom meteorological datasets, historical observation dumps,
    radar tiles, or lightning strike logs stored on the Panda host.
    """
    def __init__(self, data_dir: Path = settings.data_dir):
        self.data_dir = data_dir
        self._ensure_storage()

    def _ensure_storage(self):
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            datasets_meta = self.data_dir / "datasets_meta.json"
            if not datasets_meta.exists():
                with open(datasets_meta, "w") as f:
                    json.dump({
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "datasets": [
                            {
                                "id": "olympus-historical",
                                "name": "Mount Olympus Historical Meteorological Archive",
                                "type": "timeseries_csv",
                                "records": 0,
                                "status": "initialized",
                            },
                            {
                                "id": "global-lightning-mesh",
                                "name": "Global Lightning Telemetry & Strike Mesh",
                                "type": "geospatial_geojson",
                                "records": 0,
                                "status": "ready_for_ingestion",
                            }
                        ]
                    }, f, indent=2)
        except Exception as e:
            # Fallback if running outside docker or permissions differ
            pass

    def get_dataset_inventory(self) -> Dict[str, Any]:
        meta_file = self.data_dir / "datasets_meta.json"
        if meta_file.exists():
            try:
                with open(meta_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "storage_path": str(self.data_dir),
            "datasets": [],
            "status": "ready"
        }

dataset_service = DatasetService()
