from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path

def _get_version() -> str:
    version_file = Path(__file__).resolve().parent.parent / "VERSION"
    if version_file.exists():
        return version_file.read_text(encoding="utf-8").strip().lstrip("v")
    return "1.3.0"

class Settings(BaseSettings):
    app_name: str = "ZüsAPI"
    app_version: str = _get_version()
    app_env: str = "production"
    debug: bool = False
    
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Storage / Data directory for datasets & cache
    data_dir: Path = Path("/app/data")
    
    # PostgreSQL + PostGIS connection URL
    database_url: str = "postgresql://zus:zus_olympus_pass@zus-db:5432/zus_telemetry"
    
    # Upstream Multi-Model Meteorological Providers
    open_meteo_forecast_url: str = "https://api.open-meteo.com/v1/forecast"
    open_meteo_air_quality_url: str = "https://air-quality-api.open-meteo.com/v1/air-quality"
    open_meteo_geocoding_url: str = "https://geocoding-api.open-meteo.com/v1/search"
    
    # NWS (National Weather Service) & NOAA Aviation Weather Center (METAR + CWOP)
    nws_base_url: str = "https://api.weather.gov"
    noaa_metar_url: str = "https://aviationweather.gov/api/data/metar"
    
    # Optional Nowcasting API Keys
    pirate_weather_api_key: str = ""
    tomorrow_io_api_key: str = ""

    # Accounts & Authentication (Supabase)
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""
    supabase_jwt_secret: str = ""
    zus_admin_api_key: str = ""
    founder_auto_cutoff_iso: str = ""

    # StoreKit 2 & App Store Server Notifications
    apple_bundle_id: str = "com.zuscast.mobile"
    apple_issuer_id: str = ""
    apple_key_id: str = ""
    apple_private_key_pem: str = ""

    # Google Play Billing & Real-Time Developer Notifications
    google_play_package_name: str = "com.zuscast.mobile"
    google_service_account_json: str = ""
    google_rtdn_webhook_token: str = ""
    
    # Cache & Verification Loop Settings
    cache_ttl_seconds: int = 600  # 10 minutes for neighborhood/city cache sharing
    verification_interval_seconds: int = 7200  # 2 hours background verification pass
    standard_lapse_rate_c_per_km: float = 6.5  # Standard adiabatic lapse rate (-6.5°C / 1000m)
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
