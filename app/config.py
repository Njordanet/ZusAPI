from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path

class Settings(BaseSettings):
    app_name: str = "ZüsAPI"
    app_version: str = "2.1.0"
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
    
    # Cache & Verification Loop Settings
    cache_ttl_seconds: int = 300  # 5 minutes for neighborhood/city cache sharing
    verification_interval_seconds: int = 1800  # 30 minutes background verification pass
    standard_lapse_rate_c_per_km: float = 6.5  # Standard adiabatic lapse rate (-6.5°C / 1000m)
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
