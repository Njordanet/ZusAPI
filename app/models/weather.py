from pydantic import BaseModel, Field
from datetime import datetime, timezone
from typing import List, Optional
from enum import Enum
import uuid


class WeatherCondition(str, Enum):
    CLEAR = "Clear"
    PARTLY_CLOUDY = "Partly Cloudy"
    OVERCAST = "Overcast"
    FOGGY = "Foggy"
    DRIZZLE = "Drizzle"
    RAIN = "Rain"
    HEAVY_RAIN = "Heavy Rain"
    SNOW = "Snow"
    THUNDERSTORM = "Thunderstorm"
    SEVERE_THUNDERSTORM = "Severe Thunderstorm"
    UNKNOWN = "Unknown"

    @classmethod
    def from_wmo_code(cls, code: int) -> "WeatherCondition":
        mapping = {
            0: cls.CLEAR,
            1: cls.PARTLY_CLOUDY,
            2: cls.PARTLY_CLOUDY,
            3: cls.OVERCAST,
            45: cls.FOGGY,
            48: cls.FOGGY,
            51: cls.DRIZZLE,
            53: cls.DRIZZLE,
            55: cls.DRIZZLE,
            61: cls.RAIN,
            63: cls.RAIN,
            65: cls.HEAVY_RAIN,
            71: cls.SNOW,
            73: cls.SNOW,
            75: cls.SNOW,
            77: cls.SNOW,
            80: cls.HEAVY_RAIN,
            81: cls.HEAVY_RAIN,
            82: cls.HEAVY_RAIN,
            85: cls.SNOW,
            86: cls.SNOW,
            95: cls.THUNDERSTORM,
            96: cls.SEVERE_THUNDERSTORM,
            99: cls.SEVERE_THUNDERSTORM,
        }
        return mapping.get(code, cls.CLEAR)


class ModelWeightDetail(BaseModel):
    modelName: str
    weight: float
    accuracyPercent: float
    maeCelsius: float
    biasCelsius: float
    samplesVerified: int


class HyperlocalStationReading(BaseModel):
    stationId: str
    stationName: str
    network: str  # e.g., 'CWOP_PWS', 'NWS_METAR', 'GLOBAL_METAR'
    distanceKm: float
    elevationMeters: float
    rawTempCelsius: float
    elevationAdjustedTempCelsius: float
    humidity: Optional[int] = None
    windSpeedKmh: Optional[float] = None
    windGustKmh: Optional[float] = None
    pressureHpa: Optional[float] = None
    visibilityKm: Optional[float] = None
    presentWeather: Optional[str] = None
    qcPassed: bool = True
    zScore: float = 0.0


class WeatherAlertModel(BaseModel):
    id: str
    event: str
    headline: str
    severity: str
    urgency: str
    description: Optional[str] = None
    expires: Optional[str] = None


class NowcastPoint(BaseModel):
    timeOffsetMinutes: int
    precipProbability: int
    precipIntensityMmHr: float


class NowcastSummary(BaseModel):
    summaryText: str
    source: str
    points: List[NowcastPoint] = []


class RadarFrameModel(BaseModel):
    frameId: str
    timestamp: int
    isoTime: str
    label: str
    offsetMinutes: int
    isFuture: bool = False
    tileUrlTemplate: str


class RadarLocalTelemetry(BaseModel):
    inConusGrid: bool = True
    currentDbz: float = 0.0
    maxDbzNearby: float = 0.0
    precipRateMmHr: float = 0.0
    intensityCategory: str = "Clear"
    echoCoveragePercent: float = 0.0
    nearestEchoKm: Optional[float] = None
    nearestEchoCompass: Optional[str] = None
    stormMotionKmh: float = 0.0
    stormMotionDegrees: int = 0
    stormMotionCompass: str = "E"
    minuteRates60m: List[float] = []


class RadarManifestResponse(BaseModel):
    datasetId: str = "zus-conus-doppler-1km"
    gridResolutionKm: float = 1.0
    bounds: List[float] = [23.0, -126.0, 50.0, -65.0]  # [south, west, north, east]
    frames: List[RadarFrameModel]
    localTelemetry: Optional[RadarLocalTelemetry] = None
    cacheStats: dict = {}
    generatedAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WeatherDataModel(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    cityName: str
    temperature: float
    feelsLike: float
    condition: WeatherCondition
    tempMin: float
    tempMax: float
    humidity: int
    windSpeed: float
    windDirection: int
    uvIndex: float
    airPressure: float
    visibility: float = 10.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Pro-Tier Hyperlocal & Multi-Model Consensus Telemetry
    consensusScore: int = 95
    dominantModel: str = "ECMWF + HRRR Blend"
    localStationName: Optional[str] = None
    localStationDistanceKm: Optional[float] = None
    localStationWx: Optional[str] = None
    elevationMeters: Optional[float] = None
    usAqi: Optional[int] = None
    aqiCategory: Optional[str] = None
    dewPoint: Optional[float] = None
    comfortLevel: Optional[str] = None
    nowcastSummary: Optional[str] = None
    windGust: Optional[float] = None
    cloudCover: Optional[int] = None
    precipIntensityMmHr: float = 0.0
    observedWeatherCode: Optional[int] = None


class HourlyForecastModel(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    time: datetime
    temperature: float
    feelsLike: Optional[float] = None
    condition: WeatherCondition
    wmoCode: int = 0
    precipitationChance: int
    precipMm: float = 0.0
    windSpeed: float = 0.0


class DailyForecastModel(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    date: datetime
    condition: WeatherCondition
    wmoCode: int = 0
    minTemperature: float
    maxTemperature: float
    precipitationChance: int
    precipSumMm: float = 0.0
    uvIndexMax: float = 5.0
    windSpeedMax: float = 10.0
    windDirectionDominant: int = 180
    sunrise: Optional[str] = None
    sunset: Optional[str] = None


class WeatherResponse(BaseModel):
    current: WeatherDataModel
    hourly: List[HourlyForecastModel]
    daily: List[DailyForecastModel]
    modelWeights: List[ModelWeightDetail] = []
    nearbyStations: List[HyperlocalStationReading] = []
    alerts: List[WeatherAlertModel] = []
    nowcast: Optional[NowcastSummary] = None
    dopplerTelemetry: Optional[RadarLocalTelemetry] = None
    regionMode: str = "US_HYPERLOCAL"
    source: str = "ZüsAPI / Panda Consensus Engine"
    cached: bool = False
    generatedAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LightningStrike(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    latitude: float
    longitude: float
    distanceKm: float
    bearingDegrees: float
    intensityKa: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LightningTelemetryResponse(BaseModel):
    strikesCount24h: int
    nearestStrike: Optional[LightningStrike] = None
    stormThreatLevel: str
    capeJkg: float = 0.0
    liftedIndex: float = 0.0
    strikes: List[LightningStrike] = []
