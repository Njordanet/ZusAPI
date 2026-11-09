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
            56: cls.SNOW,
            57: cls.SNOW,
            61: cls.RAIN,
            63: cls.RAIN,
            65: cls.HEAVY_RAIN,
            66: cls.SNOW,
            67: cls.SNOW,
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
    weightShort: Optional[float] = None
    weightMed: Optional[float] = None
    weightExt: Optional[float] = None
    accuracyPercent: float
    maeCelsius: float
    biasCelsius: float
    diurnalBiasNowCelsius: Optional[float] = None
    maeWindKmh: Optional[float] = None
    brierPrecip: Optional[float] = None
    samplesVerified: int
    currentTempCelsius: Optional[float] = None
    currentPrecipProb: Optional[int] = None
    currentWindKmh: Optional[float] = None


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
    reliabilityScore: Optional[float] = None
    residualBiasCelsius: Optional[float] = None


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
    echoTrendDbzPer10m: float = 0.0
    virgaFactor: float = 1.0
    minuteRates60m: List[float] = []
    minuteByMinuteRates: List[float] = []


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
    pm25: Optional[float] = None
    dewPoint: Optional[float] = None
    comfortLevel: Optional[str] = None
    nowcastSummary: Optional[str] = None
    windGust: Optional[float] = None
    cloudCover: Optional[int] = None
    precipIntensityMmHr: float = 0.0
    observedWeatherCode: Optional[int] = None
    capeJkg: Optional[float] = None
    liftedIndex: Optional[float] = None
    wetBulbTemperature: Optional[float] = None
    precipType: Optional[str] = None
    ozone: Optional[float] = None
    isDay: Optional[bool] = None
    shortwaveRadiation: Optional[float] = None
    vpdKpa: Optional[float] = None
    cloudCoverLow: Optional[int] = None
    cloudCoverMid: Optional[int] = None
    cloudCoverHigh: Optional[int] = None
    freezingLevelMeters: Optional[float] = None
    treePollen: Optional[float] = None
    grassPollen: Optional[float] = None
    weedPollen: Optional[float] = None


class HourlyForecastModel(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    time: datetime
    temperature: float
    tempMin: Optional[float] = None
    tempMax: Optional[float] = None
    feelsLike: Optional[float] = None
    feelsMin: Optional[float] = None
    feelsMax: Optional[float] = None
    condition: WeatherCondition
    wmoCode: int = 0
    precipitationChance: int
    precipProbMin: Optional[int] = None
    precipProbMax: Optional[int] = None
    precipMm: float = 0.0
    precipMmMin: Optional[float] = None
    precipMmMax: Optional[float] = None
    windSpeed: float = 0.0
    windMin: Optional[float] = None
    windMax: Optional[float] = None
    windGusts: Optional[float] = None
    gustMin: Optional[float] = None
    gustMax: Optional[float] = None
    windDirection: Optional[int] = None
    humidity: Optional[int] = None
    uvIndex: Optional[float] = None
    uvMin: Optional[float] = None
    uvMax: Optional[float] = None
    dewPoint: Optional[float] = None
    dewMin: Optional[float] = None
    dewMax: Optional[float] = None
    cloudCover: Optional[int] = None
    cloudMin: Optional[int] = None
    cloudMax: Optional[int] = None
    visibilityKm: Optional[float] = None
    visMin: Optional[float] = None
    visMax: Optional[float] = None
    pressure: Optional[float] = None
    pressMin: Optional[float] = None
    pressMax: Optional[float] = None
    capeJkg: Optional[float] = None
    wetBulbTemperature: Optional[float] = None
    precipType: Optional[str] = None
    snowAccumulationCm: Optional[float] = None
    isDay: Optional[bool] = None
    shortwaveRadiation: Optional[float] = None
    vpdKpa: Optional[float] = None
    cloudCoverLow: Optional[int] = None
    cloudCoverMid: Optional[int] = None
    cloudCoverHigh: Optional[int] = None
    freezingLevelMeters: Optional[float] = None


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
    precipType: Optional[str] = None
    snowAccumulationCm: Optional[float] = None
    snowToLiquidRatio: Optional[float] = None


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
    timezone: str = "UTC"
    utcOffsetSeconds: int = 0
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


# Backwards-compatible aliases
WeatherData = WeatherDataModel
HourlyForecast = HourlyForecastModel
DailyForecast = DailyForecastModel
