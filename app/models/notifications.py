from pydantic import BaseModel, Field, model_validator
from typing import Optional, Literal, Dict, Any, List


PrecipSensitivity = Literal["any", "moderate", "heavy"]
TempUnitLiteral = Literal["F", "C"]
WindUnitLiteral = Literal["mph", "kmh", "kt", "ms"]

_TIER_A_B_SETTINGS_KEYS = (
    "notifications_enabled",
    "dnd_enabled",
    "dnd_start_time",
    "dnd_end_time",
    "dnd_severe_breakthrough",
    "precip_alerts_enabled",
    "precip_sensitivity",
    "severe_alerts_enabled",
)


class PushSubscriptionRegisterRequest(BaseModel):
    device_id: str = Field(..., description="Unique client device identifier")
    expo_push_token: Optional[str] = Field(None, description="Expo push token (e.g. ExponentPushToken[...])")
    lat: float = Field(..., description="Client latitude")
    lon: float = Field(..., description="Client longitude")
    timezone: str = Field("UTC", description="IANA Timezone (e.g. America/New_York)")
    notifications_enabled: bool = Field(True, description="Master toggle for notifications")
    dnd_enabled: bool = Field(False, description="Do Not Disturb window enabled")
    dnd_start_time: str = Field("22:00", description="DND start time (HH:mm)")
    dnd_end_time: str = Field("07:00", description="DND end time (HH:mm)")
    dnd_severe_breakthrough: bool = Field(True, description="Allow severe alerts during DND")
    precip_alerts_enabled: bool = Field(True, description="Enable precipitation alerts")
    precip_sensitivity: PrecipSensitivity = Field("any", description="Precipitation sensitivity: any, moderate, heavy")
    severe_alerts_enabled: bool = Field(True, description="Enable severe safety alerts")
    city: str = Field("Current Location", description="Display location name")
    temp_unit: TempUnitLiteral = Field("F", description="Preferred temperature unit: F or C")
    wind_unit: WindUnitLiteral = Field("mph", description="Preferred wind speed unit")
    morning_briefing_enabled: bool = Field(False, description="Enable server-dispatched morning briefing")
    morning_briefing_time: str = Field("07:30", description="Morning briefing local time (HH:mm)")
    evening_outlook_enabled: bool = Field(False, description="Enable server-dispatched evening outlook")
    evening_outlook_time: str = Field("20:00", description="Evening outlook local time (HH:mm)")
    freeze_alert_enabled: bool = Field(True, description="Enable freeze/frost advisory")
    aqi_alert_enabled: bool = Field(True, description="Enable AQI spike alert")
    wind_alert_threshold_mph: int = Field(35, description="Wind gust alert threshold in mph (0=off)")

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_payload(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        d = dict(data)
        if "lat" not in d and "latitude" in d:
            d["lat"] = d["latitude"]
        if "lon" not in d and "longitude" in d:
            d["lon"] = d["longitude"]
        if "expo_push_token" not in d and "push_token" in d:
            d["expo_push_token"] = d["push_token"]
        if isinstance(d.get("expo_push_token"), str):
            tok = d["expo_push_token"].strip()
            d["expo_push_token"] = tok if (tok.startswith("ExponentPushToken[") or tok.startswith("ExpoPushToken[")) else None
        if "temp_unit" in d and d["temp_unit"] is not None:
            d["temp_unit"] = "C" if str(d["temp_unit"]).strip().upper() == "C" else "F"
        if "wind_unit" in d and d["wind_unit"] is not None:
            wu = str(d["wind_unit"]).strip()
            d["wind_unit"] = wu if wu in ("mph", "kmh", "kt", "ms") else "mph"
        if isinstance(d.get("settings"), dict):
            for k in _TIER_A_B_SETTINGS_KEYS:
                if k in d["settings"] and k not in d:
                    d[k] = d["settings"][k]
        return d


class PushSubscriptionResponse(BaseModel):
    device_id: str
    expo_push_token: Optional[str] = None
    lat: float
    lon: float
    cell_lat: float
    cell_lon: float
    timezone: str
    notifications_enabled: bool
    dnd_enabled: bool
    dnd_start_time: str
    dnd_end_time: str
    dnd_severe_breakthrough: bool
    precip_alerts_enabled: bool
    precip_sensitivity: str
    severe_alerts_enabled: bool
    city: str = "Current Location"
    temp_unit: str = "F"
    wind_unit: str = "mph"
    morning_briefing_enabled: bool = False
    morning_briefing_time: str = "07:30"
    evening_outlook_enabled: bool = False
    evening_outlook_time: str = "20:00"
    freeze_alert_enabled: bool = True
    aqi_alert_enabled: bool = True
    wind_alert_threshold_mph: int = 35
    server_briefings_supported: bool = True
    updated_at: str


class PushSubscriptionUnregisterRequest(BaseModel):
    device_id: str


class NotificationStatsResponse(BaseModel):
    total_subscriptions: int
    active_subscriptions: int
    unique_cells: int
    cell_states: Dict[str, int]
    last_run_at: Optional[str] = None
    last_run_duration_ms: Optional[float] = None
    recent_dispatches: int = 0
    recent_briefing_dispatches: int = 0
    recent_suppressed_dnd: int = 0
    recent_suppressed_cooldown: int = 0
    recent_suppressed_severe_active: int = 0
    status: str = "operational"
