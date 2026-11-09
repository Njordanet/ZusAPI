from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict, Any, List


PrecipSensitivity = Literal["any", "moderate", "heavy"]


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
    recent_suppressed_dnd: int = 0
    recent_suppressed_cooldown: int = 0
    recent_suppressed_severe_active: int = 0
    status: str = "operational"
