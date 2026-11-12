# Changelog

All notable changes to this service will be documented in this file.

## [v1.5.1] - 2026-11-11

### Local Calendar-Day High/Low & Precipitation Alignment
- **Location-Local Midnight-to-Midnight Daily Aggregation**:
  - Updated 10-day daily temperature highs, lows, precipitation probabilities, and totals to aggregate strictly within each location's local calendar day (`00:00–23:59` local time), preventing evening UTC midnight rollovers from shifting daily forecasts by a day.
  - Anchored daily forecast timestamps to noon on each local calendar date and reconciled daily highs, lows, and precipitation chances against that calendar day's hourly forecast curve and current conditions.

## [v1.5.0] - 2026-11-11

### Surface-Calibrated 2-Minute HD Radar & 10-Hour Forecast Horizon
- **Surface-Verified 2-Minute Precipitation Radar**:
  - Upgraded radar ingestion to a 2-minute refresh cadence with surface-level precipitation verification, eliminating high-altitude evaporating virga and false echoes over dry ground.
  - Added sub-pixel contour anti-aliasing across regional radar packs for smooth, organic precipitation boundaries.
- **Extended 10-Hour Future Radar Timeline**:
  - Expanded the radar timeline to 28 frames (`-45m` to `+10h`), seamlessly blending short-range storm motion into extended-range high-resolution simulated precipitation out to 10 hours.

## [v1.4.2] - 2026-11-11

### Clear-Air Radar Static & Urban Clutter Suppression
- **Radar Reflectivity & Spatial Continuity Filtering**:
  - Raised the minimum precipitation reflectivity floor and added a 5×5 km spatial continuity filter to reject non-precipitating high-cloud echoes and isolated urban building, bridge, and wind-turbine reflections during morning temperature inversions.
- **Surface Station & Cloud Ceiling Cross-Checking**:
  - Cross-checks weak radar returns against nearby airport surface observations, low-level cloud cover, and forecast models so dry overcast skies are never misclassified as light rain.

## [v1.4.1] - 2026-11-09

### Location-Local Sunrise/Sunset Ephemeris & Extended Atmospheric Telemetry (`app/models/weather.py` & `app/services/weather_provider.py`)
- **Location-Local Sunrise/Sunset Conversion & NOAA Ephemeris Fallback (`app/services/weather_provider.py`)**:
  - Fixed UTC dislocation in daily `sunrise`/`sunset` strings by shifting Open-Meteo UTC timestamps by `utc_offset_seconds` (`dt_utc + timedelta(seconds=utc_offset_sec)`), indexing by local calendar date, and validating/backfilling with `calc_local_sunrise_sunset_iso` (NOAA solar ephemeris equations).
- **Extended Current & Hourly Telemetry (`app/models/weather.py` & `app/services/weather_provider.py`)**:
  - Added `isDay`, `shortwaveRadiation`, `vpdKpa`, `cloudCoverLow`, `cloudCoverMid`, `cloudCoverHigh`, `freezingLevelMeters`, `treePollen`, `grassPollen`, and `weedPollen` (CAMS `alder_pollen`, `birch_pollen`, `olive_pollen`, `grass_pollen`, `mugwort_pollen`, `ragweed_pollen`).

## [v1.4.0] - 2026-11-09

### Full End-to-End Recursive Alignment Engine & Per-Model Bias-Corrected Telemetry
- **Per-Model Current Telemetry in `ModelWeightDetail` (`app/models/weather.py` & `app/services/weather_provider.py`)**:
  - Added `currentTempCelsius`, `currentPrecipProb`, and `currentWindKmh` (diurnal-bias-corrected per model) to `ModelWeightDetail` so web and mobile clients render model convergence directly from ZüsAPI's RLS ledger.
- **Full Hourly & Daily RLS Alignment (`app/services/weather_provider.py`)**:
  - Added vector `sin`/`cos` RLS-weighted wind direction averaging, thermodynamically reconciled relative humidity, and RLS-weighted CAPE across hourly forecast horizons.
  - Reconciled dry hourly and daily WMO sky codes (`0` Clear, `2` Partly Cloudy, `3` Overcast) with RLS-blended cloud cover percentages (`0%–35%` Clear, `36%–84%` Partly Cloudy, `85%+` Overcast).
- **Eliminated Raw `current_raw` Leaks & Anchored `hourly[0]` ("NOW")**:
  - Removed raw `current_raw["weather_code"]`, `current_raw["apparent_temperature"]`, and `current_raw["cloud_cover"]` overrides so `current` always reflects IDERW station ground-truth + the RLS-blended `hourly[0]` consensus.
  - Anchored `hourly[0]` ("NOW") 100% to the station-assimilated `current` state and smoothly decayed station temperature deltas (`0.55` at `+1h`, `0.25` at `+2h`) into the forward hourly trajectory.

## [v1.3.1] - 2026-11-09

### iOS WidgetKit Telemetry Sync Relay (`/api/v1/notifications/widget-sync`)
- Added ultra-low-latency in-memory widget telemetry relay endpoints (`POST /api/v1/notifications/widget-sync` & `GET /api/v1/notifications/widget-sync`) to synchronize active location, temperature unit, and multi-model forecast snapshots between the ZüsCast mobile app and the iOS WidgetKit extension.

## [v1.3.0] - 2026-11-08

### Spatial Cell (0.05°) Push Notification Engine & DND Hysteresis Worker
- **Privacy-Preserving Spatial Cell Registration (`POST /api/v1/notifications/register`)**:
  - Quantizes device locations into anonymous `0.05°` (~5 km) spatial grid cells (`SpatialPushWorker`), avoiding exact coordinate storage.
  - Supports client notification preference configurations (precipitation, severe weather, environmental thresholds, quiet hours, and breakthrough preferences) with token in-place updating and automatic cell migration.
- **5-Minute O(1) Push Evaluation Cycle (`push_worker.py`)**:
  - Evaluates active spatial cells in parallel against live MRMS/NEXRAD Doppler radar and NWS warning polygons.
  - Implements a 90-minute per-cell storm hysteresis window (`last_notified`) to completely eliminate repetitive duplicate push alerts during ongoing weather events.
- **Exact-Minute Local Timezone DND & Severe Breakthrough**:
  - Resolves device IANA timezone and evaluates exact-minute quiet hours windows (`start_time` / `end_time`).
  - Supports severe weather breakthrough, allowing critical life-safety warnings (Tornado, Severe Thunderstorm, Flash Flood) to bypass active quiet hours.

## [v1.2.0] - 2026-11-08

### Single-Request Radar Regional Pack & Viewport Pack Endpoints
- Added `GET /api/v1/radar/regional_pack` (`doppler_service.get_regional_pack`) to slice all 14 timeline frames (`-45m` through `+2h`) in a single vectorized NumPy pass into continuous `560x560` WebMercator PNG images (`~800 km x 800 km` span) with LRU caching (`_regional_pack_cache`), eliminating multi-tile checkerboard loading flicker.
- Added `GET /api/v1/radar/viewport_pack` (`doppler_service.get_viewport_tile_pack`) for batched 3x3 tile neighborhood preloading.
- Added semantic `frame_id` resolution (`past_45m`, `past_30m`, `past_15m`, `latest`, `nowcast_p15m`..`nowcast_p120m`) in `get_tile_png`.

## [v1.1.0] - 2026-11-08
- RK2 curved point nowcasting, wet-bulb/sleet/Kuchera SLR physics, and Dark Sky v1 parity.

## [v1.0.0] - 2026-11-07
- Initialized standalone `ZusAPI` repository with semantic versioning.
