# Changelog

All notable changes to this service will be documented in this file.

## [v1.9.1] - 2026-11-22

### Daily Briefing & Local Temperature Consistency
- **Consistent Neighborhood High & Low Temperatures**:
  - Stabilized hourly temperature calibration across adjacent neighborhood cells and prevented repeated intra-hour morning updates from skewing afternoon high temperatures.
  - Preserved verified local morning lows and daytime highs across the full local calendar day, including late-evening hours after UTC midnight.
- **Scheduled Briefing Accuracy & Subscription Hygiene**:
  - Updated scheduled Morning Briefings to query subscriber coordinates directly and reconcile same-day highs and lows with live conditions.
  - Gated scheduled briefing evaluation and delivery ledgers on active push tokens so unprovisioned devices never record phantom dispatches.

## [v1.9.0] - 2026-11-21

### Detailed Weather Statements, Official Links & Advisory Notifications
- **Expanded Weather Alert Metadata**:
  - Added `instruction`, `senderName`, `areaDesc`, `effective`, and official `url` fields to weather alert responses so clients can display complete National Weather Service bulletins, preparedness actions, and direct links to official alert pages.
- **Informative Advisory Push Notifications**:
  - Updated severe weather and advisory push notification bodies to summarize the actual hazard description and instructions instead of repeating generic alert headlines.

## [v1.8.3] - 2026-11-19

### Widget Freshness, Forecast Continuity & Upstream Efficiency
- **Widget Snapshot Freshness Enforcement**:
  - Enforced a strict 30-minute maximum stale window on cached widget sync responses (`503 Service Unavailable` when expired and upstream refresh fails) so home and lock screen widgets never display multi-hour-old weather as fresh.
  - Filtered refreshed widget hourly strips to the current hour onward so upcoming hour columns stay synchronized with wall-clock time.
- **Regional Forecast Continuity & Station Assimilation**:
  - Added automatic fallback to cached short-term forecasts (`<= 3h`) and official regional public forecast guidance paired with live local weather station and radar telemetry whenever primary global guidance feeds experience rate limits or outages.
- **Upstream Request & Verification Optimization**:
  - Separated single-stream fields from multi-model queries, omitted redundant 24-hour lookbacks when unneeded, increased shared response cache TTL (`10 min`), and streamlined background verification (`2h` interval across the top `6` recently active cells with rate-limit short-circuiting).

## [v1.8.2] - 2026-11-17

### 10-Day Temperature Calibration, Snow Physics & Daily Icon Accuracy
- **Extended-Range Temperature & Precipitation Calibration**:
  - Added horizon-aware temperature bias tapering beyond 36 hours so short-range morning microclimate precision stays intact while late-week pattern shifts track warmer or cooler airmass changes accurately.
  - Expanded contiguous US multi-guidance blending and balanced extended-range weighting alongside whole-day frontal rain agreement calibration.
- **Per-Stream Winter Precipitation Physics & Daily Icon Consistency**:
  - Computed winter precipitation phase and snow accumulation per individual guidance stream prior to blending, eliminating artificial cross-stream snow totals on split rain/cold patterns.
  - Gated low-probability extended-range snow amounts, prioritized daylight cloud cover for dry daily sky icons, and aligned mixed rain/snow icons to avoid conflicting precipitation badges.

## [v1.8.1] - 2026-11-15

### Non-Mainland Precipitation Consistency & Light-Rain Threshold Alignment
- **Global & Regional Next-Hour Precipitation Alignment**:
  - Improved minute-by-minute precipitation nowcasting for non-mainland regions and island territories, ensuring high-resolution next-hour curves are consistently populated from regional observation networks.
- **Light Rain & Drizzle Threshold Calibration**:
  - Calibrated current condition thresholds for light drizzle and light rain to ensure condition indicators match ground-level precipitation rates and eliminate conflicting dry and wet reports.

## [v1.8.0] - 2026-11-14

### Privacy-First Accounts, Direct Store Entitlements & Encrypted Nightly Backups
- **Privacy-First Account Synchronization & Entitlement Lifecycle**:
  - Added optional, anonymous cloud account synchronization and subscription tier management (`endpoints/accounts.py`, `services/account_service.py`) collecting zero personal identifying information (no email addresses, names, phone numbers, IP addresses, or GPS coordinates are stored).
  - Integrated direct App Store and Google Play purchase receipt and purchase-token cryptographic verification with automatic subscription renewal, expiration, and refund webhook synchronization without third-party billing intermediaries.
  - Implemented single-use and multi-use Founder invitation code generation with rate-limited atomic redemption, automatic early-adopter grandfathering, and instant account deletion support.
- **Encrypted Nightly Database & State Backups**:
  - Automated AES-256-CBC encrypted nightly database and notification-state backup pipeline (`scripts/backup_databases.sh`) with SHA-256 checksum verification and 30-day rolling retention.

## [v1.7.0] - 2026-11-14

### Current-Location Widget Isolation & Travel-Aware Push Cell Synchronization
- **Travel-Aware Push Subscription Cell Updates**:
  - Added in-place spatial-cell coordinate and city updates when live background widget location refreshes detect subscriber movement into a new `0.05°` (~5 km) cell, keeping precipitation alerts, severe weather warnings, and scheduled daily briefings aligned with the user's physical location while traveling without requiring an app launch.
  - Preserved strict opt-in semantics so background widget refreshes never recreate subscriptions for paused or unsubscribed devices.
- **Strict Device-Scoped Widget Snapshot Isolation & Location-Required State**:
  - Removed cross-device fallback snapshot lookups so background widget refreshes remain strictly isolated per device installation.
  - Added explicit `locationRequired` state handling so devices with Location Services turned off preserve their prompt state cleanly without falling back to stale or default coordinates.

## [v1.6.0] - 2026-11-14

### Live Scheduled Daily Briefings, Unit-Aware Push Formatting & Legacy Payload Normalization
- **Timezone-Aware Spatial-Cell Morning Briefings & Evening Outlooks**:
  - Added scheduled Morning Briefing and Evening Outlook push evaluation grouped by `0.05°` (~5 km) spatial cells so forecast lookups run at most once per active cell when subscribers enter their local briefing window.
  - Formatted briefing highs and lows in each subscriber's preferred temperature unit (`°F` or `°C`) with delivery-accurate labels (`Today:` for morning briefings and `Tomorrow's Outlook:` for evening outlooks), quiet-hours suppression, and per-day SQLite deduplication (`dispatched_briefings`) with 10-day retention pruning.
- **Extended Subscription Schema & Backward-Compatible Registration**:
  - Added subscriber `city`, `temp_unit`, `wind_unit`, daily briefing schedules, and ambient alert preferences with automatic SQLite `ALTER TABLE` column migration and `server_briefings_supported` capability signaling.
  - Added request normalization for legacy nested registration payloads and push-token validation while keeping legacy clients opted out of server-dispatched briefings to prevent duplicate notifications.

## [v1.5.2] - 2026-11-11

### Widget Background Refresh Parity & Overcast Clutter Suppression
- **Overcast Ground-Truth Alignment for Background Widget Refreshes**:
  - Applied surface ground-truth precipitation filtering during background widget snapshot refreshes so weak overcast cloud echoes and uncorroborated drizzle signals never misclassify dry overcast conditions as rain.
  - Replaced legacy wet fallback strings on dry curves with explicit dry-hour labels and expanded refreshed widget payloads to include the full 7-day forecast and hourly condition labels.
- **Device & Location-Scoped Widget Snapshot Caching**:
  - Scoped synced widget snapshots by device identifier and coordinate cell so Wi-Fi-to-cellular network handoffs always preserve each user's active location and unit preferences.

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
  - Fixed UTC dislocation in daily `sunrise`/`sunset` strings by shifting upstream model UTC timestamps by `utc_offset_seconds` (`dt_utc + timedelta(seconds=utc_offset_sec)`), indexing by local calendar date, and validating/backfilling with `calc_local_sunrise_sunset_iso` (NOAA solar ephemeris equations).
- **Extended Current & Hourly Telemetry (`app/models/weather.py` & `app/services/weather_provider.py`)**:
  - Added `isDay`, `shortwaveRadiation`, `vpdKpa`, `cloudCoverLow`, `cloudCoverMid`, `cloudCoverHigh`, `freezingLevelMeters`, `treePollen`, `grassPollen`, and `weedPollen` (CAMS `alder_pollen`, `birch_pollen`, `olive_pollen`, `grass_pollen`, `mugwort_pollen`, `ragweed_pollen`).

## [v1.4.0] - 2026-11-09

### Full End-to-End Recursive Alignment Engine & Per-Model Bias-Corrected Telemetry
- **Per-Model Current Telemetry in `ModelWeightDetail` (`app/models/weather.py` & `app/services/weather_provider.py`)**:
  - Added `currentTempCelsius`, `currentPrecipProb`, and `currentWindKmh` (diurnal-bias-corrected per model) to `ModelWeightDetail` so web and mobile clients render model convergence directly from the ensemble RLS ledger.
- **Full Hourly & Daily RLS Alignment (`app/services/weather_provider.py`)**:
  - Added vector `sin`/`cos` RLS-weighted wind direction averaging, thermodynamically reconciled relative humidity, and RLS-weighted CAPE across hourly forecast horizons.
  - Reconciled dry hourly and daily WMO sky codes (`0` Clear, `2` Partly Cloudy, `3` Overcast) with RLS-blended cloud cover percentages (`0%–35%` Clear, `36%–84%` Partly Cloudy, `85%+` Overcast).
- **Eliminated Raw `current_raw` Leaks & Anchored `hourly[0]` ("NOW")**:
  - Removed raw `current_raw["weather_code"]`, `current_raw["apparent_temperature"]`, and `current_raw["cloud_cover"]` overrides so `current` always reflects calibrated surface station ground-truth + the RLS-blended `hourly[0]` consensus.
  - Anchored `hourly[0]` ("NOW") 100% to the station-assimilated `current` state and smoothly decayed station temperature deltas (`0.55` at `+1h`, `0.25` at `+2h`) into the forward hourly trajectory.

## [v1.3.1] - 2026-11-09

### iOS WidgetKit Telemetry Sync Relay (`/notifications/widget-sync`)
- Added ultra-low-latency in-memory widget telemetry relay endpoints (`POST /notifications/widget-sync` & `GET /notifications/widget-sync`) to synchronize active location, temperature unit, and multi-model forecast snapshots between the mobile app and the iOS WidgetKit extension.

## [v1.3.0] - 2026-11-08

### Spatial Cell (0.05°) Push Notification Engine & DND Hysteresis Worker
- **Privacy-Preserving Spatial Cell Registration (`POST /notifications/register`)**:
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
- Added `GET /radar/regional_pack` (`doppler_service.get_regional_pack`) to slice all 14 timeline frames (`-45m` through `+2h`) in a single vectorized NumPy pass into continuous `560x560` WebMercator PNG images (`~800 km x 800 km` span) with LRU caching (`_regional_pack_cache`), eliminating multi-tile checkerboard loading flicker.
- Added `GET /radar/viewport_pack` (`doppler_service.get_viewport_tile_pack`) for batched 3x3 tile neighborhood preloading.
- Added semantic `frame_id` resolution (`past_45m`, `past_30m`, `past_15m`, `latest`, `nowcast_p15m`..`nowcast_p120m`) in `get_tile_png`.

## [v1.1.0] - 2026-11-08
- High-precision curved point nowcasting, wet-bulb/sleet precipitation-type physics, and snow-liquid-ratio modeling.

## [v1.0.0] - 2026-11-07
- Initialized standalone repository with semantic versioning.
