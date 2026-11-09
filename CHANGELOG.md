# Changelog

All notable changes to `ZusAPI` will be documented in this file.

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
