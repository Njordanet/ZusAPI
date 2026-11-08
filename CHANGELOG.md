# Changelog

All notable changes to `ZusAPI` will be documented in this file.

## [v1.2.0] - 2026-11-08

### Single-Request Radar Regional Pack & Viewport Pack Endpoints
- Added `GET /api/v1/radar/regional_pack` (`doppler_service.get_regional_pack`) to slice all 14 timeline frames (`-45m` through `+2h`) in a single vectorized NumPy pass into continuous `560x560` WebMercator PNG images (`~800 km x 800 km` span) with LRU caching (`_regional_pack_cache`), eliminating multi-tile checkerboard loading flicker.
- Added `GET /api/v1/radar/viewport_pack` (`doppler_service.get_viewport_tile_pack`) for batched 3x3 tile neighborhood preloading.
- Added semantic `frame_id` resolution (`past_45m`, `past_30m`, `past_15m`, `latest`, `nowcast_p15m`..`nowcast_p120m`) in `get_tile_png`.

## [v1.1.0] - 2026-11-08
- RK2 curved point nowcasting, wet-bulb/sleet/Kuchera SLR physics, and Dark Sky v1 parity.

## [v1.0.0] - 2026-11-07
- Initialized standalone `ZusAPI` repository with semantic versioning.
