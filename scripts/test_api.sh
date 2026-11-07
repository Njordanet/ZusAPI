#!/usr/bin/env bash
set -e

HOST="${1:-http://localhost:8000}"
echo "⚡️ Testing ZüsAPI v2.1 at ${HOST}..."

echo "--- 1. Healthcheck ---"
curl -s "${HOST}/api/v1/health" | python3 -m json.tool

echo -e "\n--- 2. Root Index (Universal Endpoints) ---"
curl -s "${HOST}/" | python3 -m json.tool

echo -e "\n--- 3. Native Züs Weather (New York, Metric) ---"
curl -s "${HOST}/api/v1/weather?lat=40.7128&lon=-74.0060&city=New+York" | python3 -m json.tool | head -n 35

echo -e "\n--- 4. Native Züs Weather (Imperial + Slim Widget Exclude + 6h Limit) ---"
curl -s "${HOST}/api/v1/weather?lat=40.7128&lon=-74.0060&city=New+York&units=imperial&exclude=nearbyStations,modelWeights,daily&hours=6" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(json.dumps({
    'cached': d['cached'],
    'timezone': d['timezone'],
    'utcOffsetSeconds': d['utcOffsetSeconds'],
    'temp_F': d['current']['temperature'],
    'wind_mph': d['current']['windSpeed'],
    'hourly_count': len(d['hourly']),
    'stations_excluded': len(d['nearbyStations']) == 0,
    'weights_excluded': len(d['modelWeights']) == 0,
}, indent=2))
"

echo -e "\n--- 5. Dark Sky / Pirate Weather Standard Drop-In (/forecast/key/lat,lon) ---"
curl -s "${HOST}/forecast/zus_local_key/40.7128,-74.0060?units=us" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(json.dumps({
    'timezone': d['timezone'],
    'offset': d['offset'],
    'currently_icon': d['currently']['icon'],
    'currently_temp_F': d['currently']['temperature'],
    'minutely_points': len(d['minutely']['data']),
    'hourly_points': len(d['hourly']['data']),
    'daily_0_sunrise': d['daily']['data'][0]['sunriseTime'],
    'daily_0_sunset': d['daily']['data'][0]['sunsetTime'],
    'daily_0_moonPhase': d['daily']['data'][0]['moonPhase'],
}, indent=2))
"

echo -e "\n--- 6. Open-Meteo v1 Standard Drop-In (/v1/forecast) ---"
curl -s "${HOST}/v1/forecast?latitude=40.7128&longitude=-74.0060&timezone=auto&temperature_unit=fahrenheit&wind_speed_unit=mph" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(json.dumps({
    'timezone': d['timezone'],
    'utc_offset_seconds': d['utc_offset_seconds'],
    'current_temp_F': d['current']['temperature_2m'],
    'current_wind_mph': d['current']['wind_speed_10m'],
    'hourly_len': len(d['hourly']['time']),
    'daily_len': len(d['daily']['time']),
}, indent=2))
"

echo -e "\n--- 7. OpenWeatherMap (OWM) OneCall 3.0 & v2.5 Drop-In ---"
curl -s "${HOST}/data/3.0/onecall?lat=40.7128&lon=-74.0060&units=imperial" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(json.dumps({
    'onecall_timezone': d['timezone'],
    'onecall_temp_F': d['current']['temp'],
    'onecall_weather': d['current']['weather'][0],
    'minutely_len': len(d['minutely']),
}, indent=2))
"
curl -s "${HOST}/data/2.5/weather?lat=40.7128&lon=-74.0060&units=metric" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(json.dumps({
    'owm25_name': d['name'],
    'owm25_temp_C': d['main']['temp'],
    'owm25_weather': d['weather'][0],
    'cod': d['cod'],
}, indent=2))
"

echo -e "\n--- 8. Dataset Inventory ---"
curl -s "${HOST}/api/v1/dataset/inventory" | python3 -m json.tool | head -n 30
