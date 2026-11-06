# ⚡️ ZüsAPI — Sky Telemetry & Weather Backend

**ZüsAPI** is the foundational backend service for the **Züs** weather ecosystem, hosted on Ubuntu server **`panda`** via Docker Compose.

---

## 🏛 Architecture

- **Framework**: FastAPI (Python 3.12) + Uvicorn
- **Validation & Serialization**: Pydantic v2
- **Network**: Async HTTP via HTTPX with connection pooling and caching
- **Deployment**: Docker Compose on PC `panda` (Ubuntu 24.04 LTS, port `8000`)
- **Integration**: Direct schema alignment with the SwiftUI `Zus` client (`WeatherData`, `HourlyForecast`, `DailyForecast`, `WeatherCondition`)

---

## 🚀 API Endpoints

- `GET /api/v1/health` — Service healthcheck, system telemetry, and uptime.
- `GET /api/v1/weather?lat={lat}&lon={lon}&city={city}` — Weather observations and forecasts.
- `GET /api/v1/lightning?lat={lat}&lon={lon}&radius_km={radius}` — Lightning strike proximity and threat level.
- `GET /api/v1/dataset/inventory` — Dataset catalog and historical storage status.
- `GET /docs` — Interactive OpenAPI / Swagger UI.

---

## 🛠 Local Development & Deployment

### Run Locally (Docker)
```bash
docker compose up --build
```

### Deploy to `panda`
```bash
./scripts/deploy_panda.sh
```

Or manually:
```bash
rsync -avz --exclude '__pycache__' --exclude '.git' ./ panda:~/zus-api/
ssh panda "cd ~/zus-api && docker compose up -d --build"
```
