import httpx
from typing import Optional


class SharedHttpClientManager:
    """
    Manages a persistent, connection-pooled httpx.AsyncClient across all ZüsAPI services
    to eliminate repeated TCP + TLS 1.3 handshake overhead on upstream meteorological queries.
    """

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None

    def get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(10.0, connect=4.0),
                limits=httpx.Limits(
                    max_keepalive_connections=30,
                    max_connections=80,
                    keepalive_expiry=60.0,
                ),
                headers={"User-Agent": "ZusWeatherEngine/2.1 (panda-telemetry)"},
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


http_pool = SharedHttpClientManager()
