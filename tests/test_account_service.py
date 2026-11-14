import pytest
from unittest.mock import patch
from fastapi import HTTPException

from app.services.account_service import _get_supabase_headers


def test_get_supabase_headers_legacy_jwt():
    with patch("app.services.account_service.settings") as mock_settings:
        mock_settings.supabase_url = "https://example.supabase.co"
        mock_settings.supabase_service_role_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.dummy.sig"
        headers = _get_supabase_headers()
        assert headers["apikey"] == "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.dummy.sig"
        assert headers["Authorization"] == "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.dummy.sig"
        assert headers["Content-Type"] == "application/json"
        assert headers["Prefer"] == "return=representation"


def test_get_supabase_headers_sb_secret():
    with patch("app.services.account_service.settings") as mock_settings:
        mock_settings.supabase_url = "https://example.supabase.co"
        mock_settings.supabase_service_role_key = "sb_secret_abc123xyz"
        headers = _get_supabase_headers()
        assert headers["apikey"] == "sb_secret_abc123xyz"
        assert "Authorization" not in headers
        assert headers["Content-Type"] == "application/json"
        assert headers["Prefer"] == "return=representation"


def test_get_supabase_headers_missing_config():
    with patch("app.services.account_service.settings") as mock_settings:
        mock_settings.supabase_url = None
        mock_settings.supabase_service_role_key = None
        with pytest.raises(HTTPException) as exc_info:
            _get_supabase_headers()
        assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_redeem_founder_code_rpc_invalid_code():
    from unittest.mock import AsyncMock, MagicMock
    from app.services.account_service import account_service, _REDEMPTION_FAILED_ATTEMPTS

    _REDEMPTION_FAILED_ATTEMPTS.clear()
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = [{"ok": False, "error_code": "invalid_code"}]

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_res

    with patch("app.services.account_service.http_pool.get_client", return_value=mock_client), \
         patch("app.services.account_service.settings") as mock_settings:
        mock_settings.supabase_url = "https://example.supabase.co"
        mock_settings.supabase_service_role_key = "sb_secret_test"
        with pytest.raises(HTTPException) as exc_info:
            await account_service.redeem_founder_code("user-invalid", "ZUS-FOUNDER-AAAA-BBBB", "ip-invalid")
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == "Invalid founder code."


@pytest.mark.asyncio
async def test_redeem_founder_code_rpc_already_redeemed():
    from unittest.mock import AsyncMock, MagicMock
    from app.services.account_service import account_service, _REDEMPTION_FAILED_ATTEMPTS

    _REDEMPTION_FAILED_ATTEMPTS.clear()
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = [{"ok": False, "error_code": "already_redeemed"}]

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_res

    with patch("app.services.account_service.http_pool.get_client", return_value=mock_client), \
         patch("app.services.account_service.settings") as mock_settings:
        mock_settings.supabase_url = "https://example.supabase.co"
        mock_settings.supabase_service_role_key = "sb_secret_test"
        with pytest.raises(HTTPException) as exc_info:
            await account_service.redeem_founder_code("user-already", "ZUS-FOUNDER-AAAA-BBBB", "ip-already")
        assert exc_info.value.status_code == 409
        assert "already redeemed" in exc_info.value.detail


@pytest.mark.asyncio
async def test_redeem_founder_code_rpc_success():
    from unittest.mock import AsyncMock, MagicMock
    from app.services.account_service import account_service, _REDEMPTION_FAILED_ATTEMPTS
    from app.models.accounts import EntitlementResponse

    _REDEMPTION_FAILED_ATTEMPTS.clear()
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = [
        {"ok": True, "error_code": None, "tier": "founder", "status": "active", "source": "founder_code"}
    ]

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_res

    expected_entitlement = EntitlementResponse(
        user_id="user-success",
        tier="founder",
        status="active",
        source="founder_code",
        is_founder=True,
    )

    with patch("app.services.account_service.http_pool.get_client", return_value=mock_client), \
         patch("app.services.account_service.settings") as mock_settings, \
         patch.object(account_service, "get_entitlement", new_callable=AsyncMock) as mock_get_ent:
        mock_settings.supabase_url = "https://example.supabase.co"
        mock_settings.supabase_service_role_key = "sb_secret_test"
        mock_get_ent.return_value = expected_entitlement

        result = await account_service.redeem_founder_code("user-success", "ZUS-FOUNDER-AAAA-BBBB", "ip-success")
        assert isinstance(result, EntitlementResponse)
        assert result.tier == "founder"
        assert result.status == "active"
        mock_get_ent.assert_awaited_once_with("user-success")


def test_account_service_source_contract_assertions():
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "app" / "services" / "account_service.py"
    content = path.read_text(encoding="utf-8")
    for pattern in ["user_profiles?id=", "times_redeemed", "\"code_redemption\"", "\"google_play\""]:
        assert content.count(pattern) == 0, f"Found forbidden pattern '{pattern}' in account_service.py"

