import hmac
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Header, Query, Request, Body, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config import settings
from app.models.accounts import (
    EntitlementResponse,
    BootstrapAccountRequest,
    RedeemCodeRequest,
    AppleVerifyRequest,
    GoogleVerifyRequest,
    CreateFounderCodesRequest,
    CreateFounderCodesResponse,
    AdminGrantFounderRequest,
    AdminGrantFounderResponse,
)
from app.services.account_service import account_service

router = APIRouter(prefix="/accounts", tags=["Accounts & Billing"])

_bearer_security = HTTPBearer(auto_error=False)


async def get_current_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_security),
) -> str:
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Bearer authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await account_service.verify_supabase_jwt(credentials.credentials)


async def verify_admin_key(
    x_zus_admin_key: Optional[str] = Header(None, alias="X-Zus-Admin-Key"),
) -> None:
    if not settings.zus_admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin API key is not configured on server",
        )
    if not x_zus_admin_key or not hmac.compare_digest(x_zus_admin_key, settings.zus_admin_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin credentials",
        )


@router.post("/bootstrap", response_model=EntitlementResponse)
async def bootstrap(
    body: BootstrapAccountRequest = Body(default_factory=BootstrapAccountRequest),
    user_id: str = Depends(get_current_user_id),
):
    """
    Ensures user profile and entitlement records exist upon app launch.
    Automatically applies early adopter founder status if registered before cutoff.
    """
    return await account_service.bootstrap_account(user_id, body.first_seen_version)


@router.get("/entitlements", response_model=EntitlementResponse)
async def get_entitlements(user_id: str = Depends(get_current_user_id)):
    """
    Returns current active entitlement status for the authenticated user.
    """
    return await account_service.get_entitlement(user_id)


@router.post("/redeem", response_model=EntitlementResponse, include_in_schema=False)
@router.post("/redeem-code", response_model=EntitlementResponse)
async def redeem_code(
    body: RedeemCodeRequest,
    request: Request,
    user_id: str = Depends(get_current_user_id),
):
    """
    Redeems a founder or lifetime invitation code.
    Protected by rate-limiting (max 6 failed attempts per 15 minutes per user/IP).
    """
    client_ip = (
        request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        or (request.client.host if request.client else "unknown")
    )
    return await account_service.redeem_founder_code(user_id, body.code, client_ip)


@router.post("/verify/apple", response_model=EntitlementResponse, include_in_schema=False)
@router.post("/billing/verify-apple", response_model=EntitlementResponse)
async def verify_apple(
    body: AppleVerifyRequest,
    user_id: str = Depends(get_current_user_id),
):
    """
    Verifies App Store StoreKit 2 transaction JWS or App Store Server API transaction ID.
    """
    return await account_service.verify_apple_purchase(user_id, body)


@router.post("/verify/google", response_model=EntitlementResponse, include_in_schema=False)
@router.post("/billing/verify-google", response_model=EntitlementResponse)
async def verify_google(
    body: GoogleVerifyRequest,
    user_id: str = Depends(get_current_user_id),
):
    """
    Verifies Google Play purchase token via Google Android Publisher API v3.
    """
    return await account_service.verify_google_purchase(user_id, body)


@router.post("/webhooks/apple")
async def apple_webhook(request: Request):
    """
    Public webhook receiver for App Store Server Notifications V2.
    Cryptographically authenticated via Apple Root CA certificate chain inside JWS.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    signed_payload = body.get("signedPayload") if isinstance(body, dict) else None
    if not signed_payload:
        raw_text = (await request.body()).decode("utf-8").strip()
        if raw_text.startswith("ey"):
            signed_payload = raw_text

    if not signed_payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing signedPayload in StoreKit webhook notification",
        )

    return await account_service.handle_apple_webhook(signed_payload)


@router.post("/webhooks/google")
async def google_webhook(
    request: Request,
    body: Dict[str, Any] = Body(...),
    token: Optional[str] = Query(None),
    x_goog_channel_token: Optional[str] = Header(None, alias="X-Goog-Channel-Token"),
):
    """
    Google Cloud Pub/Sub Real-Time Developer Notifications (RTDN) webhook receiver.
    Guarded by verification token validation when configured.
    """
    if settings.google_rtdn_webhook_token:
        provided = token or x_goog_channel_token or ""
        if not hmac.compare_digest(provided, settings.google_rtdn_webhook_token):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: RTDN webhook verification token mismatch",
            )

    return await account_service.handle_google_rtdn(body)


@router.delete("/me")
async def delete_me(user_id: str = Depends(get_current_user_id)):
    """
    Permanently deletes user account, profile, and entitlement records.
    Store compliance & GDPR/CCPA erasure.
    """
    return await account_service.delete_account(user_id)


@router.post("/admin/founder-codes", response_model=CreateFounderCodesResponse)
async def create_founder_codes(
    body: CreateFounderCodesRequest,
    _: None = Depends(verify_admin_key),
):
    """
    Admin: Generates batch of single-use or multi-use founder codes.
    Requires X-Zus-Admin-Key authentication.
    """
    return await account_service.create_founder_codes(body)


@router.post("/admin/grant-founder", response_model=AdminGrantFounderResponse)
async def grant_founder(
    body: AdminGrantFounderRequest,
    _: None = Depends(verify_admin_key),
):
    """
    Admin: Grants founder tier to a user or grandfathered cohort.
    Requires X-Zus-Admin-Key authentication.
    """
    return await account_service.admin_grant_founder(body)
