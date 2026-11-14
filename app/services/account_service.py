import base64
import hashlib
import hmac
import json
import logging
import re
import secrets
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Tuple

import jwt
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from fastapi import HTTPException, status

from app.config import settings
from app.models.accounts import (
    EntitlementResponse,
    AppleVerifyRequest,
    GoogleVerifyRequest,
    CreateFounderCodesRequest,
    CreateFounderCodesResponse,
    AdminGrantFounderRequest,
    AdminGrantFounderResponse,
)
from app.services.http_client import http_pool

logger = logging.getLogger("zus_api.accounts")

# Apple Root CA SHA-256 Fingerprints
APPLE_ROOT_CA_G3_FP = "63343abfb89a6a03ebb57e9b3f5fa7be7c4f5c756f3017b3a8c488c3653e9179"
APPLE_INC_ROOT_FP = "b0b1730ecbc7ff4505142c49f1295e6eda6bcaed7e2c68c5be91b5a11001f024"

# Code generation charset (unambiguous characters)
FOUNDER_CODE_CHARSET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

# In-memory caches & rate limiters
_JWKS_CACHE: Dict[str, Any] = {}
_JWKS_CACHED_AT: float = 0.0
_JWKS_TTL_SECONDS = 900.0  # 15 minutes

_GOOGLE_TOKEN_CACHE: Dict[str, Any] = {}

# Sliding-window rate limiting for code redemption: max 6 failed attempts per 900 seconds per (user_id, client_key)
_REDEMPTION_FAILED_ATTEMPTS: Dict[Tuple[str, str], List[float]] = {}
_RATE_LIMIT_WINDOW = 900.0
_RATE_LIMIT_MAX_FAILED = 6


def _check_redemption_rate_limit(user_id: str, client_key: str) -> None:
    now = time.time()
    key = (user_id, client_key)
    attempts = [t for t in _REDEMPTION_FAILED_ATTEMPTS.get(key, []) if now - t < _RATE_LIMIT_WINDOW]
    _REDEMPTION_FAILED_ATTEMPTS[key] = attempts
    if len(attempts) >= _RATE_LIMIT_MAX_FAILED:
        retry_after = int(_RATE_LIMIT_WINDOW - (now - attempts[0])) if attempts else 900
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many failed redemption attempts. Please wait {max(1, retry_after)} seconds before trying again.",
            headers={"Retry-After": str(max(1, retry_after))},
        )


def _record_failed_attempt(user_id: str, client_key: str) -> None:
    now = time.time()
    key = (user_id, client_key)
    attempts = [t for t in _REDEMPTION_FAILED_ATTEMPTS.get(key, []) if now - t < _RATE_LIMIT_WINDOW]
    attempts.append(now)
    _REDEMPTION_FAILED_ATTEMPTS[key] = attempts


def _parse_semver(v: Optional[str]) -> Optional[Tuple[int, ...]]:
    if not v:
        return None
    cleaned = v.strip().lstrip("v")
    match = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?", cleaned)
    if not match:
        return None
    parts = [int(p) if p is not None else 0 for p in match.groups()]
    return tuple(parts)


def _get_supabase_headers() -> Dict[str, str]:
    if not settings.supabase_url or not settings.supabase_service_role_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Account service not yet configured",
        )
    key = settings.supabase_service_role_key
    headers = {
        "apikey": key,
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    if not key.startswith("sb_secret_"):
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _get_supabase_rest_url(endpoint: str) -> str:
    base = settings.supabase_url.rstrip("/")
    path = endpoint.lstrip("/")
    return f"{base}/rest/v1/{path}"


def _get_supabase_auth_url(endpoint: str) -> str:
    base = settings.supabase_url.rstrip("/")
    path = endpoint.lstrip("/")
    return f"{base}/auth/v1/{path}"


class AccountService:
    """
    Manages user accounts, cryptographic token validation (Supabase JWT, Apple StoreKit 2 JWS, Google Play),
    server-side entitlements, and founder code redemptions.
    Zero PII is ever collected or persisted (no email, name, IP, or GPS coordinates).
    """

    async def verify_supabase_jwt(self, token: str) -> str:
        """
        Validates incoming Supabase Bearer JWT.
        Supports HS256 (symmetric shared secret) and RS256/ES256 (cached JWKS via Supabase auth).
        Returns validated user_id (sub claim).
        """
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        try:
            unverified_header = jwt.get_unverified_header(token)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Malformed authentication token: {e}",
                headers={"WWW-Authenticate": "Bearer"},
            )

        alg = unverified_header.get("alg", "HS256")

        try:
            if alg == "HS256":
                if not settings.supabase_jwt_secret:
                    # Fallback to anon or service role key if user passed secret as secret
                    if not settings.supabase_url:
                        raise HTTPException(
                            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Supabase authentication secret is not configured",
                        )
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Supabase JWT secret is not configured for HS256 validation",
                    )
                payload = jwt.decode(
                    token,
                    settings.supabase_jwt_secret,
                    algorithms=["HS256"],
                    options={"verify_aud": False},
                )
            else:
                # Asymmetric verification via Supabase JWKS
                if not settings.supabase_url:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Supabase URL is not configured for JWKS verification",
                    )

                jwks = await self._get_supabase_jwks()
                kid = unverified_header.get("kid")
                keys = jwks.get("keys", [])
                matching_key = next((k for k in keys if k.get("kid") == kid), None) if kid else None

                if not matching_key and keys:
                    # If kid is missing or not matched, try first key matching alg
                    matching_key = next((k for k in keys if k.get("alg") == alg or "kty" in k), keys[0])

                if not matching_key:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="No matching public key found in Supabase JWKS",
                        headers={"WWW-Authenticate": "Bearer"},
                    )

                pyjwk = jwt.PyJWK.from_dict(matching_key)
                payload = jwt.decode(
                    token,
                    pyjwk.key,
                    algorithms=[alg, "RS256", "ES256"],
                    options={"verify_aud": False},
                )

            sub = payload.get("sub")
            if not sub or not isinstance(sub, str):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token missing valid subject (user_id)",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return sub

        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.PyJWTError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Token verification failed: {e}",
                headers={"WWW-Authenticate": "Bearer"},
            )

    async def _get_supabase_jwks(self) -> Dict[str, Any]:
        global _JWKS_CACHE, _JWKS_CACHED_AT
        now = time.time()
        if _JWKS_CACHE and (now - _JWKS_CACHED_AT < _JWKS_TTL_SECONDS):
            return _JWKS_CACHE

        jwks_url = f"{settings.supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json"
        client = http_pool.get_client()
        try:
            res = await client.get(jwks_url, timeout=5.0)
            if res.status_code == 200:
                _JWKS_CACHE = res.json()
                _JWKS_CACHED_AT = now
                return _JWKS_CACHE
        except Exception as e:
            logger.warning(f"Failed to fetch Supabase JWKS from {jwks_url}: {e}")
            if _JWKS_CACHE:
                return _JWKS_CACHE

        return {"keys": []}

    # -------------------------------------------------------------------------
    # Account Lifecycle & Entitlements
    # -------------------------------------------------------------------------

    async def bootstrap_account(
        self, user_id: str, first_seen_version: Optional[str] = None
    ) -> EntitlementResponse:
        """
        Ensures user_profiles and user_entitlements rows exist for the user.
        Applies early-adopter founder grandfathering if created prior to cutoff.
        """
        headers = _get_supabase_headers()
        client = http_pool.get_client()

        # 1. Fetch or create user profile
        profile_url = _get_supabase_rest_url(f"user_profiles?user_id=eq.{user_id}&select=*")
        p_res = await client.get(profile_url, headers=headers)
        profile_data: Optional[Dict[str, Any]] = None

        if p_res.status_code == 200 and p_res.json():
            profile_data = p_res.json()[0]
            # Update first_seen_version if previously unset
            if first_seen_version and not profile_data.get("first_seen_version"):
                patch_url = _get_supabase_rest_url(f"user_profiles?user_id=eq.{user_id}")
                await client.patch(
                    patch_url,
                    headers=headers,
                    json={"first_seen_version": first_seen_version},
                )
                profile_data["first_seen_version"] = first_seen_version
        else:
            # Create user_profile
            create_payload = {
                "user_id": user_id,
                "first_seen_version": first_seen_version,
                "is_early_adopter": False,
            }
            c_res = await client.post(
                _get_supabase_rest_url("user_profiles"),
                headers=headers,
                json=create_payload,
            )
            if c_res.status_code in (200, 201) and c_res.json():
                profile_data = c_res.json()[0]
            else:
                profile_data = create_payload

        # 2. Fetch or create user entitlement
        ent_url = _get_supabase_rest_url(f"user_entitlements?user_id=eq.{user_id}&select=*")
        e_res = await client.get(ent_url, headers=headers)
        ent_data: Optional[Dict[str, Any]] = None

        if e_res.status_code == 200 and e_res.json():
            ent_data = e_res.json()[0]
        else:
            default_ent = {
                "user_id": user_id,
                "tier": "free",
                "status": "active",
                "source": "none",
                "auto_renew": False,
            }
            c_res = await client.post(
                _get_supabase_rest_url("user_entitlements"),
                headers=headers,
                json=default_ent,
            )
            if c_res.status_code in (200, 201) and c_res.json():
                ent_data = c_res.json()[0]
            else:
                ent_data = default_ent

        # 3. Automatic founder upgrade if cutoff configured and created before cutoff
        if settings.founder_auto_cutoff_iso and profile_data:
            try:
                cutoff_dt = datetime.fromisoformat(
                    settings.founder_auto_cutoff_iso.replace("Z", "+00:00")
                )
                account_created_str = profile_data.get("account_created_at")
                if account_created_str:
                    created_dt = datetime.fromisoformat(
                        account_created_str.replace("Z", "+00:00")
                    )
                    if created_dt <= cutoff_dt and ent_data.get("tier") != "founder":
                        logger.info(
                            f"Auto-upgrading early adopter {user_id} to Founder (created {account_created_str} <= {settings.founder_auto_cutoff_iso})"
                        )
                        patch_ent_url = _get_supabase_rest_url(
                            f"user_entitlements?user_id=eq.{user_id}"
                        )
                        await client.patch(
                            patch_ent_url,
                            headers=headers,
                            json={
                                "tier": "founder",
                                "source": "early_adopter_grant",
                                "status": "active",
                                "expires_at": None,
                                "auto_renew": False,
                            },
                        )
                        patch_prof_url = _get_supabase_rest_url(
                            f"user_profiles?user_id=eq.{user_id}"
                        )
                        await client.patch(
                            patch_prof_url,
                            headers=headers,
                            json={"is_early_adopter": True},
                        )
            except Exception as e:
                logger.warning(f"Error evaluating founder auto-cutoff: {e}")

        return await self.get_entitlement(user_id)

    async def get_entitlement(self, user_id: str) -> EntitlementResponse:
        """
        Reads user_entitlements and user_profiles from Supabase.
        Calculates active flags, honoring a 1-hour grace window on pro expirations.
        Founder and lifetime grants are permanent.
        """
        headers = _get_supabase_headers()
        client = http_pool.get_client()

        ent_url = _get_supabase_rest_url(f"user_entitlements?user_id=eq.{user_id}&select=*")
        prof_url = _get_supabase_rest_url(f"user_profiles?user_id=eq.{user_id}&select=*")

        e_res = await client.get(ent_url, headers=headers)
        p_res = await client.get(prof_url, headers=headers)

        if e_res.status_code != 200 or not e_res.json() or p_res.status_code != 200 or not p_res.json():
            # If not initialized, bootstrap first
            return await self.bootstrap_account(user_id)

        ent = e_res.json()[0]
        prof = p_res.json()[0]

        tier = ent.get("tier", "free")
        status_val = ent.get("status", "none")
        source = ent.get("source", "none")
        product_id = ent.get("product_id")
        expires_at = ent.get("expires_at")
        auto_renew = bool(ent.get("auto_renew", False))
        is_early_adopter = bool(prof.get("is_early_adopter", False))
        account_created_at = prof.get("account_created_at")
        first_seen_version = prof.get("first_seen_version")
        updated_at = ent.get("updated_at")

        is_pro = False
        is_lifetime = False
        is_founder = False

        now = datetime.now(timezone.utc)

        # Pro subscription check with 1-hour grace period past expires_at
        if tier == "pro":
            if expires_at:
                try:
                    exp_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                    if now > (exp_dt + timedelta(hours=1)):
                        status_val = "expired"
                        is_pro = False
                    else:
                        is_pro = status_val in ("active", "trialing", "grace_period")
                except Exception:
                    is_pro = status_val in ("active", "trialing", "grace_period")
            else:
                is_pro = status_val in ("active", "trialing", "grace_period")

        elif tier in ("founder", "lifetime"):
            active = status_val in ("active", "trialing", "grace_period", "none")
            is_pro = active
            is_lifetime = active
            is_founder = (tier == "founder") and active

        return EntitlementResponse(
            user_id=user_id,
            tier=tier,
            status=status_val,
            product_id=product_id,
            source=source,
            expires_at=expires_at,
            auto_renew=auto_renew,
            is_pro=is_pro,
            is_lifetime=is_lifetime,
            is_founder=is_founder,
            is_early_adopter=is_early_adopter,
            account_created_at=account_created_at,
            first_seen_version=first_seen_version,
            updated_at=updated_at,
        )

    # -------------------------------------------------------------------------
    # Founder Code Redemption
    # -------------------------------------------------------------------------

    async def redeem_founder_code(
        self, user_id: str, raw_code: str, client_key: str
    ) -> EntitlementResponse:
        """
        Redeems a founder invitation code atomically.
        Rate-limited: max 6 failed attempts per 900 seconds per (user_id, client_key).
        """
        _check_redemption_rate_limit(user_id, client_key)

        normalized = raw_code.strip().upper()
        if not normalized:
            _record_failed_attempt(user_id, client_key)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Founder code cannot be empty",
            )

        code_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        headers = _get_supabase_headers()
        client = http_pool.get_client()

        # 1. Try atomic Supabase RPC if installed
        rpc_url = _get_supabase_rest_url("rpc/redeem_founder_code_atomic")
        rpc_payload = {"p_code_hash": code_hash, "p_user_id": user_id}

        try:
            res = await client.post(rpc_url, headers=headers, json=rpc_payload)
            if res.status_code == 200:
                result = res.json()
                row = (
                    result[0]
                    if isinstance(result, list) and result
                    else (result if isinstance(result, dict) else None)
                )
                if isinstance(row, dict) and (
                    row.get("ok") is False or row.get("error_code") or row.get("error")
                ):
                    err = str(row.get("error_code") or row.get("error") or "unknown")
                    return self._handle_redemption_error(err, user_id, client_key)
                return await self.get_entitlement(user_id)

            if res.status_code == 404:
                # RPC not installed; fallback to direct PostgREST transaction logic
                return await self._redeem_code_direct(code_hash, user_id, client_key)

            # Check RPC error response
            err_text = res.text.lower()
            return self._handle_redemption_error(err_text, user_id, client_key)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error calling redeem_founder_code_atomic RPC: {e}")
            return await self._redeem_code_direct(code_hash, user_id, client_key)

    def _handle_redemption_error(self, err_text: str, user_id: str, client_key: str) -> None:
        if "invalid_code" in err_text:
            _record_failed_attempt(user_id, client_key)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid founder code.",
            )
        if "code_expired" in err_text:
            _record_failed_attempt(user_id, client_key)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This founder code has expired.",
            )
        if "code_exhausted" in err_text:
            _record_failed_attempt(user_id, client_key)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This founder code has reached its maximum redemptions.",
            )
        if "already_redeemed" in err_text:
            # Not counted against brute-force limit
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This account has already redeemed a founder grant.",
            )
        _record_failed_attempt(user_id, client_key)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to redeem founder code.",
        )

    async def _redeem_code_direct(
        self, code_hash: str, user_id: str, client_key: str
    ) -> EntitlementResponse:
        headers = _get_supabase_headers()
        client = http_pool.get_client()

        # Check existing entitlement
        curr = await self.get_entitlement(user_id)
        if curr.tier == "founder":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This account has already redeemed a founder grant.",
            )

        code_url = _get_supabase_rest_url(
            f"founder_redemption_codes?code_hash=eq.{code_hash}&select=*"
        )
        c_res = await client.get(code_url, headers=headers)
        if c_res.status_code != 200 or not c_res.json():
            _record_failed_attempt(user_id, client_key)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid founder code.",
            )

        code_row = c_res.json()[0]

        # Check expiration
        expires_at_str = code_row.get("expires_at")
        if expires_at_str:
            exp_dt = datetime.fromisoformat(expires_at_str.replace("Z", "+00:00"))
            if datetime.now(timezone.utc) > exp_dt:
                _record_failed_attempt(user_id, client_key)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="This founder code has expired.",
                )

        # Check max redemptions
        redemption_count = code_row.get("redemption_count", 0)
        max_redemptions = code_row.get("max_redemptions", 1)
        if redemption_count >= max_redemptions:
            _record_failed_attempt(user_id, client_key)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This founder code has reached its maximum redemptions.",
            )

        grant_tier = code_row.get("grant_tier", "founder")

        # Increment redemption_count
        patch_code_url = _get_supabase_rest_url(
            f"founder_redemption_codes?code_hash=eq.{code_hash}"
        )
        await client.patch(
            patch_code_url,
            headers=headers,
            json={"redemption_count": redemption_count + 1},
        )

        # Upgrade user entitlement
        patch_ent_url = _get_supabase_rest_url(f"user_entitlements?user_id=eq.{user_id}")
        await client.patch(
            patch_ent_url,
            headers=headers,
            json={
                "tier": grant_tier,
                "status": "active",
                "source": "founder_code",
                "expires_at": None,
                "auto_renew": False,
            },
        )

        # Mark early adopter in user_profiles
        patch_prof_url = _get_supabase_rest_url(f"user_profiles?user_id=eq.{user_id}")
        await client.patch(
            patch_prof_url,
            headers=headers,
            json={"is_early_adopter": True},
        )

        return await self.get_entitlement(user_id)

    # -------------------------------------------------------------------------
    # Apple StoreKit 2 Direct Verification & App Store Server Notifications V2
    # -------------------------------------------------------------------------

    def _verify_apple_jws(self, signed_jws: str) -> Dict[str, Any]:
        """
        Cryptographically verifies Apple StoreKit 2 JWS signatures.
        Validates the x5c certificate chain against Apple Root CA - G3 or Apple Inc. Root Certificate.
        Checks validity windows and verifies ES256 signature using the leaf cert's public key.
        """
        try:
            header = jwt.get_unverified_header(signed_jws)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Malformed StoreKit JWS header: {e}",
            )

        # Allow Xcode StoreKit Testing bypass only when settings.debug is True
        if settings.debug:
            try:
                unverified_payload = jwt.decode(
                    signed_jws, options={"verify_signature": False}
                )
                if unverified_payload.get("environment") == "Xcode":
                    logger.info("Allowing Xcode StoreKit Testing JWS in debug mode")
                    return unverified_payload
            except Exception:
                pass

        x5c = header.get("x5c")
        if not x5c or not isinstance(x5c, list) or len(x5c) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="StoreKit JWS header must contain at least 2 certificates in x5c chain",
            )

        certs: List[x509.Certificate] = []
        try:
            for cert_b64 in x5c:
                cert_der = base64.b64decode(cert_b64)
                cert = x509.load_der_x509_certificate(cert_der)
                certs.append(cert)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to parse x5c certificate chain: {e}",
            )

        now = datetime.now(timezone.utc)
        for i, cert in enumerate(certs):
            if not (cert.not_valid_before_utc <= now <= cert.not_valid_after_utc):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"StoreKit certificate #{i} is not within its validity period",
                )

        # Verify each certificate's signature against the next cert in chain
        for i in range(len(certs) - 1):
            try:
                certs[i].verify_directly_issued_by(certs[i + 1])
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"StoreKit certificate chain signature check failed at index {i}: {e}",
                )

        # Validate root certificate SHA-256 fingerprint
        root_cert = certs[-1]
        root_fp = root_cert.fingerprint(hashes.SHA256()).hex().lower()
        if root_fp not in (APPLE_ROOT_CA_G3_FP, APPLE_INC_ROOT_FP):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"StoreKit root certificate fingerprint does not match Apple Root CA ({root_fp})",
            )

        # Verify JWS signature using leaf certificate's public key
        leaf_pubkey = certs[0].public_key()
        try:
            payload = jwt.decode(
                signed_jws,
                leaf_pubkey,
                algorithms=["ES256"],
                options={"verify_aud": False},
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"StoreKit JWS signature verification failed: {e}",
            )

        # Validate bundle ID matches configuration
        if settings.apple_bundle_id:
            bundle_id = payload.get("bundleId")
            if bundle_id and bundle_id != settings.apple_bundle_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"StoreKit bundleId mismatch: expected {settings.apple_bundle_id}, got {bundle_id}",
                )

        return payload

    async def verify_apple_purchase(
        self, user_id: str, req: AppleVerifyRequest
    ) -> EntitlementResponse:
        """
        Verifies direct Apple StoreKit 2 transactions or App Store Server API queries.
        Maps products to entitlements without ever downgrading existing founder or lifetime tiers.
        """
        tx_data: Optional[Dict[str, Any]] = None

        if req.signed_transaction_info:
            tx_data = self._verify_apple_jws(req.signed_transaction_info)
        elif req.transaction_id and settings.apple_issuer_id and settings.apple_key_id and settings.apple_private_key_pem:
            tx_data = await self._fetch_apple_transaction_api(req.transaction_id)

        if not tx_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing signed_transaction_info or App Store Connect credentials",
            )

        original_tx_id = str(
            tx_data.get("originalTransactionId") or tx_data.get("transactionId") or ""
        )
        if not original_tx_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Transaction does not contain originalTransactionId",
            )

        product_id = str(tx_data.get("productId") or req.product_id or "")
        expires_date_ms = tx_data.get("expiresDate")
        revocation_date_ms = tx_data.get("revocationDate")

        store_tx_lookup_hash = hashlib.sha256(
            f"apple:{original_tx_id}".encode("utf-8")
        ).hexdigest()

        # Determine target tier and expiration
        status_val = "revoked" if revocation_date_ms else "active"
        if product_id.endswith(".lifetime") or "lifetime" in product_id.lower():
            tier_val = "lifetime"
            expires_at_val = None
            auto_renew_val = False
        else:
            tier_val = "pro"
            auto_renew_val = True
            expires_at_val = (
                datetime.fromtimestamp(expires_date_ms / 1000.0, tz=timezone.utc).isoformat()
                if expires_date_ms
                else None
            )

        # Check existing entitlement to never downgrade founder or lifetime
        current = await self.get_entitlement(user_id)
        final_tier = tier_val
        final_status = status_val
        final_expires = expires_at_val
        final_auto_renew = auto_renew_val

        if current.tier == "founder":
            final_tier = "founder"
            final_status = "active"
            final_expires = None
            final_auto_renew = False
        elif current.tier == "lifetime" and tier_val == "pro":
            final_tier = "lifetime"
            final_status = "active"
            final_expires = None
            final_auto_renew = False

        headers = _get_supabase_headers()
        client = http_pool.get_client()

        patch_url = _get_supabase_rest_url(f"user_entitlements?user_id=eq.{user_id}")
        await client.patch(
            patch_url,
            headers=headers,
            json={
                "tier": final_tier,
                "status": final_status,
                "product_id": product_id,
                "source": "app_store",
                "store_tx_lookup_hash": store_tx_lookup_hash,
                "expires_at": final_expires,
                "auto_renew": final_auto_renew,
            },
        )

        return await self.get_entitlement(user_id)

    async def _fetch_apple_transaction_api(self, transaction_id: str) -> Dict[str, Any]:
        now_ts = int(time.time())
        claims = {
            "iss": settings.apple_issuer_id,
            "iat": now_ts,
            "exp": now_ts + 300,
            "aud": "appstoreconnect-v1",
            "bid": settings.apple_bundle_id,
        }
        token = jwt.encode(
            claims,
            settings.apple_private_key_pem,
            algorithm="ES256",
            headers={"kid": settings.apple_key_id},
        )

        client = http_pool.get_client()
        urls = [
            f"https://api.storekit.itunes.apple.com/inApps/v1/transactions/{transaction_id}",
            f"https://api.storekit-sandbox.itunes.apple.com/inApps/v1/transactions/{transaction_id}",
        ]

        for url in urls:
            try:
                res = await client.get(url, headers={"Authorization": f"Bearer {token}"})
                if res.status_code == 200:
                    data = res.json()
                    signed_tx = data.get("signedTransactionInfo")
                    if signed_tx:
                        return self._verify_apple_jws(signed_tx)
            except Exception as e:
                logger.warning(f"Error querying StoreKit API at {url}: {e}")

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Apple transaction {transaction_id} not found in production or sandbox",
        )

    async def handle_apple_webhook(self, signed_payload: str) -> Dict[str, Any]:
        """
        Receives and cryptographically verifies App Store Server Notifications V2 JWS payloads.
        Synchronizes renewal, expiration, and refund events into user_entitlements.
        """
        outer = self._verify_apple_jws(signed_payload)
        notification_type = outer.get("notificationType")
        subtype = outer.get("subtype")
        data = outer.get("data", {})
        signed_tx_info = data.get("signedTransactionInfo")

        if signed_tx_info:
            tx = self._verify_apple_jws(signed_tx_info)
            original_tx_id = str(
                tx.get("originalTransactionId") or tx.get("transactionId") or ""
            )
            lookup_hash = hashlib.sha256(
                f"apple:{original_tx_id}".encode("utf-8")
            ).hexdigest()

            headers = _get_supabase_headers()
            client = http_pool.get_client()

            # Find matching entitlement row
            query_url = _get_supabase_rest_url(
                f"user_entitlements?store_tx_lookup_hash=eq.{lookup_hash}&select=*"
            )
            res = await client.get(query_url, headers=headers)
            if res.status_code == 200 and res.json():
                ent = res.json()[0]
                user_id = ent["user_id"]
                current_tier = ent.get("tier", "free")

                # Never downgrade founder tier
                if current_tier != "founder":
                    new_status = ent.get("status", "active")
                    if notification_type in ("REFUND", "REVOKE"):
                        new_status = "revoked"
                    elif notification_type in ("EXPIRED", "GRACE_PERIOD_EXPIRED"):
                        new_status = "expired"
                    elif notification_type == "DID_FAIL_TO_RENEW":
                        new_status = "grace_period"
                    elif notification_type in ("SUBSCRIBED", "DID_RENEW"):
                        new_status = "active"

                    expires_at = ent.get("expires_at")
                    if tx.get("expiresDate"):
                        expires_at = datetime.fromtimestamp(
                            tx["expiresDate"] / 1000.0, tz=timezone.utc
                        ).isoformat()

                    patch_url = _get_supabase_rest_url(
                        f"user_entitlements?user_id=eq.{user_id}"
                    )
                    await client.patch(
                        patch_url,
                        headers=headers,
                        json={"status": new_status, "expires_at": expires_at},
                    )
                    logger.info(
                        f"Apple webhook updated user {user_id}: status={new_status}, notif={notification_type}"
                    )

        return {"status": "processed", "notificationType": notification_type}

    # -------------------------------------------------------------------------
    # Google Play Billing Direct Verification & RTDN Webhook
    # -------------------------------------------------------------------------

    async def _get_google_access_token(self) -> str:
        global _GOOGLE_TOKEN_CACHE
        now = time.time()
        if _GOOGLE_TOKEN_CACHE.get("token") and now < _GOOGLE_TOKEN_CACHE.get("expires_at", 0) - 60:
            return _GOOGLE_TOKEN_CACHE["token"]

        if not settings.google_service_account_json:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Google Service Account credentials are not configured",
            )

        try:
            raw = settings.google_service_account_json.strip()
            sa_data = json.loads(raw)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to parse google_service_account_json: {e}",
            )

        client_email = sa_data.get("client_email")
        private_key = sa_data.get("private_key")
        token_uri = sa_data.get("token_uri", "https://oauth2.googleapis.com/token")

        if not client_email or not private_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Google service account JSON is missing client_email or private_key",
            )

        iat = int(now)
        claims = {
            "iss": client_email,
            "scope": "https://www.googleapis.com/auth/androidpublisher",
            "aud": token_uri,
            "exp": iat + 3600,
            "iat": iat,
        }
        assertion = jwt.encode(claims, private_key, algorithm="RS256")

        client = http_pool.get_client()
        res = await client.post(
            token_uri,
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                "assertion": assertion,
            },
        )
        if res.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Google OAuth token exchange failed: {res.text}",
            )

        data = res.json()
        token = data["access_token"]
        expires_in = data.get("expires_in", 3600)

        _GOOGLE_TOKEN_CACHE["token"] = token
        _GOOGLE_TOKEN_CACHE["expires_at"] = now + expires_in
        return token

    async def verify_google_purchase(
        self, user_id: str, req: GoogleVerifyRequest
    ) -> EntitlementResponse:
        """
        Directly queries the Google Play Android Publisher API v3 to verify in-app purchases or subscriptions.
        Automatically acknowledges unacknowledged tokens.
        """
        token = await self._get_google_access_token()
        pkg = settings.google_play_package_name
        client = http_pool.get_client()
        auth_hdr = {"Authorization": f"Bearer {token}"}

        is_sub = (
            req.is_subscription
            if req.is_subscription is not None
            else ("lifetime" not in req.product_id.lower())
        )

        tier_val = "pro" if is_sub else "lifetime"
        status_val = "active"
        expires_at_val: Optional[str] = None
        auto_renew_val = is_sub

        if is_sub:
            sub_url = f"https://androidpublisher.googleapis.com/androidpublisher/v3/applications/{pkg}/purchases/subscriptionsv2/tokens/{req.purchase_token}"
            res = await client.get(sub_url, headers=auth_hdr)
            if res.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Google Play subscription verification failed: {res.text}",
                )
            sub_data = res.json()
            sub_state = sub_data.get("subscriptionState", "")

            # Check acknowledgement
            line_items = sub_data.get("lineItems", [{}])
            exp_time = line_items[0].get("expiryTime") if line_items else None
            expires_at_val = exp_time

            if sub_data.get("acknowledgementState") == "ACKNOWLEDGEMENT_STATE_PENDING":
                ack_url = f"https://androidpublisher.googleapis.com/androidpublisher/v3/applications/{pkg}/purchases/subscriptions/{req.product_id}/tokens/{req.purchase_token}:acknowledge"
                await client.post(ack_url, headers=auth_hdr, json={})

            if "EXPIRED" in sub_state or "REVOKED" in sub_state:
                status_val = "expired"
        else:
            # One-time product (lifetime)
            prod_url = f"https://androidpublisher.googleapis.com/androidpublisher/v3/applications/{pkg}/purchases/products/{req.product_id}/tokens/{req.purchase_token}"
            res = await client.get(prod_url, headers=auth_hdr)
            if res.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Google Play one-time product verification failed: {res.text}",
                )
            prod_data = res.json()
            if prod_data.get("purchaseState") != 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Google Play purchase is not in valid purchased state",
                )
            if prod_data.get("acknowledgementState") == 0:
                ack_url = f"https://androidpublisher.googleapis.com/androidpublisher/v3/applications/{pkg}/purchases/products/{req.product_id}/tokens/{req.purchase_token}:acknowledge"
                await client.post(ack_url, headers=auth_hdr, json={})

            tier_val = "lifetime"
            status_val = "active"
            expires_at_val = None
            auto_renew_val = False

        store_tx_lookup_hash = hashlib.sha256(
            f"google:{req.purchase_token}".encode("utf-8")
        ).hexdigest()

        # Check existing entitlement to never downgrade founder or lifetime
        current = await self.get_entitlement(user_id)
        final_tier = tier_val
        final_status = status_val
        final_expires = expires_at_val
        final_auto_renew = auto_renew_val

        if current.tier == "founder":
            final_tier = "founder"
            final_status = "active"
            final_expires = None
            final_auto_renew = False
        elif current.tier == "lifetime" and tier_val == "pro":
            final_tier = "lifetime"
            final_status = "active"
            final_expires = None
            final_auto_renew = False

        headers = _get_supabase_headers()
        patch_url = _get_supabase_rest_url(f"user_entitlements?user_id=eq.{user_id}")
        await client.patch(
            patch_url,
            headers=headers,
            json={
                "tier": final_tier,
                "status": final_status,
                "product_id": req.product_id,
                "source": "play_store",
                "store_tx_lookup_hash": store_tx_lookup_hash,
                "expires_at": final_expires,
                "auto_renew": final_auto_renew,
            },
        )

        return await self.get_entitlement(user_id)

    async def handle_google_rtdn(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Receives Google Cloud Pub/Sub Real-Time Developer Notifications (RTDN).
        Re-verifies transaction state and updates user_entitlements.
        """
        data_dict = body
        message = body.get("message")
        if message and isinstance(message, dict) and message.get("data"):
            try:
                decoded = base64.b64decode(message["data"]).decode("utf-8")
                data_dict = json.loads(decoded)
            except Exception as e:
                logger.warning(f"Failed to decode Google RTDN Pub/Sub data: {e}")

        sub_notif = data_dict.get("subscriptionNotification")
        otp_notif = data_dict.get("oneTimeProductNotification")
        purchase_token = None

        if sub_notif:
            purchase_token = sub_notif.get("purchaseToken")
        elif otp_notif:
            purchase_token = otp_notif.get("purchaseToken")

        if purchase_token:
            lookup_hash = hashlib.sha256(
                f"google:{purchase_token}".encode("utf-8")
            ).hexdigest()

            headers = _get_supabase_headers()
            client = http_pool.get_client()

            query_url = _get_supabase_rest_url(
                f"user_entitlements?store_tx_lookup_hash=eq.{lookup_hash}&select=*"
            )
            res = await client.get(query_url, headers=headers)
            if res.status_code == 200 and res.json():
                ent = res.json()[0]
                user_id = ent["user_id"]
                current_tier = ent.get("tier", "free")

                if current_tier != "founder":
                    notif_type = sub_notif.get("notificationType") if sub_notif else None
                    new_status = ent.get("status", "active")

                    # RTDN Notification Types:
                    # 1=RECOVERED, 2=RENEWED, 3=CANCELED, 4=PURCHASED, 5=ON_HOLD,
                    # 6=IN_GRACE_PERIOD, 7=EXPIRED, 12=REVOKED, 13=EXPIRED
                    if notif_type in (1, 2, 4):
                        new_status = "active"
                    elif notif_type in (5, 6):
                        new_status = "grace_period"
                    elif notif_type in (7, 13):
                        new_status = "expired"
                    elif notif_type == 12:
                        new_status = "revoked"

                    patch_url = _get_supabase_rest_url(
                        f"user_entitlements?user_id=eq.{user_id}"
                    )
                    await client.patch(
                        patch_url,
                        headers=headers,
                        json={"status": new_status},
                    )
                    logger.info(
                        f"Google RTDN updated user {user_id}: status={new_status}, notifType={notif_type}"
                    )

        return {"status": "processed"}

    # -------------------------------------------------------------------------
    # Account Deletion (GDPR / CCPA / Store Compliance)
    # -------------------------------------------------------------------------

    async def delete_account(self, user_id: str) -> Dict[str, Any]:
        """
        Completely removes user from Supabase Auth admin API, cascading profile and entitlement tables.
        Guarantees zero residual state.
        """
        headers = _get_supabase_headers()
        client = http_pool.get_client()

        # Delete auth user
        auth_del_url = _get_supabase_auth_url(f"admin/users/{user_id}")
        await client.delete(auth_del_url, headers=headers)

        # Ensure database tables are purged
        await client.delete(
            _get_supabase_rest_url(f"user_entitlements?user_id=eq.{user_id}"),
            headers=headers,
        )
        await client.delete(
            _get_supabase_rest_url(f"user_profiles?user_id=eq.{user_id}"),
            headers=headers,
        )
        await client.delete(
            _get_supabase_rest_url(f"user_sync_state?user_id=eq.{user_id}"),
            headers=headers,
        )

        return {"deleted": True, "user_id": user_id}

    # -------------------------------------------------------------------------
    # Admin Founder Management
    # -------------------------------------------------------------------------

    async def create_founder_codes(
        self, req: CreateFounderCodesRequest
    ) -> CreateFounderCodesResponse:
        """
        Generates formatted ZUS-FOUNDER-XXXX-XXXX codes.
        Stores SHA-256 hashes and prefixes for secure, zero-plaintext storage.
        """
        headers = _get_supabase_headers()
        client = http_pool.get_client()

        codes: List[str] = []
        rows: List[Dict[str, Any]] = []

        expires_at_iso: Optional[str] = None
        if req.expires_days:
            expires_at_iso = (
                datetime.now(timezone.utc) + timedelta(days=req.expires_days)
            ).isoformat()

        for _ in range(req.count):
            p1 = "".join(secrets.choice(FOUNDER_CODE_CHARSET) for _ in range(4))
            p2 = "".join(secrets.choice(FOUNDER_CODE_CHARSET) for _ in range(4))
            code = f"ZUS-FOUNDER-{p1}-{p2}"
            codes.append(code)

            code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
            code_prefix = code[:15]

            rows.append(
                {
                    "code_hash": code_hash,
                    "code_prefix": code_prefix,
                    "grant_tier": req.grant_tier,
                    "max_redemptions": req.max_redemptions,
                    "redemption_count": 0,
                    "note": req.note,
                    "expires_at": expires_at_iso,
                }
            )

        insert_url = _get_supabase_rest_url("founder_redemption_codes")
        res = await client.post(insert_url, headers=headers, json=rows)
        if res.status_code not in (200, 201):
            logger.error(f"Failed to insert founder codes: {res.text}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to persist founder redemption codes",
            )

        return CreateFounderCodesResponse(
            codes=codes,
            grant_tier=req.grant_tier,
            max_redemptions=req.max_redemptions,
        )

    async def admin_grant_founder(
        self, req: AdminGrantFounderRequest
    ) -> AdminGrantFounderResponse:
        """
        Manually grants founder or lifetime status to a specific user, or bulk upgrades
        early adopters registered prior to a specified ISO timestamp or semantic version.
        """
        headers = _get_supabase_headers()
        client = http_pool.get_client()

        if req.user_id:
            # Grant specific user
            patch_ent_url = _get_supabase_rest_url(
                f"user_entitlements?user_id=eq.{req.user_id}"
            )
            await client.patch(
                patch_ent_url,
                headers=headers,
                json={
                    "tier": req.grant_tier,
                    "status": "active",
                    "source": "manual_admin",
                    "expires_at": None,
                    "auto_renew": False,
                },
            )
            patch_prof_url = _get_supabase_rest_url(
                f"user_profiles?user_id=eq.{req.user_id}"
            )
            await client.patch(
                patch_prof_url,
                headers=headers,
                json={"is_early_adopter": True},
            )
            return AdminGrantFounderResponse(updated_count=1, grant_tier=req.grant_tier)

        if req.before_created_at or req.before_version:
            # Fetch all user profiles
            query_url = _get_supabase_rest_url(
                "user_profiles?select=user_id,account_created_at,first_seen_version,is_early_adopter"
            )
            res = await client.get(query_url, headers=headers)
            if res.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to retrieve user profiles for bulk grant",
                )

            profiles = res.json()
            updated_count = 0

            target_date: Optional[datetime] = None
            if req.before_created_at:
                target_date = datetime.fromisoformat(
                    req.before_created_at.replace("Z", "+00:00")
                )

            target_version = _parse_semver(req.before_version) if req.before_version else None

            for prof in profiles:
                u_id = prof["user_id"]
                match = False

                if target_date and prof.get("account_created_at"):
                    try:
                        c_date = datetime.fromisoformat(
                            prof["account_created_at"].replace("Z", "+00:00")
                        )
                        if c_date <= target_date:
                            match = True
                    except Exception:
                        pass

                if target_version and prof.get("first_seen_version"):
                    prof_ver = _parse_semver(prof["first_seen_version"])
                    if prof_ver and prof_ver < target_version:
                        match = True

                if match:
                    patch_ent = _get_supabase_rest_url(
                        f"user_entitlements?user_id=eq.{u_id}"
                    )
                    await client.patch(
                        patch_ent,
                        headers=headers,
                        json={
                            "tier": req.grant_tier,
                            "status": "active",
                            "source": "early_adopter_grant",
                            "expires_at": None,
                            "auto_renew": False,
                        },
                    )
                    patch_prof = _get_supabase_rest_url(f"user_profiles?user_id=eq.{u_id}")
                    await client.patch(
                        patch_prof,
                        headers=headers,
                        json={"is_early_adopter": True},
                    )
                    updated_count += 1

            return AdminGrantFounderResponse(
                updated_count=updated_count, grant_tier=req.grant_tier
            )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must specify user_id, before_created_at, or before_version",
        )


account_service = AccountService()
