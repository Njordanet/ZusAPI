from typing import Optional, List, Literal
from pydantic import BaseModel, Field


class EntitlementResponse(BaseModel):
    user_id: str
    tier: Literal["free", "pro", "lifetime", "founder"]
    status: Literal["none", "trialing", "active", "grace_period", "expired", "revoked"]
    product_id: Optional[str] = None
    source: str = "none"
    expires_at: Optional[str] = None
    auto_renew: bool = False
    is_pro: bool = False
    is_lifetime: bool = False
    is_founder: bool = False
    is_early_adopter: bool = False
    account_created_at: Optional[str] = None
    first_seen_version: Optional[str] = None
    updated_at: Optional[str] = None


class BootstrapAccountRequest(BaseModel):
    first_seen_version: Optional[str] = None


class RedeemCodeRequest(BaseModel):
    code: str


class AppleVerifyRequest(BaseModel):
    signed_transaction_info: Optional[str] = None
    transaction_receipt: Optional[str] = None
    product_id: Optional[str] = None
    transaction_id: Optional[str] = None


class GoogleVerifyRequest(BaseModel):
    purchase_token: str
    product_id: str
    is_subscription: Optional[bool] = None


class CreateFounderCodesRequest(BaseModel):
    count: int = Field(default=1, ge=1, le=200)
    max_redemptions: int = Field(default=1, ge=1, le=10000)
    grant_tier: Literal["founder", "lifetime"] = "founder"
    note: Optional[str] = None
    expires_days: Optional[int] = Field(default=None, ge=1, le=3650)


class CreateFounderCodesResponse(BaseModel):
    codes: List[str]
    grant_tier: str
    max_redemptions: int


class AdminGrantFounderRequest(BaseModel):
    user_id: Optional[str] = None
    before_created_at: Optional[str] = None
    before_version: Optional[str] = None
    grant_tier: Literal["founder", "lifetime"] = "founder"


class AdminGrantFounderResponse(BaseModel):
    updated_count: int
    grant_tier: str
