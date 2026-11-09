from fastapi import APIRouter, HTTPException, Query, Body, status
from typing import Optional, Dict, Any

from app.models.notifications import (
    PushSubscriptionRegisterRequest,
    PushSubscriptionResponse,
    PushSubscriptionUnregisterRequest,
    NotificationStatsResponse,
)
from app.services.push_worker import push_worker

router = APIRouter(prefix="/notifications", tags=["Notifications"])


@router.post(
    "/register",
    response_model=PushSubscriptionResponse,
    status_code=status.HTTP_200_OK,
    summary="Register or update client push notification subscription",
    description=(
        "Registers a client device for spatial precipitation and severe weather push notifications. "
        "Server quantizes coordinates to 0.05° (~5km) cells and overwrites in-place without retaining "
        "historical trails."
    ),
)
async def register_subscription(
    payload: PushSubscriptionRegisterRequest,
) -> PushSubscriptionResponse:
    if not payload.device_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="device_id cannot be empty",
        )
    return await push_worker.register_subscription(payload)


@router.delete(
    "/unregister",
    status_code=status.HTTP_200_OK,
    summary="Unregister client push notification subscription",
    description="Removes device subscription and stops push notifications.",
)
async def unregister_subscription(
    device_id: Optional[str] = Query(None, description="Device identifier to unregister"),
    body: Optional[PushSubscriptionUnregisterRequest] = Body(None),
) -> Dict[str, Any]:
    target_id = (device_id or (body.device_id if body else None) or "").strip()
    if not target_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="device_id query parameter or body is required",
        )

    deleted = await push_worker.unregister_subscription(target_id)
    return {
        "device_id": target_id,
        "unregistered": deleted,
        "status": "success" if deleted else "not_found",
    }


@router.get(
    "/stats",
    response_model=NotificationStatsResponse,
    summary="Retrieve spatial subscription registry diagnostics and stats",
    description="Returns aggregate subscription numbers, spatial cells, cell state machine counts, and recent dispatch metrics.",
)
async def get_notification_stats() -> NotificationStatsResponse:
    return await push_worker.get_stats()


@router.post(
    "/evaluate",
    summary="Manually trigger an anti-spam spatial evaluation pass",
    description="Executes a single cycle of the O(1) cell evaluation engine for testing and verification.",
)
async def trigger_evaluation() -> Dict[str, Any]:
    return await push_worker.evaluate_and_dispatch()
