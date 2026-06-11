"""
Stripe payment client for the Commerce Agent.

Handles creation of Stripe Checkout Sessions so users can pay
for their recommended PC build directly within the chat.
"""

from __future__ import annotations

import os
from typing import Any

# Lazy import — Stripe may not be installed in all environments
try:
    import stripe  # type: ignore
    _STRIPE_AVAILABLE = True
except ImportError:
    _STRIPE_AVAILABLE = False


# ── Stripe Configuration ─────────────────────────────────────────────

def _get_stripe_client() -> Any:
    """
    Initialise and return the Stripe client.

    Reads STRIPE_SECRET_KEY from environment (set by src.config via
    python-dotenv).  The caller should ensure load_dotenv() has been
    called before invoking any Stripe function.
    """
    if not _STRIPE_AVAILABLE:
        raise ImportError(
            "stripe package is not installed. Run: pip install stripe"
        )
    secret_key = os.getenv("STRIPE_SECRET_KEY", "")
    if not secret_key:
        raise ValueError(
            "STRIPE_SECRET_KEY is not set. "
            "Please configure it in your .env file or environment."
        )
    stripe.api_key = secret_key
    return stripe


# ── Product / Price helpers ──────────────────────────────────────────

def _build_line_items(
    hardware_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Convert hardware bundle items into Stripe Checkout line items.

    Each item is modelled as an ad-hoc product via `price_data`.
    In production, you would pre-create Stripe Products & Prices
    and reference them by ID.
    """
    line_items: list[dict[str, Any]] = []
    for item in hardware_items:
        # Stripe expects integer in smallest currency unit.
        # We use USD in test mode for maximum compatibility.
        # Convert RMB price → USD (approx rate 7.2 → cents).
        unit_amount = int(item.get("price", 0) / 7.2 * 100)
        if unit_amount < 1:
            unit_amount = 1  # Stripe requires ≥1 for most currencies
        line_items.append({
            "price_data": {
                "currency": "usd",
                "product_data": {
                    "name": item.get("name", "Unknown Hardware"),
                    "description": (
                        f"{item.get('category', '')} | "
                        f"Original: ¥{item.get('price', 0):,.0f} RMB"
                    ),
                },
                "unit_amount": unit_amount,
            },
            "quantity": 1,
        })
    return line_items


# ── create_hardware_bundle_link (Prompt 3 核心函数) ──────────────────

def create_hardware_bundle_link(
    bundle_items: list[dict[str, Any]],
    success_url: str = "https://example.com/success",
    cancel_url: str = "https://example.com/cancel",
) -> str:
    """
    Create a Stripe Checkout Session for a hardware bundle and return
    the payment URL.

    This is the canonical entry point for the Agent tool
    ``generate_checkout_link`` — it accepts a list of hardware items
    (each with id, name, category, price) and returns a Stripe-hosted
    payment page URL.

    Args:
        bundle_items: List of dicts, each containing at minimum:
                      - id (str)
                      - name (str)
                      - category (str)
                      - price (float, RMB)
        success_url:  Redirect target after successful payment.
        cancel_url:   Redirect target if the user cancels.

    Returns:
        A Stripe Checkout Session URL string on success, or an error
        message string prefixed with "ERROR:" on failure.
    """
    stripe_client = _get_stripe_client()
    line_items = _build_line_items(bundle_items)

    try:
        session = stripe_client.checkout.Session.create(
            payment_method_types=["card"],
            line_items=line_items,
            mode="payment",
            success_url=success_url,
            cancel_url=cancel_url,
            metadata={"source": "commerce-agent"},
        )
        return session.url  # ← 直接返回 URL 字符串
    except stripe_client.error.StripeError as e:
        return f"ERROR: Stripe 支付会话创建失败 — {e}"
    except Exception as e:
        return f"ERROR: 未知错误 — {e}"


# ── Create Checkout Session (dict-based, for programmatic use) ───────

def create_checkout_session(
    hardware_items: list[dict[str, Any]],
    success_url: str = "https://example.com/success",
    cancel_url: str = "https://example.com/cancel",
    customer_email: str | None = None,
    metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Create a Stripe Checkout Session for a bundle of hardware items.

    Returns a dict with full session details (useful for UI rendering
    and logging).  Prefer ``create_hardware_bundle_link`` for the
    Agent tool path.

    Args:
        hardware_items:  List of {id, name, category, price}.
        success_url:     URL to redirect after successful payment.
        cancel_url:      URL to redirect if the user cancels.
        customer_email:  Optional pre-fill email.
        metadata:        Optional key-value pairs attached to the session.

    Returns:
        Dict with session_id, checkout_url, and payment_status.
    """
    stripe_client = _get_stripe_client()
    line_items = _build_line_items(hardware_items)

    session_kwargs: dict[str, Any] = {
        "payment_method_types": ["card"],
        "line_items": line_items,
        "mode": "payment",
        "success_url": success_url,
        "cancel_url": cancel_url,
    }

    if customer_email:
        session_kwargs["customer_email"] = customer_email
    if metadata:
        session_kwargs["metadata"] = metadata

    try:
        session = stripe_client.checkout.Session.create(**session_kwargs)
        return {
            "success": True,
            "session_id": session.id,
            "checkout_url": session.url,
            "payment_status": session.payment_status,
        }
    except stripe_client.error.StripeError as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }


# ── Retrieve Session (polling / webhook companion) ───────────────────

def retrieve_checkout_session(session_id: str) -> dict[str, Any]:
    """Retrieve a Stripe Checkout Session by ID (e.g., after webhook)."""
    stripe_client = _get_stripe_client()
    try:
        session = stripe_client.checkout.Session.retrieve(session_id)
        return {
            "success": True,
            "session_id": session.id,
            "payment_status": session.payment_status,
            "amount_total": session.amount_total,
            "currency": session.currency,
            "customer_email": session.customer_details.email
            if session.customer_details else None,
        }
    except stripe_client.error.StripeError as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }


# ── Stripe Webhook helper ────────────────────────────────────────────

def construct_event(
    payload: bytes,
    sig_header: str,
    endpoint_secret: str | None = None,
) -> dict[str, Any]:
    """
    Verify and construct a Stripe webhook event.

    Args:
        payload:          Raw request body (bytes).
        sig_header:       Value of the `Stripe-Signature` header.
        endpoint_secret:  Your webhook signing secret (whsec_...).

    Returns:
        Dict with success flag and the event object (or error).
    """
    stripe_client = _get_stripe_client()
    secret = endpoint_secret or os.getenv("STRIPE_WEBHOOK_SECRET", "")

    if not secret:
        return {
            "success": False,
            "error": "STRIPE_WEBHOOK_SECRET is not configured.",
        }

    try:
        event = stripe_client.Webhook.construct_event(
            payload, sig_header, secret,
        )
        return {"success": True, "event": event}
    except ValueError as e:
        return {"success": False, "error": f"Invalid payload: {e}"}
    except stripe_client.error.SignatureVerificationError as e:
        return {"success": False, "error": f"Invalid signature: {e}"}
