"""
Hardware retrieval & analysis tools for the Commerce Agent.

Implements search_hardware (filtering the product DB),
calculate_bundle_price (pricing + budget validation), and
generate_checkout_link (Stripe payment) as callable tools
compatible with OpenAI-compatible function-calling protocols.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

# ── Path resolution ────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PRODUCT_DB_PATH = _PROJECT_ROOT / "data" / "products.json"

# Ensure project root is on sys.path so Stripe client can import config
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _load_products() -> list[dict]:
    """Load the hardware product database from disk."""
    with open(_PRODUCT_DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Tool: search_hardware ──────────────────────────────────────────

def search_hardware(
    query: str = "",
    category: str | None = None,
    max_price: float | None = None,
) -> list[dict[str, Any]]:
    """
    Search the hardware database by keyword, category, and/or max price.

    Args:
        query:     Search keyword matched against name and tags (fuzzy, case-insensitive).
        category:  Optional category filter: CPU / 显卡 / 主板 / 内存 / 固态硬盘 / 电源 / 散热器 / 机箱.
        max_price: Optional upper price bound (RMB).

    Returns:
        List of matching hardware items, each containing id, name, category,
        price, tags, stock, and category-specific fields.
    """
    products = _load_products()
    results: list[dict] = []

    query_lower = query.strip().lower() if query else ""

    for p in products:
        # ── category filter ──
        if category and p.get("category", "") != category:
            continue

        # ── max_price filter ──
        if max_price is not None and p.get("price", 0) > max_price:
            continue

        # ── keyword search ──
        if query_lower:
            name = p.get("name", "").lower()
            tags = [t.lower() for t in p.get("tags", [])]
            searchable = f"{name} {' '.join(tags)}"
            if query_lower not in searchable:
                continue

        # ── only return in-stock items ──
        if p.get("stock", 0) <= 0:
            continue

        results.append(p)

    # Sort: stock-aware (prioritize items with more stock) then price ascending
    results.sort(key=lambda x: (-x.get("stock", 0), x.get("price", 0)))
    return results


# ── Tool: calculate_bundle_price ────────────────────────────────────

def calculate_bundle_price(
    hardware_ids: list[str],
    user_budget: float | None = None,
) -> dict[str, Any]:
    """
    Calculate the total price of a recommended hardware bundle and
    validate it against the user's budget.

    Args:
        hardware_ids: List of hardware IDs to include in the bundle.
        user_budget:  The user's stated budget (RMB). If provided, the result
                       includes budget-fit analysis.

    Returns:
        Dict with:
            - items:           List of {id, name, category, price, stock}
            - total_price:     Sum of all item prices
            - budget:          The user's budget (if provided)
            - over_budget:     Amount exceeding budget (0 if within)
            - within_budget:   bool
            - budget_usage_pct: Percentage of budget consumed
    """
    products = _load_products()
    product_map: dict[str, dict] = {p["id"]: p for p in products}

    items: list[dict] = []
    missing_ids: list[str] = []

    for hid in hardware_ids:
        item = product_map.get(hid)
        if item:
            items.append({
                "id": item["id"],
                "name": item["name"],
                "category": item["category"],
                "price": item["price"],
                "stock": item["stock"],
            })
        else:
            missing_ids.append(hid)

    total_price = sum(i["price"] for i in items)

    result: dict[str, Any] = {
        "items": items,
        "total_price": total_price,
        "item_count": len(items),
        "missing_ids": missing_ids,
    }

    if user_budget is not None:
        over_budget = max(0, total_price - user_budget)
        within_5pct = total_price <= user_budget * 1.05
        result.update({
            "budget": user_budget,
            "over_budget": round(over_budget, 2),
            "within_budget": total_price <= user_budget,
            "within_5pct_tolerance": within_5pct,
            "budget_usage_pct": round(total_price / user_budget * 100, 1) if user_budget > 0 else 0,
        })

    return result


# ── Tool schemas for DashScope / OpenAI function calling ───────────

SEARCH_HARDWARE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_hardware",
        "description": "从硬件数据库中检索匹配的电脑硬件，可按关键词、类别和最高价格筛选。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词，会匹配硬件名称和标签，如 'RTX 4060'、'DDR5'、'性价比之王'",
                },
                "category": {
                    "type": "string",
                    "enum": ["CPU", "显卡", "主板", "内存", "固态硬盘", "电源", "散热器", "机箱"],
                    "description": "硬件类别筛选",
                },
                "max_price": {
                    "type": "number",
                    "description": "最高价格上限（人民币元）",
                },
            },
            "required": [],
        },
    },
}

CALCULATE_BUNDLE_PRICE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculate_bundle_price",
        "description": "计算推荐配置单的总价，并校验是否超出用户预算。",
        "parameters": {
            "type": "object",
            "properties": {
                "hardware_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "配置单中所有硬件的 ID 列表",
                },
                "user_budget": {
                    "type": "number",
                    "description": "用户的总预算（人民币元），用于校验是否超预算",
                },
            },
            "required": ["hardware_ids"],
        },
    },
}

# ── Tool: generate_checkout_link ─────────────────────────────────────

def generate_checkout_link(bundle_items: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Generate a Stripe Checkout payment link for a confirmed hardware bundle.

    Calls the real Stripe API to create a Checkout Session, so the returned
    URL leads to a working Stripe-hosted payment page (test mode).

    Args:
        bundle_items: List of hardware items, each with id, name, category, price.

    Returns:
        Dict with checkout_url and a user-friendly message.
    """
    from src.payment.stripe_client import create_hardware_bundle_link

    url = create_hardware_bundle_link(bundle_items)

    if url.startswith("ERROR:"):
        return {
            "success": False,
            "error": url.removeprefix("ERROR:").strip(),
            "checkout_url": None,
        }

    return {
        "success": True,
        "checkout_url": url,
        "message": (
            f"✅ 支付链接已生成！\n\n"
            f"🔗 {url}\n\n"
            f"💡 测试卡号: 4242 4242 4242 4242 | 任意未来日期 | 任意 CVC"
        ),
    }


GENERATE_CHECKOUT_LINK_SCHEMA = {
    "type": "function",
    "function": {
        "name": "generate_checkout_link",
        "description": (
            "用户主动要求购买时才调用。生成 Stripe 沙盒支付链接。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "bundle_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "硬件 ID"},
                            "name": {"type": "string", "description": "硬件中文名称"},
                            "category": {"type": "string", "description": "硬件类别"},
                            "price": {"type": "number", "description": "价格（人民币元）"},
                        },
                        "required": ["id", "name", "category", "price"],
                    },
                    "description": "用户确认购买的硬件列表，来自之前 calculate_bundle_price 返回的 items。",
                },
            },
            "required": ["bundle_items"],
        },
    },
}

# ── Unified tool registry ──────────────────────────────────────────

TOOL_DEFINITIONS = [
    SEARCH_HARDWARE_SCHEMA,
    CALCULATE_BUNDLE_PRICE_SCHEMA,
    GENERATE_CHECKOUT_LINK_SCHEMA,
]

TOOL_DISPATCHER = {
    "search_hardware": search_hardware,
    "calculate_bundle_price": calculate_bundle_price,
    "generate_checkout_link": generate_checkout_link,
}
