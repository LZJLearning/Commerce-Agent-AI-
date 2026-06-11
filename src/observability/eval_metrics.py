"""
Evaluation metrics for the Commerce Agent.

Two-layer architecture:
1. Local scoring functions — run in-process on bundle data.
2. Langfuse score fetcher — pulls LLM-as-judge scores from the
   Langfuse API for live dashboard rendering.

=========== ======================================================
Metric      Description
=========== ======================================================
budget      预算约束合规率
intent      意图对齐度 (LLM judge)
grounded    严谨性无幻觉率 (LLM judge)
tool_prec   工具调用精确度
=========== ======================================================
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

# ── Lazy Langfuse import ──────────────────────────────────────────────
try:
    from langfuse import Langfuse  # type: ignore
    _LANGFUSE_AVAILABLE = True
except ImportError:
    _LANGFUSE_AVAILABLE = False


# ── Local scoring (runs in-process) ───────────────────────────────────

REQUIRED_CATEGORIES = ["CPU", "显卡", "主板", "内存", "固态硬盘", "电源", "散热器", "机箱"]


def budget_compliance(
    total_price: float,
    user_budget: float,
    tolerance_pct: float = 5.0,
) -> dict[str, Any]:
    upper_bound = user_budget * (1 + tolerance_pct / 100)
    compliant = total_price <= upper_bound
    if total_price <= user_budget:
        score = 1.0
    elif compliant:
        score = 0.7
    else:
        exceed_pct = (total_price - user_budget) / user_budget * 100
        score = max(0, 1.0 - (exceed_pct - tolerance_pct) / 20)
    return {
        "metric": "budget_compliance",
        "score": round(score, 3),
        "compliant": compliant,
    }


def bundle_completeness(recommended_categories: list[str]) -> dict[str, Any]:
    missing = [c for c in REQUIRED_CATEGORIES if c not in recommended_categories]
    score = 1.0 - len(missing) / len(REQUIRED_CATEGORIES)
    return {
        "metric": "bundle_completeness",
        "score": round(score, 3),
        "covered": len(REQUIRED_CATEGORIES) - len(missing),
        "missing": missing,
    }


def evaluate_recommendation(
    total_price: float,
    user_budget: float,
    recommended_categories: list[str],
    matched_keywords: int = 0,
    total_keywords: int = 0,
) -> dict[str, Any]:
    """Run local metrics including recall, precision, F1."""
    b = budget_compliance(total_price, user_budget)
    c = bundle_completeness(recommended_categories)

    # ---- Precision & Recall ----
    # Recall = categories covered / required
    recall = c["score"]

    # Precision = how many items match user's stated preferences
    if total_keywords > 0:
        precision = min(1.0, matched_keywords / max(total_keywords, 1))
    else:
        precision = 1.0  # no specific preference → perfect precision by default

    # F1 = 2 * P * R / (P + R)
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    overall = b["score"] * 0.3 + recall * 0.25 + precision * 0.25 + f1 * 0.2

    return {
        "overall_score": round(overall, 3),
        "budget_compliance": b,
        "bundle_completeness": c,
        "recall": {"score": round(recall, 3), "covered": c["covered"], "total": 8},
        "precision": {"score": round(precision, 3)},
        "f1": {"score": round(f1, 3)},
    }


# ── Langfuse score fetcher ────────────────────────────────────────────

# Score names we expect from the LLM-as-judge eval pipeline
_EVAL_DIMENSIONS = {
    "预算约束合规率":    {"key": "budget_compliance",   "max": 1.0, "icon": "💰"},
    "意图对齐度":        {"key": "intent_alignment",   "max": 5.0, "icon": "🎯"},
    "严谨性无幻觉率":    {"key": "groundedness",       "max": 5.0, "icon": "📋"},
    "工具调用精确度":    {"key": "tool_precision",     "max": 5.0, "icon": "🔧"},
}

# In-memory cache: {session_id: {scores, fetched_at}}
_eval_cache: dict[str, dict[str, Any]] = {}
_CACHE_TTL_S = 15  # seconds before a re-fetch is allowed


def get_session_eval_scores(
    session_id: str,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """
    Fetch evaluation scores from Langfuse for *session_id*.

    Returns a dict with keys matching _EVAL_DIMENSIONS labels, plus
    metadata fields ``fetched`` (bool), ``timestamp`` (float), and
    ``source`` (str).

    When Langfuse is unavailable returns a stub with ``fetched=False``.
    """
    now = time.time()

    # Return cached if fresh enough
    if not force_refresh and session_id in _eval_cache:
        entry = _eval_cache[session_id]
        if (now - entry.get("fetched_at", 0)) < _CACHE_TTL_S:
            return entry["data"]

    if not _LANGFUSE_AVAILABLE:
        stub = _build_stub("Langfuse SDK not installed")
        _update_cache(session_id, stub, now)
        return stub

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or "pk-lf-" not in public_key:
        stub = _build_stub("Langfuse keys not configured")
        _update_cache(session_id, stub, now)
        return stub

    try:
        client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)

        # Fetch scores via the Langfuse REST API
        # filter by session_id and also by name (the four eval dimensions)
        response = client.api.scores.get_many(
            session_id=session_id,
            limit=50,
        )
        raw_scores = list(response.data) if hasattr(response, "data") else []

        result = _parse_scores(raw_scores)
        _update_cache(session_id, result, now)
        return result

    except Exception as exc:
        stub = _build_stub(f"Fetch error: {exc}")
        _update_cache(session_id, stub, now)
        return stub


def _session_matches(score_obj: Any, session_id: str) -> bool:
    """Check if a Langfuse score object belongs to *session_id*."""
    try:
        if hasattr(score_obj, "session_id"):
            return score_obj.session_id == session_id
        if hasattr(score_obj, "trace"):
            trace = score_obj.trace
            if hasattr(trace, "session_id"):
                return trace.session_id == session_id
        if isinstance(score_obj, dict):
            sid = score_obj.get("session_id", "")
            if sid:
                return sid == session_id
    except Exception:
        pass
    return False


def _parse_scores(raw_scores: list[Any]) -> dict[str, Any]:
    """Convert raw Langfuse score objects (Pydantic models or dicts) into dashboard format."""
    result: dict[str, dict[str, Any]] = {}
    for s in raw_scores:
        # Support both Pydantic model and plain dict
        if hasattr(s, "model_dump"):
            d = s.model_dump()
        elif hasattr(s, "dict"):
            d = s.dict()
        elif isinstance(s, dict):
            d = s
        else:
            continue

        name = d.get("name", "")
        value = d.get("value", 0)
        comment = d.get("comment", "")

        # Map score names to our dimensions
        matched = False
        for label, dim in _EVAL_DIMENSIONS.items():
            if dim["key"] in name.lower().replace(" ", "_"):
                result[label] = {
                    "value": float(value),
                    "max": dim["max"],
                    "ratio": round(float(value) / dim["max"], 3),
                    "comment": comment or "",
                    "icon": dim["icon"],
                }
                matched = True
                break
        if not matched:
            result[name] = {
                "value": float(value),
                "max": 5.0,
                "ratio": round(float(value) / 5.0, 3),
                "comment": comment or "",
                "icon": "📊",
            }

    # Fill missing dimensions
    for label, dim in _EVAL_DIMENSIONS.items():
        if label not in result:
            result[label] = {
                "value": None,
                "max": dim["max"],
                "ratio": None,
                "comment": "评估数据计算中...",
                "icon": dim["icon"],
            }

    return {
        "fetched": len(raw_scores) > 0,
        "source": "langfuse",
        "timestamp": time.time(),
        "scores": result,
        "raw_count": len(raw_scores),
    }


def _build_stub(reason: str) -> dict[str, Any]:
    """Return a stub result when Langfuse is not reachable."""
    scores = {}
    for label, dim in _EVAL_DIMENSIONS.items():
        scores[label] = {
            "value": None,
            "max": dim["max"],
            "ratio": None,
            "comment": "评估数据计算中...",
            "icon": dim["icon"],
        }
    return {
        "fetched": False,
        "source": "stub",
        "reason": reason,
        "timestamp": time.time(),
        "scores": scores,
        "raw_count": 0,
    }


def _update_cache(session_id: str, data: dict[str, Any], now: float) -> None:
    _eval_cache[session_id] = {"data": data, "fetched_at": now}


def clear_eval_cache(session_id: str | None = None) -> None:
    """Clear cached eval scores. If session_id is None, clears all."""
    if session_id is None:
        _eval_cache.clear()
    else:
        _eval_cache.pop(session_id, None)
