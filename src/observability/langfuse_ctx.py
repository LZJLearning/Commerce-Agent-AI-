"""
Langfuse observability context manager (v4.7+ compatible).

Provides tracing, metric collection, and structured logging for every
agent interaction — enabling real-time dashboards and offline evaluation.

Langfuse v4 API:
- ``client.start_observation()`` creates spans/observations
- ``trace_context`` carries trace + session binding
- ``.end(output=...)`` finalises an observation
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

# ── Lazy import Langfuse ──────────────────────────────────────────────
try:
    from langfuse import Langfuse  # type: ignore
    _LANGFUSE_AVAILABLE = True
except ImportError:
    _LANGFUSE_AVAILABLE = False


# ── In-memory metrics store ───────────────────────────────────────────
_metrics_store: dict[str, dict[str, Any]] = {}


def _ensure_session(session_id: str) -> dict[str, Any]:
    if session_id not in _metrics_store:
        _metrics_store[session_id] = {
            "total_tokens_in": 0,
            "total_tokens_out": 0,
            "total_latency_ms": 0.0,
            "call_count": 0,
            "checkout_count": 0,
        }
    return _metrics_store[session_id]


def generate_session_id() -> str:
    return uuid.uuid4().hex  # 32-char hex, Langfuse v4 compatible


def get_session_metrics(session_id: str) -> dict[str, Any]:
    m = _ensure_session(session_id)
    return {
        "total_tokens": m["total_tokens_in"] + m["total_tokens_out"],
        "avg_latency_ms": round(m["total_latency_ms"] / m["call_count"], 1) if m["call_count"] else 0,
        "call_count": m["call_count"],
        "checkout_count": m["checkout_count"],
    }


def record_llm_call(session_id: str, tokens_in: int = 0, tokens_out: int = 0, latency_ms: float = 0.0) -> None:
    m = _ensure_session(session_id)
    m["total_tokens_in"] += tokens_in
    m["total_tokens_out"] += tokens_out
    m["total_latency_ms"] += latency_ms
    m["call_count"] += 1


def record_checkout(session_id: str) -> None:
    m = _ensure_session(session_id)
    m["checkout_count"] += 1


# ── Langfuse singleton ────────────────────────────────────────────────

_langfuse_client: Any = None


def _get_langfuse_client() -> Any:
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client
    if not _LANGFUSE_AVAILABLE:
        _langfuse_client = False
        return None
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    if not public_key or "pk-lf-" not in public_key:
        _langfuse_client = False
        return None
    _langfuse_client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
    print(f"[langfuse] Connected to {host}")
    return _langfuse_client


# ── Trace context manager (Langfuse v4) ───────────────────────────────

@contextmanager
def trace_agent_interaction(
    session_id: str = "unknown",
    user_input: str = "",
    metadata: dict[str, Any] | None = None,
):
    """
    Context manager wrapping one agent turn in a Langfuse trace.

    Yields a ``TraceHandle``.  When Langfuse is unavailable yields a no-op.
    """
    client = _get_langfuse_client()
    trace_id = uuid.uuid4().hex  # 32-char hex required by Langfuse v4

    if client:
        # v4: create a root span as the trace container
        from langfuse.types import TraceContext  # type: ignore
        trace_ctx = TraceContext(
            trace_id=trace_id,
            session_id=session_id,
            user_id="streamlit-user",
        )
        root = client.start_observation(
            trace_context=trace_ctx,
            name="agent-turn",
            as_type="span",
            input=user_input[:2000] if user_input else "",
            metadata=metadata or {},
        )
        handle: TraceHandle = _LiveTraceHandle(client, root, session_id)
    else:
        handle = _NoOpTraceHandle(session_id)

    try:
        yield handle
    finally:
        handle._finalize()


# ── Trace handles ─────────────────────────────────────────────────────

class TraceHandle:
    def llm_span(self, name: str = "", input: Any = "", output: Any = "",
                 tokens_in: int = 0, tokens_out: int = 0, latency_ms: float = 0.0) -> "SpanHandle":
        raise NotImplementedError

    def tool_span(self, name: str = "", input: Any = "", output: Any = "") -> "SpanHandle":
        raise NotImplementedError

    def _finalize(self) -> None:
        pass


class _LiveTraceHandle(TraceHandle):
    def __init__(self, client: Any, root_obs: Any, session_id: str) -> None:
        self._client = client
        self._root = root_obs
        self._session_id = session_id

    def llm_span(self, name="", input="", output="", tokens_in=0, tokens_out=0, latency_ms=0.0):
        child = self._root.start_observation(
            name=f"LLM:{name}",
            as_type="generation",
            input=input if isinstance(input, str) else str(input)[:500],
            output=output if isinstance(output, str) else str(output)[:500],
            model="qwen-plus",
            usage_details={"input": tokens_in, "output": tokens_out} if tokens_in or tokens_out else None,
            metadata={"latency_ms": latency_ms},
        )
        record_llm_call(self._session_id, tokens_in, tokens_out, latency_ms)
        return _LiveSpanHandle(child)

    def tool_span(self, name="", input="", output=""):
        child = self._root.start_observation(
            name=f"TOOL:{name}",
            as_type="tool",
            input=input if isinstance(input, str) else str(input)[:500],
        )
        return _LiveSpanHandle(child, output_data=output)

    def _finalize(self) -> None:
        try:
            self._root.end()
            self._client.flush()
        except Exception:
            pass


class _NoOpTraceHandle(TraceHandle):
    def __init__(self, session_id: str) -> None:
        self._session_id = session_id

    def llm_span(self, name="", input="", output="", tokens_in=0, tokens_out=0, latency_ms=0.0):
        record_llm_call(self._session_id, tokens_in, tokens_out, latency_ms)
        return _NoOpSpanHandle()

    def tool_span(self, name="", input="", output=""):
        return _NoOpSpanHandle()


# ── Span handles ──────────────────────────────────────────────────────

class SpanHandle:
    def end(self, output: Any = None) -> None:
        pass


class _LiveSpanHandle(SpanHandle):
    def __init__(self, obs: Any, output_data: Any = None) -> None:
        self._obs = obs
        self._output_data = output_data

    def end(self, output: Any = None) -> None:
        final = output if output is not None else self._output_data
        if final is not None:
            out_str = final if isinstance(final, str) else str(final)[:1000]
            self._obs.update(output=out_str)
        self._obs.end()


class _NoOpSpanHandle(SpanHandle):
    pass
