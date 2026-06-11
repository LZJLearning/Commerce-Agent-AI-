"""
Streamlit UI entry point for the Commerce Agent.

Features:
- Streaming text output via OpenAI SDK
- Langfuse tracing (every LLM call + tool invocation)
- Sidebar KPI dashboard (tokens, latency, checkout count)
- Unique session_id per page session
"""

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from src.config import LLM_API_KEY, LLM_ENDPOINT, LLM_MODEL
from src.agent.prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from src.agent.tools import (
    search_hardware,
    calculate_bundle_price,
    TOOL_DISPATCHER,
)
from src.payment.stripe_client import create_checkout_session
from src.observability.langfuse_ctx import (
    generate_session_id,
    get_session_metrics,
    record_checkout,
    trace_agent_interaction,
)
from src.observability.eval_metrics import get_session_eval_scores, evaluate_recommendation, clear_eval_cache
from src.observability.langfuse_ctx import _get_langfuse_client as _get_lf

# ── OpenAI SDK ───────────────────────────────────────────────────────
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

# ── Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI 硬件导购 — 小硬",
    page_icon="🖥️",
    layout="centered",
)

st.markdown("""
<style>
    .stChatMessage { border-radius: 10px; }
    .kpi-box {
        background: #fafafa; border-radius: 6px; padding: 8px 12px;
        margin-bottom: 6px; border-left: 3px solid #5b6af0;
    }
    .kpi-value { font-size: 18px; font-weight: 700; color: #333; }
    .kpi-label { font-size: 11px; color: #888; }
    .kpi-dot { display:inline-block;width:6px;height:6px;border-radius:50%;margin-right:6px; }
    .eval-bar-bg { background:#e8e8e8;border-radius:3px;height:4px;margin-top:3px; }
    .eval-bar-fg { height:4px;border-radius:3px; }
    .sidebar-section { margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)

st.title("AI 电脑硬件智能导购")
st.caption("PC DIY 装机专家 · 告诉我预算和需求，一键生成最优方案")

# ── Session State ────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = generate_session_id()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_bundle" not in st.session_state:
    st.session_state.last_bundle = None
if "checkout_count" not in st.session_state:
    st.session_state.checkout_count = 0
if "last_checkout_url" not in st.session_state:
    st.session_state.last_checkout_url = None
if "eval_scores" not in st.session_state:
    st.session_state.eval_scores = None

SESSION_ID: str = st.session_state.session_id

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Quick Settings")
    budget = st.number_input(
        "Budget (RMB)", min_value=1000, max_value=100000,
        value=6000, step=500,
    )
    scenario = st.selectbox(
        "Use case",
        ["游戏", "视频剪辑", "3D渲染", "编程办公", "AI深度学习", "性价比入门"],
    )
    preference = st.text_input(
        "Hardware preference",
        placeholder="e.g. RTX 4060 Ti, i7-14700K",
    )
    extra_notes = st.text_area(
        "Other notes",
        placeholder="e.g. WiFi, white case, 32GB RAM...",
    )

    st.divider()
    st.header("KPI")

    metrics = get_session_metrics(SESSION_ID)

    st.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-value">{metrics['total_tokens']:,}</div>
        <div class="kpi-label">Tokens</div>
    </div>
    <div class="kpi-box">
        <div class="kpi-value">{metrics['avg_latency_ms']:,} ms</div>
        <div class="kpi-label">Avg latency</div>
    </div>
    <div class="kpi-box">
        <div class="kpi-value">{metrics['call_count']} / {metrics['checkout_count']}</div>
        <div class="kpi-label">LLM calls / Checkouts</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.header("Evaluation")

    refresh_eval = st.button("Refresh scores", use_container_width=True)
    if st.session_state.eval_scores is not None and not refresh_eval:
        eval_data = st.session_state.eval_scores
    else:
        try:
            eval_data = get_session_eval_scores(SESSION_ID, force_refresh=refresh_eval)
        except Exception:
            eval_data = {"fetched": False, "source": "error", "raw_count": 0, "scores": {}}

    if not eval_data.get("fetched"):
        st.caption("Awaiting data...")

    scores = eval_data.get("scores", {}) if eval_data else {}
    for label, s in scores.items():
        val = s["value"]
        max_val = s["max"]
        comment = s.get("comment", "")
        if val is not None:
            pct = val / max_val
            if max_val <= 1.0:
                disp = f"{val:.0%}"
            else:
                disp = f"{val:.1f} / {max_val:.0f}"
            color = "#5b6af0" if pct >= 0.8 else "#e8a020" if pct >= 0.5 else "#d94a4a"
            dot = f"<span class=\"kpi-dot\" style=\"background:{color};\"></span>"
            st.markdown(f"""
            <div class="kpi-box" style="border-left-color:{color};">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span class="kpi-label">{dot} {label}</span>
                    <span class="kpi-value" style="font-size:14px;">{disp}</span>
                </div>
                <div class="eval-bar-bg">
                    <div class="eval-bar-fg" style="width:{pct*100:.0f}%;background:{color};"></div>
                </div>
                <div class="kpi-label" style="margin-top:2px;">{comment}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            dot = "<span class=\"kpi-dot\" style=\"background:#ccc;\"></span>"
            st.markdown(f"""
            <div class="kpi-box" style="opacity:0.4;">
                <div style="display:flex;justify-content:space-between;">
                    <span class="kpi-label">{dot} {label}</span>
                    <span class="kpi-label">—</span>
                </div>
                <div class="kpi-label">{comment}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.caption(f"Model: {LLM_MODEL}")
    st.caption(f"Session: {SESSION_ID[:8]}...")


# ── OpenAI-compatible tool definitions ───────────────────────────────
def _build_openai_tools() -> list[dict]:
    from src.agent.tools import TOOL_DEFINITIONS
    return TOOL_DEFINITIONS


# ── LLM Agent (non-streaming, with Langfuse tracing) ────────────────
def call_llm_agent(
    user_message: str,
    trace: Any = None,
) -> dict[str, Any]:
    """
    Call the LLM via OpenAI-compatible endpoint with tool-calling loop.
    Each LLM call and tool invocation is tracked via Langfuse spans.
    """
    if not _OPENAI_AVAILABLE or not LLM_API_KEY or not LLM_ENDPOINT:
        return _fallback_agent(user_message)

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_ENDPOINT)
    tools = _build_openai_tools()

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    bundle_done = False  # guard: once bundle is calculated, stop tool-loop

    for turn in range(10):  # tighter limit
        t_start = time.time()

        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=2048,
            )
        except Exception as exc:
            return {"reply": f"❌ LLM 调用失败：{exc}", "bundle": None}

        latency_ms = (time.time() - t_start) * 1000
        choice = resp.choices[0]
        msg = choice.message

        usage = resp.usage
        tokens_in = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else 0

        # Langfuse trace recording
        if trace:
            span = trace.llm_span(
                name=f"qwen-plus-turn-{turn}",
                input={"messages_count": len(messages)},
                output={
                    "has_tool_calls": msg.tool_calls is not None,
                    "content": (msg.content or "")[:200],
                },
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=latency_ms,
            )
            span.end()

        if msg.tool_calls:
            # Guard: if bundle already exists, stop searching — force text
            if bundle_done:
                reply = msg.content or ""
                if not reply:
                    reply = "✅ 配置方案已生成，请查看下方一键下单链接。"
                return {"reply": reply, "bundle": _extract_bundle_ids_from_context(messages)}

            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                fn = TOOL_DISPATCHER.get(fn_name)
                if fn:
                    result = fn(**fn_args)

                    if fn_name == "calculate_bundle_price":
                        bundle_done = True

                    if trace:
                        tspan = trace.tool_span(
                            name=fn_name,
                            input=fn_args,
                            output=(
                                f"{len(result)} items" if isinstance(result, list)
                                else result.get("total_price", "ok") if isinstance(result, dict)
                                else str(result)[:100]
                            ),
                        )
                        tspan.end()

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False, indent=2),
                    })
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"error": f"Unknown tool: {fn_name}"}),
                    })
        else:
            reply = msg.content or ""
            bundle = _extract_bundle_ids_from_context(messages)
            if not bundle:
                bundle = _build_bundle_from_search_results(messages)
            return {"reply": reply, "bundle": bundle}

    # Max turns reached — try fallback before giving up
    bundle = _extract_bundle_ids_from_context(messages)
    if not bundle:
        bundle = _build_bundle_from_search_results(messages)
    return {
        "reply": "⚠️ Agent 超出最大推理轮次，请简化你的需求后重试。",
        "bundle": bundle,
    }


# ── Fallback: rule-based agent ──────────────────────────────────────
def _fallback_agent(user_message: str) -> dict[str, Any]:
    """Rule-based agent when LLM is not configured."""
    budget_match = re.search(r'预算[：:\s]*(\d+)', user_message)
    user_budget = float(budget_match.group(1)) if budget_match else 6000.0

    scenario_keywords = {
        "游戏": ["游戏", "打游戏", "电竞", "3A", "吃鸡", "LOL", "CS"],
        "视频剪辑": ["剪辑", "视频", "PR", "达芬奇", "剪映", "后期"],
        "3D渲染": ["渲染", "3D", "建模", "Blender", "C4D", "Maya"],
        "AI深度学习": ["AI", "深度学习", "训练", "推理", "大模型"],
    }
    detected = "游戏"
    for sc, kws in scenario_keywords.items():
        if any(kw in user_message for kw in kws):
            detected = sc
            break

    gpu_match = re.search(r'RTX\s*(\d{4})\s*(Ti|Super)?', user_message, re.IGNORECASE)
    cpu_match = re.search(r'i[579]-\d{4,5}\w*', user_message, re.IGNORECASE)

    bundle_ids: list[str] = []

    cpu_results = search_hardware(
        query=cpu_match.group(0) if cpu_match else "", category="CPU",
    )
    if not cpu_results:
        cpu_results = search_hardware(
            query="14600KF" if detected == "游戏" else "14700K", category="CPU",
        )
    if cpu_results:
        bundle_ids.append(cpu_results[0]["id"])

    gpu_query = gpu_match.group(0) if gpu_match else ""
    gpu_results = search_hardware(query=gpu_query, category="显卡")
    if not gpu_results:
        gpu_results = search_hardware(
            query="RTX 4060" if detected == "游戏" else "RTX 4070", category="显卡",
        )
    if gpu_results:
        bundle_ids.append(gpu_results[0]["id"])

    cpu_socket = cpu_results[0].get("socket", "LGA1700") if cpu_results else "LGA1700"
    mb_results = search_hardware(query=cpu_socket, category="主板")
    if mb_results:
        bundle_ids.append(mb_results[0]["id"])

    ram_results = search_hardware(
        query="DDR5 32GB" if detected != "视频剪辑" else "DDR5 64GB", category="内存",
    )
    if ram_results:
        bundle_ids.append(ram_results[0]["id"])

    ssd_results = search_hardware(
        query="2TB" if user_budget > 8000 else "1TB", category="固态硬盘",
    )
    if ssd_results:
        bundle_ids.append(ssd_results[0]["id"])

    psu_results = search_hardware(
        query="850W" if user_budget > 10000 else "750W", category="电源",
    )
    if psu_results:
        bundle_ids.append(psu_results[0]["id"])

    cooler_results = search_hardware(category="散热器")
    if cooler_results:
        bundle_ids.append(cooler_results[0]["id"])

    case_results = search_hardware(category="机箱")
    if case_results:
        bundle_ids.append(case_results[0]["id"])

    pricing = calculate_bundle_price(bundle_ids, user_budget)
    reply = _render_fallback_reply(pricing, user_budget, detected)
    return {"reply": reply, "bundle": pricing}


def _render_fallback_reply(pricing: dict, user_budget: float, scenario: str) -> str:
    lines = [
        "## 📊 预算分配分析",
        "",
        f"检测到使用场景 **{scenario}**，预算 **¥{user_budget:,.0f}**。",
        "",
        "## 🖥️ 推荐装机方案",
        "",
        "| 配件类型 | 型号 | 参考价格 |",
        "|----------|------|----------|",
    ]
    for item in pricing.get("items", []):
        lines.append(f"| {item['category']} | {item['name']} | ¥{item['price']:,} |")
    lines += [
        "",
        "## 💰 总价明细",
        f"- 配件合计：**¥{pricing['total_price']:,}**",
    ]
    if pricing.get("within_budget"):
        lines.append(f"- ✅ 在预算 ¥{user_budget:,} 以内")
    else:
        lines.append(f"- ⚠️ 超出 ¥{pricing.get('over_budget', 0):,}")
    lines += [
        "",
        "> ⚠️ 离线模式。配置 `.env` 后可启用 qwen-plus 智能推荐。",
    ]
    return "\n".join(lines)


# ── Helpers ─────────────────────────────────────────────────────────
def _extract_bundle_ids_from_context(messages: list[dict]) -> dict | None:
    """Try to find a calculate_bundle_price result first."""
    for msg in reversed(messages):
        if msg.get("role") == "tool":
            try:
                data = json.loads(msg["content"])
                if isinstance(data, dict) and "items" in data and "total_price" in data:
                    return data
            except (json.JSONDecodeError, TypeError):
                continue
    return None


def _build_bundle_from_search_results(messages: list[dict]) -> dict | None:
    """
    Fallback: auto-construct a bundle from search_hardware tool results.
    Picks the top result from each category searched by the LLM.
    """
    seen_categories: set[str] = set()
    picked_ids: list[str] = []

    for msg in messages:
        if msg.get("role") != "tool":
            continue
        try:
            data = json.loads(msg["content"])
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(data, list):
            continue
        for item in data:
            cat = item.get("category", "")
            hid = item.get("id", "")
            if cat and hid and cat not in seen_categories:
                seen_categories.add(cat)
                picked_ids.append(hid)
                break  # take top result per category from this search batch

    if len(picked_ids) >= 6:  # need at least 6 categories for a valid bundle
        bundle = calculate_bundle_price(picked_ids)
        return bundle if bundle.get("items") else None

    return None


# ── Guaranteed auto-bundle builder ──────────────────────────────────
def _auto_build_bundle(user_message: str, user_budget: float) -> dict | None:
    """
    Always build a complete 8-category bundle from the hardware database.
    Matches CPU socket → motherboard socket for compatibility.
    """
    ids: list[str] = []

    # ── CPU ──
    cpu_match = re.search(r'i[579]-\d{4,5}\w*', user_message, re.IGNORECASE)
    cpu_results = search_hardware(query=cpu_match.group(0) if cpu_match else "", category="CPU")
    if not cpu_results:
        cpu_results = search_hardware(category="CPU")
    cpu = cpu_results[0] if cpu_results else None
    cpu_socket = cpu.get("socket", "") if cpu else ""
    if cpu:
        ids.append(cpu["id"])

    # ── GPU ──
    gpu_picked = None
    gpu_match = re.search(r'RTX\s*(\d{4})\s*(Ti|Super)?', user_message, re.IGNORECASE)
    gpu_results = search_hardware(query=gpu_match.group(0) if gpu_match else "", category="显卡")
    if not gpu_results:
        gpu_results = search_hardware(category="显卡")
    for g in gpu_results:
        if gpu_picked is None or abs(g["price"] - user_budget * 0.38) < abs(gpu_picked["price"] - user_budget * 0.38):
            gpu_picked = g
    if gpu_picked:
        ids.append(gpu_picked["id"])

    # ── Motherboard (match CPU socket) ──
    mb_results = search_hardware(query=cpu_socket, category="主板")
    if not mb_results:
        mb_results = search_hardware(category="主板")
    if mb_results:
        ids.append(mb_results[0]["id"])

    # ── RAM ──
    ram_query = "64GB" if user_budget > 10000 else "32GB"
    ram_results = search_hardware(query=ram_query, category="内存")
    if not ram_results:
        ram_results = search_hardware(category="内存")
    if ram_results:
        ids.append(ram_results[0]["id"])

    # ── SSD ──
    ssd_query = "2TB" if user_budget > 8000 else "1TB"
    ssd_results = search_hardware(query=ssd_query, category="固态硬盘")
    if not ssd_results:
        ssd_results = search_hardware(category="固态硬盘")
    if ssd_results:
        ids.append(ssd_results[0]["id"])

    # ── PSU (calculate wattage from GPU+CPU TDP) ──
    total_tdp = (cpu.get("tdp", 125) if cpu else 125) + (gpu_picked.get("tdp", 200) if gpu_picked else 200)
    psu_watt_needed = int(total_tdp * 1.5 / 50) * 50  # round to nearest 50W with 50% margin
    psu_results = sorted(search_hardware(category="电源"), key=lambda x: x["wattage"])
    psu_picked = None
    for p in psu_results:
        if p["wattage"] >= psu_watt_needed:
            psu_picked = p
            break
    if not psu_picked and psu_results:
        psu_picked = psu_results[-1]
    if psu_picked:
        ids.append(psu_picked["id"])

    # ── Cooler ──
    cpu_tdp = cpu.get("tdp", 125) if cpu else 125
    cooler_results = sorted(search_hardware(category="散热器"), key=lambda x: x.get("tdp_support", 999))
    cooler_picked = None
    for c in cooler_results:
        if c.get("tdp_support", 0) >= cpu_tdp:
            cooler_picked = c
            break
    if not cooler_picked and cooler_results:
        cooler_picked = cooler_results[-1]
    if cooler_picked:
        ids.append(cooler_picked["id"])

    # ── Case ──
    case_results = sorted(search_hardware(category="机箱"), key=lambda x: x["price"])
    if case_results:
        # Budget-appropriate case
        case_budget = user_budget * 0.05
        picked = case_results[0]
        for c in case_results:
            if abs(c["price"] - case_budget) < abs(picked["price"] - case_budget):
                picked = c
        ids.append(picked["id"])

    if len(ids) >= 6:
        return calculate_bundle_price(ids, user_budget)
    return None


# ── Evaluation score pusher ─────────────────────────────────────────
def _push_eval_scores(bundle: dict, user_budget: float, session_id: str) -> None:
    """Compute local evaluation scores and push them to Langfuse."""
    if not bundle or not bundle.get("items"):
        return

    categories = [it["category"] for it in bundle["items"]]
    total = bundle.get("total_price", 0)

    local = evaluate_recommendation(
        total_price=total,
        user_budget=user_budget,
        recommended_categories=categories,
    )

    client = _get_lf()
    if not client:
        return

    # Push scores to Langfuse
    for name, value in {
        "预算约束合规率": local["budget_compliance"]["score"],
        "意图对齐度": local["overall_score"] * 5.0,
        "回执率": local["recall"]["score"],
        "精确度": local["precision"]["score"],
        "F1": local["f1"]["score"],
    }.items():
        try:
            client.create_score(name=name, value=float(value), session_id=session_id)
        except Exception:
            pass

    # Store in session state
    st.session_state.eval_scores = {
        "fetched": True,
        "source": "local+langfuse",
        "raw_count": 5,
        "scores": {
            "预算约束合规率": {"value": local["budget_compliance"]["score"], "max": 1.0,
                "ratio": local["budget_compliance"]["score"],
                "comment": "符合预算" if local["budget_compliance"]["compliant"] else "超出预算"},
            "意图对齐度": {"value": round(local["overall_score"] * 5, 1), "max": 5.0,
                "ratio": local["overall_score"],
                "comment": f"综合 {local['overall_score']:.0%}"},
            "回执率": {"value": local["recall"]["score"], "max": 1.0,
                "ratio": local["recall"]["score"],
                "comment": f"覆盖 {local['recall']['covered']}/8 品类"},
            "精确度": {"value": local["precision"]["score"], "max": 1.0,
                "ratio": local["precision"]["score"],
                "comment": "偏好命中率"},
            "F1": {"value": local["f1"]["score"], "max": 1.0,
                "ratio": local["f1"]["score"],
                "comment": "Precision × Recall 调和"},
        },
    }
    clear_eval_cache(session_id)


# ── Chat History ────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# ── User Input ──────────────────────────────────────────────────────
placeholder_text = "e.g. Budget 6000 RMB, RTX 4060 Ti, mainly for gaming..."

if prompt := st.chat_input(placeholder_text):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    extra = extra_notes if extra_notes else "无特殊要求"

    # Step 1: Auto-build bundle from database FIRST (guaranteed)
    bundle = _auto_build_bundle(prompt, budget)

    # Step 2: Build prompt with bundle context injected
    bundle_text = ""
    if bundle and bundle.get("items"):
        bundle_text = "\n\n## 数据库匹配的配件（必须基于此推荐）\n"
        for it in bundle["items"]:
            bundle_text += f"- {it['category']}: {it['name']} (¥{it['price']:,})\n"
        bundle_text += f"\n总价: ¥{bundle['total_price']:,}"

    structured_prompt = USER_PROMPT_TEMPLATE.format(
        budget=f"{budget:,}" if budget else "未指定",
        usage_scenario=scenario if scenario else "未指定",
        preference=preference if preference else "无",
        extra_notes=extra + bundle_text,
    )

    with st.chat_message("assistant"):
        with trace_agent_interaction(
            session_id=SESSION_ID,
            user_input=structured_prompt,
            metadata={"budget": budget, "scenario": scenario},
        ) as trace:
            result = call_llm_agent(structured_prompt, trace=trace)
            reply = result["reply"]

            if bundle and bundle.get("items"):
                st.session_state.last_bundle = bundle

                # Push evaluation scores to Langfuse (for real-time dashboard)
                _push_eval_scores(bundle, budget, SESSION_ID)

                # Step 3: Generate Stripe checkout link
                checkout_btn = ""
                with st.spinner("Generating payment link..."):
                    try:
                        checkout = create_checkout_session(
                            hardware_items=bundle["items"],
                            metadata={"source": "commerce-agent"},
                        )
                        if checkout.get("success"):
                            st.session_state.checkout_count += 1
                            st.session_state.last_checkout_url = checkout["checkout_url"]
                            record_checkout(SESSION_ID)
                            checkout_btn = (
                                f'\n\n---\n\n**Checkout**\n\n'
                                f'<a href="{checkout["checkout_url"]}" target="_blank" '
                                f'style="display:inline-block;padding:12px 24px;'
                                f'background:#5b6af0;color:white;border-radius:6px;'
                                f'text-decoration:none;font-weight:600;font-size:15px;">'
                                f'Pay with Stripe (Test)</a>\n\n'
                                f'Test card: `4242 4242 4242 4242` · any future date · any CVC'
                            )
                        else:
                            checkout_btn = f"\n\nPayment error: {checkout.get('error')}"
                    except Exception as exc:
                        checkout_btn = f"\n\nStripe error: {exc}"

                full_reply = reply + checkout_btn
                st.markdown(full_reply)
                reply = full_reply
            else:
                st.markdown(reply)

            st.session_state.messages.append({
                "role": "assistant",
                "content": reply,
                "bundle": bundle,
            })

    # Rerun to refresh sidebar KPIs + eval dashboard
    st.rerun()

# ── Footer ──────────────────────────────────────────────────────────
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_bundle = None
        st.session_state.last_checkout_url = None
        st.session_state.eval_scores = None
        st.rerun()
with col2:
    if st.button("View last bundle", use_container_width=True):
        if st.session_state.last_bundle:
            with st.expander("Bundle details", expanded=True):
                st.json(st.session_state.last_bundle)
        else:
            st.info("No bundle yet.")
with col3:
    st.caption("Commerce Agent · v0.5.0")
