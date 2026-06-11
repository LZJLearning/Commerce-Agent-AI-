"""
Application configuration via python-dotenv.

Loads environment variables from a .env file and exposes them
as module-level constants.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM (Alibaba Cloud Model Studio — OpenAI-compatible) ──────────
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-plus")

# ── Langfuse Observability ────────────────────────────────────────
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# ── Stripe Payment ────────────────────────────────────────────────
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")

# ── Validation (warn on missing keys, don't crash) ────────────────
_MANDATORY = {
    "LLM_API_KEY": LLM_API_KEY,
    "LLM_ENDPOINT": LLM_ENDPOINT,
    "LANGFUSE_PUBLIC_KEY": LANGFUSE_PUBLIC_KEY,
    "LANGFUSE_SECRET_KEY": LANGFUSE_SECRET_KEY,
    "STRIPE_SECRET_KEY": STRIPE_SECRET_KEY,
}

for _name, _val in _MANDATORY.items():
    if not _val:
        print(f"[config] WARNING: {_name} is not set. Some features may not work.")
