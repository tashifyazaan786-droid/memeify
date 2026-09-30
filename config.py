"""
Central configuration for the Meme Knowledge Agent.

Loads variables from a local `.env` file so the API key never needs to be
hardcoded or exported manually every session.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "memes.md"
PERSIST_DIR = ROOT_DIR / "chroma_db"
ENV_PATH = ROOT_DIR / ".env"

load_dotenv(ENV_PATH, override=False)


def _require_env(name: str, hint: str = "") -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        msg = (
            f"{name} is not set.\n"
            f"  1. Copy .env.example -> .env\n"
            f"  2. Put your key in .env:  {name}=...\n"
            f"  3. Restart the app (python app.py)\n"
        )
        if hint:
            msg += f"\n{hint}"
        raise RuntimeError(msg)
    return value


def get_groq_api_key() -> str:
    return _require_env(
        "GROQ_API_KEY",
        hint="Get a free key at https://console.groq.com/keys",
    )


# ---------------------------------------------------------------------------
# Model / embeddings
# ---------------------------------------------------------------------------
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b").strip()
# Second, cheaper/faster model used for the structured-extraction step where
# we just need reliable JSON, not creative writing.
GROQ_EXTRACTION_MODEL = os.environ.get(
    "GROQ_EXTRACTION_MODEL", "openai/gpt-oss-20b"
).strip()
# Optional multimodal model used for meme-image understanding.
GROQ_VISION_MODEL = os.environ.get(
    "GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"
).strip()

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.6"))

# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------
RETRIEVER_K = 4
# If the top retrieved chunk's relevance score is below this, we treat the
# knowledge base as "no strong match" and trigger live research instead.
KB_MATCH_SCORE_THRESHOLD = 0.55

# ---------------------------------------------------------------------------
# Verification / confidence thresholds
# ---------------------------------------------------------------------------
# HIGH  -> auto-saved to the permanent knowledge base, no human step needed
# MEDIUM/LOW -> surfaced to the user as a draft, saved only on approval
MIN_SOURCES_FOR_HIGH = 2  # distinct corroborating domains/sources required

# ---------------------------------------------------------------------------
# Reddit / trends
# ---------------------------------------------------------------------------
SUBREDDITS = ["memes", "IndianDankMemes", "dankmemes", "memetemplatesofficial"]
REDDIT_TIMEOUT = 6
TREND_LIMIT_DEFAULT = 10

FALLBACK_TRENDS = [
    "r/memes (cached): Gucci Morty / AI drip edits still circulating",
    "r/memes (cached): 'locked in' gym & study captions",
    "r/IndianDankMemes (cached): UPI / payment-app outage panic memes",
    "r/IndianDankMemes (cached): Ravi Kishan / jaldi-the-late style clips",
    "r/dankmemes (cached): Gen Alpha brainrot + Italian brainrot animals",
    "r/memes (cached): IShowSpeed reaction remixes",
    "r/dankmemes (cached): 'bro thought he cooked' / mid takes",
]

# ---------------------------------------------------------------------------
# Web research
# ---------------------------------------------------------------------------
WEB_SEARCH_MAX_RESULTS = 5

# ---------------------------------------------------------------------------
# UI languages
# ---------------------------------------------------------------------------
LANGUAGE_INSTRUCTIONS = {
    "English": "Respond in clear, natural English.",
    "Hinglish": (
        "Respond in casual Hinglish (Hindi + English mix, Latin script). "
        "Keep tone fun and conversational."
    ),
    "Hindi": "Respond in pure Hindi using Devanagari script.",
}

SHARE = os.environ.get("SHARE", "1").strip() != "0"
