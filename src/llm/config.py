"""
LLM Configuration
─────────────────
Configures IBM WatsonX AI LLM via LangChain, following the same
pattern used in the IBM_RAG project (C:\\Workspace\\Coursera\\IBM_RAG).

Uses ChatWatsonx (chat model) with the modern langchain agent API.

Environment variables (loaded from .env):
  WATSONX_APIKEY       – IBM Cloud API key
  WATSONX_PROJECT_ID   – WatsonX project GUID
  WATSONX_URL          – WatsonX endpoint (default: us-south)
  WATSONX_MODEL_ID     – Chat model to use (default: meta-llama/llama-3-3-70b-instruct)
  LLM_ENABLED          – "true" to enable LLM, "false" for rule-based only
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env from project root
load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────

WATSONX_APIKEY: str = os.getenv("WATSONX_APIKEY", "")
WATSONX_PROJECT_ID: str = os.getenv("WATSONX_PROJECT_ID", "")
WATSONX_URL: str = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_MODEL_ID: str = os.getenv("WATSONX_MODEL_ID", "meta-llama/llama-3-3-70b-instruct")
LLM_ENABLED: bool = os.getenv("LLM_ENABLED", "true").lower() in ("true", "1", "yes")


def is_llm_available() -> bool:
    """Check if LLM is enabled and credentials are configured."""
    if not LLM_ENABLED:
        return False
    if not WATSONX_APIKEY or not WATSONX_PROJECT_ID:
        logger.warning("LLM enabled but WATSONX credentials not set in .env")
        return False
    return True


_llm_instance = None


def get_llm():
    """
    Get a configured ChatWatsonx instance (singleton).

    Returns None if LLM is not available.
    Uses langchain_ibm.ChatWatsonx following the IBM_RAG project pattern,
    adapted for langchain 1.2+ which requires chat models for agents.
    """
    global _llm_instance

    if not is_llm_available():
        return None

    if _llm_instance is not None:
        return _llm_instance

    try:
        from langchain_ibm import ChatWatsonx

        _llm_instance = ChatWatsonx(
            model_id=WATSONX_MODEL_ID,
            url=WATSONX_URL,
            project_id=WATSONX_PROJECT_ID,
            apikey=WATSONX_APIKEY,
            params={
                "max_tokens": 2048,
                "temperature": 0.1,
            },
        )
        logger.info("ChatWatsonx LLM initialized: model=%s", WATSONX_MODEL_ID)
        return _llm_instance

    except ImportError:
        logger.error(
            "langchain_ibm not installed. Run: pip install langchain-ibm ibm_watsonx_ai"
        )
        return None
    except Exception as e:
        logger.error("Failed to initialize ChatWatsonx LLM: %s", e)
        return None


def reset_llm():
    """Reset the cached LLM instance (useful for testing or credential changes)."""
    global _llm_instance
    _llm_instance = None
