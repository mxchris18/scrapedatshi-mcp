"""
scrapedatshi_mcp.server
~~~~~~~~~~~~~~~~~~~~~~~
MCP server exposing scrapedatshi pipeline tools to Claude Desktop and any
MCP-compatible AI client.

Tools exposed:
    verify_provider_key      — Verify an LLM or embedding API key + get live model list
    get_usage_guide          — Returns the guided wizard flow Claude should follow
    scrape_url               — Scrape a URL and return clean Markdown (no chunking)
    chunk_url                — Scrape & chunk a single URL into RAG-ready text segments
    chunk_file               — Upload a local file (PDF/MD/TXT/CSV/XLSX/DOCX/IPYNB/code/etc), chunk it, return JSON
    crawl_site               — Crawl a whole site (sitemap or spider mode)
    extract_data             — Extract structured schema from a URL using your LLM
    extract_crawl            — Multi-page schema extraction via site crawl
    sync_to_vectordb         — Full pipeline: scrape URL → embed → inject into vector DB
    ingest_file              — Full pipeline: upload local file → embed → inject into vector DB
    ingest_scraped           — Full pipeline: bulk-ingest a folder of pre-scraped files → embed → inject into vector DB
    autorag                  — Full pipeline: crawl site → chunk → embed → inject into vector DB
    inspect_vectordb         — Read vector DB metadata: dimension, vector count, suggested models (free)
    query_vectordb           — Semantic search: embed a query and retrieve top-N chunks
    rag_chat                 — RAG Chat: retrieve chunks + generate a grounded LLM answer
    list_embedding_providers — Discover supported embedding providers + required fields
    list_vector_db_providers — Discover supported vector DBs + required fields

Guided Wizard Flow:
    For any operation requiring an LLM or embedding key, Claude MUST:
      1. Call verify_provider_key → get live model list for the user's key
      2. Present models to user, ask them to choose
      3. Ask if JS rendering is needed (SPA/JS-heavy page)
      4. Present Contextual Retrieval as a recommended upgrade (improves accuracy 35-50%)
      5. Call the actual tool with all confirmed parameters

Key Fallback Pattern (secure BYOK):
    Sensitive API keys are resolved in this priority order:
      1. Argument passed directly in the tool call (explicit override)
      2. Environment variable set in the MCP config (preferred secure path)
      3. Clear error message if neither is found

    Supported environment variables:
        SCRAPEDATSHI_API_KEY   — Your scrapedatshi API key (required)
        OPENAI_API_KEY         — OpenAI key (LLM + embedding)
        ANTHROPIC_API_KEY      — Anthropic key (LLM)
        GEMINI_API_KEY         — Google Gemini key (LLM + embedding)
        COHERE_API_KEY         — Cohere key (embedding)
        MISTRAL_API_KEY        — Mistral key (embedding)
        VOYAGE_API_KEY         — Voyage AI key (embedding)
        PINECONE_API_KEY       — Pinecone vector DB key
        QDRANT_API_KEY         — Qdrant vector DB key (optional)
        WEAVIATE_API_KEY       — Weaviate vector DB key (optional)

Run as stdio MCP server (standard for Claude Desktop):
    python -m scrapedatshi_mcp.server
    # or after pip install:
    scrapedatshi-mcp
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server

from scrapedatshi import ScrapedatshiClient
from scrapedatshi.exceptions import (
    AuthError,
    InsufficientCreditsError,
    RateLimitError,
    ScrapedatshiError,
    ServerBusyError,
    ValidationError,
)
from scrapedatshi.providers import (
    EMBEDDING_PROVIDERS,
    LLM_PROVIDERS,
    VECTOR_DB_PROVIDERS,
)

# ── Server instance ───────────────────────────────────────────────────────────

server = Server("scrapedatshi")

# ── Provider model discovery ──────────────────────────────────────────────────


async def _discover_openai_llm_models(api_key: str) -> dict:
    """Discover available OpenAI text generation models."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        return {
            "valid": False,
            "models": [],
            "error": "openai package not installed. Run: pip install scrapedatshi-mcp[openai]",
        }
    try:
        client = AsyncOpenAI(api_key=api_key)
        response = await client.models.list()
        llm_prefixes = ("gpt-", "o1-", "o3-", "o4-", "chatgpt-")
        models = sorted(
            [
                m.id
                for m in response.data
                if any(m.id.startswith(p) for p in llm_prefixes)
            ],
            reverse=True,
        )
        if not models:
            return {
                "valid": False,
                "models": [],
                "error": (
                    "Key is valid but no text generation models were found for this account. "
                    "Check your OpenAI account permissions — your key may be restricted to "
                    "specific model families. Contact OpenAI support if unexpected."
                ),
            }
        return {"valid": True, "models": models, "error": None}
    except Exception as e:
        msg = str(e)
        if "401" in msg or "authentication" in msg.lower() or "api_key" in msg.lower():
            return {
                "valid": False,
                "models": [],
                "error": "Invalid API key. Check your OpenAI key.",
            }
        return {"valid": False, "models": [], "error": f"OpenAI error: {msg[:200]}"}


async def _discover_anthropic_llm_models(api_key: str) -> dict:
    """Discover available Anthropic models via live API."""
    try:
        import anthropic
    except ImportError:
        return {
            "valid": False,
            "models": [],
            "error": "anthropic package not installed. Run: pip install scrapedatshi-mcp[anthropic]",
        }
    try:
        client = anthropic.AsyncAnthropic(api_key=api_key)
        response = await client.models.list()
        models = [m.id for m in response.data if m.id]
        if not models:
            return {
                "valid": False,
                "models": [],
                "error": "No models found for this API key.",
            }
        return {"valid": True, "models": models, "error": None}
    except Exception as e:
        msg = str(e)
        if "401" in msg or "authentication" in msg.lower() or "api_key" in msg.lower():
            return {
                "valid": False,
                "models": [],
                "error": "Invalid API key. Check your Anthropic key.",
            }
        return {"valid": False, "models": [], "error": f"Anthropic error: {msg[:200]}"}


async def _discover_gemini_llm_models(api_key: str) -> dict:
    """Discover available Gemini text generation models."""
    try:
        from google import genai
    except ImportError:
        return {
            "valid": False,
            "models": [],
            "error": "google-genai package not installed. Run: pip install scrapedatshi-mcp[gemini]",
        }
    try:
        client = genai.Client(api_key=api_key)

        def _list():
            return list(client.models.list())

        all_models = await asyncio.to_thread(_list)
        EXCLUDE = (
            "embedding",
            "imagen",
            "veo",
            "lyria",
            "tts",
            "audio",
            "aqa",
            "robotics",
            "translate",
            "live",
        )
        llm_models = sorted(
            set(
                m.name
                for m in all_models
                if not any(kw in m.name.lower() for kw in EXCLUDE)
                and ("gemini" in m.name.lower() or "gemma" in m.name.lower())
            ),
            reverse=True,
        )
        if not llm_models:
            return {
                "valid": False,
                "models": [],
                "error": "No Gemini text generation models found.",
            }
        return {"valid": True, "models": llm_models, "error": None}
    except Exception as e:
        msg = str(e)
        if "401" in msg or "403" in msg or "api_key" in msg.lower():
            return {
                "valid": False,
                "models": [],
                "error": "Invalid API key. Check your Google Gemini key.",
            }
        return {"valid": False, "models": [], "error": f"Gemini error: {msg[:200]}"}


async def _discover_openai_embed_models(api_key: str) -> dict:
    """Discover available OpenAI embedding models."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        return {
            "valid": False,
            "models": [],
            "error": "openai package not installed. Run: pip install scrapedatshi-mcp[openai]",
        }
    try:
        client = AsyncOpenAI(api_key=api_key)
        response = await client.models.list()
        models = sorted(
            [m.id for m in response.data if "embedding" in m.id.lower()], reverse=True
        )
        if not models:
            return {
                "valid": False,
                "models": [],
                "error": (
                    "Key is valid but no embedding models were found for this account. "
                    "Check your OpenAI account permissions — your key may be restricted. "
                    "Contact OpenAI support if unexpected."
                ),
            }
        return {"valid": True, "models": models, "error": None}
    except Exception as e:
        msg = str(e)
        if "401" in msg or "authentication" in msg.lower():
            return {
                "valid": False,
                "models": [],
                "error": "Invalid API key. Check your OpenAI key.",
            }
        return {"valid": False, "models": [], "error": f"OpenAI error: {msg[:200]}"}


async def _discover_cohere_embed_models(api_key: str) -> dict:
    """Discover available Cohere embedding models."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://api.cohere.com/v1/models",
                params={"endpoint": "embed"},
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Accept": "application/json",
                },
            )
        if resp.status_code == 401:
            return {
                "valid": False,
                "models": [],
                "error": "Invalid API key. Check your Cohere key.",
            }
        if resp.status_code != 200:
            return {
                "valid": False,
                "models": [],
                "error": f"Cohere API error: {resp.status_code}",
            }
        data = resp.json()
        models = [m["name"] for m in data.get("models", []) if m.get("name")]
        if not models:
            return {
                "valid": False,
                "models": [],
                "error": (
                    "Key is valid but no embedding models were found for this account. "
                    "Check your Cohere account permissions or contact Cohere support."
                ),
            }
        return {"valid": True, "models": models, "error": None}
    except Exception as e:
        return {"valid": False, "models": [], "error": f"Cohere error: {str(e)[:200]}"}


async def _discover_gemini_embed_models(api_key: str) -> dict:
    """Discover available Gemini embedding models."""
    try:
        from google import genai
    except ImportError:
        return {
            "valid": False,
            "models": [],
            "error": "google-genai package not installed. Run: pip install scrapedatshi-mcp[gemini]",
        }
    try:
        client = genai.Client(api_key=api_key)

        def _list():
            return list(client.models.list())

        all_models = await asyncio.to_thread(_list)
        models = sorted(
            [m.name for m in all_models if "embedding" in m.name.lower()], reverse=True
        )
        if not models:
            return {
                "valid": False,
                "models": [],
                "error": "No Gemini embedding models found.",
            }
        return {"valid": True, "models": models, "error": None}
    except Exception as e:
        msg = str(e)
        if "401" in msg or "403" in msg:
            return {
                "valid": False,
                "models": [],
                "error": "Invalid API key. Check your Google Gemini key.",
            }
        return {"valid": False, "models": [], "error": f"Gemini error: {msg[:200]}"}


async def _discover_mistral_embed_models(api_key: str) -> dict:
    """Discover available Mistral embedding models."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://api.mistral.ai/v1/models",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Accept": "application/json",
                },
            )
        if resp.status_code == 401:
            return {
                "valid": False,
                "models": [],
                "error": "Invalid API key. Check your Mistral key.",
            }
        if resp.status_code != 200:
            return {
                "valid": False,
                "models": [],
                "error": f"Mistral API error: {resp.status_code}",
            }
        data = resp.json()
        models = [
            m["id"] for m in data.get("data", []) if "embed" in m.get("id", "").lower()
        ]
        if not models:
            return {
                "valid": False,
                "models": [],
                "error": (
                    "Key is valid but no embedding models were found for this account. "
                    "Check your Mistral account permissions or contact Mistral support."
                ),
            }
        return {"valid": True, "models": models, "error": None}
    except Exception as e:
        return {"valid": False, "models": [], "error": f"Mistral error: {str(e)[:200]}"}


async def _discover_voyage_embed_models(api_key: str) -> dict:
    """Verify Voyage AI key and return current model catalog."""
    VOYAGE_MODELS = [
        "voyage-3",
        "voyage-3-lite",
        "voyage-code-3",
        "voyage-finance-2",
        "voyage-law-2",
    ]
    try:
        import voyageai
    except ImportError:
        return {
            "valid": False,
            "models": [],
            "error": "voyageai package not installed. Run: pip install scrapedatshi-mcp[voyage]",
        }
    try:

        def _test():
            client = voyageai.Client(api_key=api_key)
            client.embed(["test"], model="voyage-3", input_type="document")

        await asyncio.to_thread(_test)
        return {"valid": True, "models": VOYAGE_MODELS, "error": None}
    except Exception as e:
        msg = str(e)
        if "401" in msg or "authentication" in msg.lower() or "api_key" in msg.lower():
            return {
                "valid": False,
                "models": [],
                "error": "Invalid API key. Check your Voyage AI key.",
            }
        return {"valid": False, "models": [], "error": f"Voyage AI error: {msg[:200]}"}


async def _run_discovery(provider: str, provider_type: str, api_key: str) -> dict:
    """Dispatch to the correct discovery function."""
    dispatch = {
        ("openai", "llm"): _discover_openai_llm_models,
        ("anthropic", "llm"): _discover_anthropic_llm_models,
        ("gemini", "llm"): _discover_gemini_llm_models,
        ("openai", "embedding"): _discover_openai_embed_models,
        ("cohere", "embedding"): _discover_cohere_embed_models,
        ("gemini", "embedding"): _discover_gemini_embed_models,
        ("mistral", "embedding"): _discover_mistral_embed_models,
        ("voyage", "embedding"): _discover_voyage_embed_models,
    }
    fn = dispatch.get((provider.lower(), provider_type.lower()))
    if fn is None:
        return {
            "valid": False,
            "models": [],
            "error": f"Unsupported combination: provider='{provider}', type='{provider_type}'. "
            f"LLM providers: openai, anthropic, gemini. "
            f"Embedding providers: openai, cohere, gemini, mistral, voyage.",
        }
    return await fn(api_key)


# ── Key resolution helpers ────────────────────────────────────────────────────


def _resolve_scrapedatshi_key() -> str | None:
    return os.environ.get("SCRAPEDATSHI_API_KEY")


def _resolve_llm_key(arguments: dict, provider: str | None = None) -> str | None:
    explicit = arguments.get("llm_api_key")
    if explicit:
        return explicit
    provider_env_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    if provider and provider in provider_env_map:
        val = os.environ.get(provider_env_map[provider])
        if val:
            return val
    for env_var in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]:
        val = os.environ.get(env_var)
        if val:
            return val
    return None


def _resolve_embedding_key(arguments: dict, provider: str | None = None) -> str | None:
    if provider == "ollama":
        return arguments.get("embedding_api_key", "")
    explicit = arguments.get("embedding_api_key")
    if explicit:
        return explicit
    provider_env_map = {
        "openai": "OPENAI_API_KEY",
        "cohere": "COHERE_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "voyage": "VOYAGE_API_KEY",
    }
    if provider and provider in provider_env_map:
        val = os.environ.get(provider_env_map[provider])
        if val:
            return val
    for env_var in ["OPENAI_API_KEY", "COHERE_API_KEY", "GEMINI_API_KEY"]:
        val = os.environ.get(env_var)
        if val:
            return val
    return None


def _resolve_vector_db_config(arguments: dict, vector_db: str) -> dict:
    config: dict = {}
    raw_config = arguments.get("vector_db_config", {})
    if isinstance(raw_config, str):
        try:
            config = json.loads(raw_config)
        except json.JSONDecodeError:
            config = {}
    elif isinstance(raw_config, dict):
        config = dict(raw_config)

    if vector_db == "pinecone" and not config.get("api_key"):
        env_key = os.environ.get("PINECONE_API_KEY")
        if env_key:
            config["api_key"] = env_key
    elif vector_db == "qdrant" and not config.get("api_key"):
        env_key = os.environ.get("QDRANT_API_KEY")
        if env_key:
            config["api_key"] = env_key
    elif vector_db == "weaviate" and not config.get("api_key"):
        env_key = os.environ.get("WEAVIATE_API_KEY")
        if env_key:
            config["api_key"] = env_key
    return config


def _get_client() -> ScrapedatshiClient:
    api_key = _resolve_scrapedatshi_key()
    if not api_key:
        raise AuthError(
            "No scrapedatshi API key found. Set SCRAPEDATSHI_API_KEY in your MCP "
            "environment config or pass it explicitly."
        )
    # Fetch mode: "local" (default) = MCP server fetches URLs using its own IP.
    # Set SCRAPEDATSHI_FETCH_MODE=server in your MCP env config to use our server's
    # IP instead (billed at 2× the standard per-URL rate).
    fetch_mode = os.environ.get("SCRAPEDATSHI_FETCH_MODE", "local")
    return ScrapedatshiClient(api_key=api_key, fetch_mode=fetch_mode)


def _format_error(exc: Exception) -> str:
    if isinstance(exc, InsufficientCreditsError):
        return (
            f"❌ Insufficient credits: {exc}\n"
            "Top up your balance at https://scrapedatshi.com/portal/billing"
        )
    if isinstance(exc, AuthError):
        return (
            f"❌ Authentication error: {exc}\n"
            "Check your SCRAPEDATSHI_API_KEY in the MCP config."
        )
    if isinstance(exc, ValidationError):
        return f"❌ Validation error: {exc}\nCheck your request parameters."
    if isinstance(exc, RateLimitError):
        return (
            f"❌ Rate limit reached: {exc}\n"
            "Do NOT retry automatically. Inform the user and wait for their instruction before trying again."
        )
    if isinstance(exc, ServerBusyError):
        retry = getattr(exc, "retry_after", None)
        wait_msg = f" Suggested wait: {retry} seconds." if retry else ""
        return (
            f"❌ Server temporarily at capacity: {exc}.{wait_msg}\n"
            "Inform the user and ask if they'd like to retry after waiting."
        )
    if isinstance(exc, ScrapedatshiError):
        return f"❌ scrapedatshi API error: {exc}"
    return f"❌ Unexpected error: {exc}"


# ── Tool definitions ──────────────────────────────────────────────────────────

_PREFLIGHT_NOTE = (
    "\n\n📋 PRE-FLIGHT REQUIRED — before calling this tool:\n"
    "1. Call verify_provider_key with the user's provider + key → get live model list\n"
    "2. Present the model list and ask the user to choose one\n"
    "3. Ask: 'Is this a JavaScript-heavy page or single-page app (SPA)?' → sets js_render\n"
    "4. Present Contextual Retrieval as a recommended upgrade: 'Would you like to enable "
    "Contextual Retrieval (RAG 2.0)? It uses your LLM to enrich each chunk with context, "
    "improving retrieval accuracy by 35–50%. Recommended for production RAG pipelines. "
    "Costs ~$0.001/chunk extra.' → sets contextual_retrieval\n"
    "5. Then call this tool with all confirmed parameters."
)


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        # ── verify_provider_key ───────────────────────────────────────────────
        types.Tool(
            name="verify_provider_key",
            description=(
                "Verify an LLM or embedding API key and return the live list of models "
                "available for that key. Call this BEFORE any operation that requires an "
                "LLM or embedding provider — never assume or hardcode model names.\n\n"
                "Returns: key validity, list of available model names (live from the provider's API), "
                "and an error message if the key is invalid.\n\n"
                "Supported LLM providers: openai, anthropic, gemini\n"
                "Supported embedding providers: openai, cohere, gemini, mistral, voyage\n\n"
                "The API key can be omitted if the corresponding env var is set "
                "(OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, COHERE_API_KEY, "
                "MISTRAL_API_KEY, VOYAGE_API_KEY)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": "Provider to verify. LLM: 'openai', 'anthropic', 'gemini'. Embedding: 'openai', 'cohere', 'gemini', 'mistral', 'voyage'.",
                        "enum": [
                            "openai",
                            "anthropic",
                            "gemini",
                            "cohere",
                            "mistral",
                            "voyage",
                        ],
                    },
                    "provider_type": {
                        "type": "string",
                        "description": "'llm' for text generation models (used in extract_data, extract_crawl, contextual_retrieval). 'embedding' for vector embedding models (used in sync_to_vectordb).",
                        "enum": ["llm", "embedding"],
                    },
                    "api_key": {
                        "type": "string",
                        "description": "The API key to verify. Can be omitted if the corresponding env var is set.",
                    },
                },
                "required": ["provider", "provider_type"],
            },
        ),
        # ── get_usage_guide ───────────────────────────────────────────────────
        types.Tool(
            name="get_usage_guide",
            description=(
                "Returns the complete guided workflow for using scrapedatshi tools. "
                "Call this at the start of any scrapedatshi conversation to understand "
                "which tool to use for each task and the required pre-flight sequence."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        # ── scrape_url ────────────────────────────────────────────────────────
        types.Tool(
            name="scrape_url",
            description=(
                "Scrape a single web URL and return the full page content as clean Markdown. "
                "No chunking — returns the raw text in one piece. No embedding or vector DB required.\n\n"
                "Use this when the user wants to read, summarize, or display a page's content "
                "directly — without splitting it into chunks. For RAG/embedding use cases where "
                "you need the content split into segments, use chunk_url instead.\n\n"
                "The response includes selectors_found — a list of CSS selectors for the main "
                "content sections detected on the page (e.g. ['article', '#main-content']). "
                "Use one to re-scrape just that section via the selector= parameter.\n\n"
                "Supports authenticated scraping via cookies, headers, and Playwright storage state."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The web URL to scrape.",
                    },
                    "selector": {
                        "type": "string",
                        "description": "Optional CSS selector to target a specific element (e.g. 'article', '.content', 'main').",
                    },
                    "js_render": {
                        "type": "boolean",
                        "description": "Use headless Chromium to render JavaScript before scraping. Required for SPAs and JS-heavy pages. Ask the user before enabling. Adds a small surcharge.",
                        "default": False,
                    },
                    "cookies": {
                        "type": "object",
                        "description": (
                            "Optional dict of cookies to include with the request for authenticated scraping "
                            '(e.g. {"session": "abc123", "auth_token": "xyz"}). '
                            "Only used in local-fetch mode (default). Never forwarded to the server. "
                            "Use this for pages behind a login wall — pass your browser session cookie."
                        ),
                        "additionalProperties": {"type": "string"},
                    },
                    "headers": {
                        "type": "object",
                        "description": (
                            "Optional dict of additional HTTP request headers "
                            '(e.g. {"Authorization": "Bearer eyJ..."}). '
                            "Only used in local-fetch mode (default). Never forwarded to the server."
                        ),
                        "additionalProperties": {"type": "string"},
                    },
                    "storage_state": {
                        "type": "object",
                        "description": (
                            "Optional Playwright storage state dict for enterprise SSO/MFA authenticated scraping. "
                            "Captured locally using capture_session() from scrapedatshi[auth] — "
                            "opens a real browser, user logs in manually, session is captured. "
                            "Contains cookies and localStorage tokens. "
                            "Use this for Okta, Duo, or any SSO flow that blocks automated login. "
                            "Load from a saved session.auth.json file."
                        ),
                        "additionalProperties": True,
                    },
                },
                "required": ["url"],
            },
        ),
        # ── chunk_url ─────────────────────────────────────────────────────────
        types.Tool(
            name="chunk_url",
            description=(
                "Scrape a single web URL, chunk its content into RAG-ready text segments, "
                "and return the structured chunks. No embedding or vector DB required — "
                "this is the fastest and cheapest operation for RAG pipelines.\n\n"
                "Use this when the user wants the content of a web page split into chunks "
                "for embedding, retrieval, or RAG — NOT when they just want to read or "
                "summarize the page (use scrape_url for that).\n\n"
                "If contextual_retrieval=true is requested, follow the PRE-FLIGHT sequence:\n"
                "1. Call verify_provider_key(provider, 'llm') → get live model list\n"
                "2. Ask user to choose a model from the list\n"
                "3. Ask: 'Is this a JavaScript-heavy page or SPA?' → js_render\n"
                "4. Present Contextual Retrieval as a recommended upgrade: 'Would you like "
                "Contextual Retrieval (RAG 2.0)? It enriches each chunk with LLM-generated "
                "context, improving retrieval accuracy by 35–50%. Costs ~$0.001/chunk extra.'\n\n"
                "LLM keys can be omitted if OPENAI_API_KEY, ANTHROPIC_API_KEY, or "
                "GEMINI_API_KEY is set in the MCP environment config."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The web URL to scrape and chunk.",
                    },
                    "selector": {
                        "type": "string",
                        "description": "Optional CSS selector to target a specific element (e.g. 'article', '.content', 'main').",
                    },
                    "chunk_size": {
                        "type": "integer",
                        "description": "Target token count per chunk. Default: 512. Range: 64–4096.",
                        "default": 512,
                        "minimum": 64,
                        "maximum": 4096,
                    },
                    "overlap": {
                        "type": "integer",
                        "description": "Token overlap between consecutive chunks. Default: 50.",
                        "default": 50,
                        "minimum": 0,
                        "maximum": 500,
                    },
                    "js_render": {
                        "type": "boolean",
                        "description": "Use headless Chromium to render JavaScript before scraping. Required for SPAs and JS-heavy pages. Ask the user before enabling. Adds a small surcharge.",
                        "default": False,
                    },
                    "contextual_retrieval": {
                        "type": "boolean",
                        "description": "Enable RAG 2.0 contextual enrichment. An LLM generates a unique context string for each chunk, boosting retrieval accuracy by 35–50%. Present this as a recommended upgrade. Requires llm_provider and llm_model (from verify_provider_key).",
                        "default": False,
                    },
                    "llm_provider": {
                        "type": "string",
                        "description": "LLM provider for contextual retrieval. One of: 'openai', 'anthropic', 'gemini'. Verify with verify_provider_key first.",
                        "enum": ["openai", "anthropic", "gemini"],
                    },
                    "llm_api_key": {
                        "type": "string",
                        "description": "API key for the LLM provider. Can be omitted if set as env var.",
                    },
                    "llm_model": {
                        "type": "string",
                        "description": "LLM model name. MUST be chosen from the list returned by verify_provider_key — do not guess or hardcode.",
                    },
                    "cookies": {
                        "type": "object",
                        "description": (
                            "Optional dict of cookies to include with the request for authenticated scraping "
                            '(e.g. {"session": "abc123", "auth_token": "xyz"}). '
                            "Only used in local-fetch mode (default). Never forwarded to the server. "
                            "Use this for pages behind a login wall — pass your browser session cookie."
                        ),
                        "additionalProperties": {"type": "string"},
                    },
                    "headers": {
                        "type": "object",
                        "description": (
                            "Optional dict of additional HTTP request headers "
                            '(e.g. {"Authorization": "Bearer eyJ..."}). '
                            "Only used in local-fetch mode (default). Never forwarded to the server."
                        ),
                        "additionalProperties": {"type": "string"},
                    },
                    "storage_state": {
                        "type": "object",
                        "description": (
                            "Optional Playwright storage state dict for enterprise SSO/MFA authenticated scraping. "
                            "Captured locally using capture_session() from scrapedatshi[auth] — "
                            "opens a real browser, user logs in manually, session is captured. "
                            "Contains cookies and localStorage tokens. "
                            "Use this for Okta, Duo, or any SSO flow that blocks automated login. "
                            "Load from a saved session.auth.json file."
                        ),
                        "additionalProperties": True,
                    },
                },
                "required": ["url"],
            },
        ),
        # ── crawl_site ────────────────────────────────────────────────────────
        types.Tool(
            name="crawl_site",
            description=(
                "Crawl an entire website, chunk all pages, and return structured JSON chunks. "
                "Two modes: 'sitemap' (reads sitemap.xml — best for docs/blogs) and 'spider' "
                "(follows links — works on any site).\n\n"
                "Use this when the user wants chunks from MULTIPLE pages WITHOUT extracting "
                "structured fields. For structured field extraction across pages, use extract_crawl.\n\n"
                "⚠️ ALWAYS confirm the max_pages limit with the user before calling. "
                "Default is 10 pages. For large sites, warn about credit usage first.\n\n"
                "If contextual_retrieval is requested, follow the PRE-FLIGHT sequence:\n"
                "1. Call verify_provider_key(provider, 'llm') → get live model list\n"
                "2. Ask user to choose a model\n"
                "3. Ask about JS rendering\n"
                "4. Present Contextual Retrieval as a recommended upgrade\n\n"
                "LLM keys can be omitted if set as environment variables."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The root domain or sitemap URL to crawl.",
                    },
                    "max_pages": {
                        "type": "integer",
                        "description": "Maximum pages to crawl. Default: 10. Maximum: 200. Always confirm with user for large sites.",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 200,
                    },
                    "crawl_mode": {
                        "type": "string",
                        "description": "'sitemap': reads sitemap.xml (best for docs/blogs). 'spider': follows links from root URL (works on any site).",
                        "enum": ["sitemap", "spider"],
                        "default": "sitemap",
                    },
                    "selector": {
                        "type": "string",
                        "description": "Optional CSS selector applied to every crawled page.",
                    },
                    "include_pattern": {
                        "type": "string",
                        "description": "Only crawl URLs containing this substring (e.g. '/docs/').",
                    },
                    "exclude_pattern": {
                        "type": "string",
                        "description": "Skip URLs containing this substring (e.g. '/blog/').",
                    },
                    "js_render": {
                        "type": "boolean",
                        "description": "Use headless browser to render JS before scraping each page. Ask the user before enabling. Adds surcharge per page.",
                        "default": False,
                    },
                    "contextual_retrieval": {
                        "type": "boolean",
                        "description": "Enable RAG 2.0 contextual enrichment. Present as a recommended upgrade. Requires llm_provider and llm_model from verify_provider_key.",
                        "default": False,
                    },
                    "llm_provider": {
                        "type": "string",
                        "description": "LLM provider for contextual retrieval. Verify with verify_provider_key first.",
                        "enum": ["openai", "anthropic", "gemini"],
                    },
                    "llm_api_key": {
                        "type": "string",
                        "description": "API key for the LLM provider. Can be omitted if set as env var.",
                    },
                    "llm_model": {
                        "type": "string",
                        "description": "LLM model name from verify_provider_key. Do not guess or hardcode.",
                    },
                    "cookies": {
                        "type": "object",
                        "description": (
                            "Optional dict of cookies for authenticated crawling "
                            '(e.g. {"session": "abc123"}). '
                            "Only sent to URLs within the permitted domain scope — never leaked to external domains. "
                            "Only used in local-fetch mode (default). Never forwarded to the server."
                        ),
                        "additionalProperties": {"type": "string"},
                    },
                    "headers": {
                        "type": "object",
                        "description": (
                            "Optional dict of additional HTTP request headers "
                            '(e.g. {"Authorization": "Bearer eyJ..."}). '
                            "Same domain-isolation rules apply as cookies. "
                            "Only used in local-fetch mode (default)."
                        ),
                        "additionalProperties": {"type": "string"},
                    },
                    "allow_subdomains": {
                        "type": "boolean",
                        "description": (
                            "If true, also crawl subdomains of the root domain "
                            "(e.g. wiki.company.com when root is company.com). "
                            "Cookies and headers are shared across all subdomains within scope. "
                            "Multi-part TLDs (.co.uk, .com.br) are handled safely. "
                            "Default: false (exact domain match only)."
                        ),
                        "default": False,
                    },
                    "storage_state": {
                        "type": "object",
                        "description": (
                            "Optional Playwright storage state dict for enterprise SSO/MFA authenticated crawling. "
                            "Captured locally using capture_session() from scrapedatshi[auth] — "
                            "opens a real browser, user logs in manually, session is captured. "
                            "Contains cookies and localStorage tokens. "
                            "Use this for Okta, Duo, or any SSO flow that blocks automated login. "
                            "Load from a saved session.auth.json file."
                        ),
                        "additionalProperties": True,
                    },
                },
                "required": ["url"],
            },
        ),
        # ── extract_data ──────────────────────────────────────────────────────
        types.Tool(
            name="extract_data",
            description=(
                "Scrape a URL and extract structured data matching a user-defined schema "
                "using an LLM. Returns a JSON object (or array if extract_as_list=true).\n\n"
                "Use this when the user wants specific FIELDS from a page "
                "(e.g. product name, price, stock status; article author, date, summary).\n\n"
                "PRE-FLIGHT REQUIRED — before calling:\n"
                "1. Call verify_provider_key(provider, 'llm') → get live model list\n"
                "2. Present models to user, ask them to choose one\n"
                "3. Ask: 'Is this a JavaScript-heavy page or SPA?' → js_render\n"
                "4. Present Contextual Retrieval is NOT applicable here (extraction only)\n\n"
                "LLM keys can be omitted if OPENAI_API_KEY, ANTHROPIC_API_KEY, or "
                "GEMINI_API_KEY is set in the MCP environment config."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The web URL to scrape and extract structured data from.",
                    },
                    "schema": {
                        "type": "object",
                        "description": (
                            "Dict mapping field names to description strings. "
                            'Example: {"title": "string — the product name", "price": "number — price in USD", "in_stock": "boolean — whether in stock"}'
                        ),
                        "additionalProperties": {"type": "string"},
                    },
                    "llm_provider": {
                        "type": "string",
                        "description": "LLM provider. One of: 'openai', 'anthropic', 'gemini'. Call verify_provider_key first.",
                        "enum": ["openai", "anthropic", "gemini"],
                    },
                    "llm_api_key": {
                        "type": "string",
                        "description": "API key for the LLM provider. Can be omitted if set as env var.",
                    },
                    "llm_model": {
                        "type": "string",
                        "description": "LLM model name from verify_provider_key. Do not guess or hardcode. Use an advanced model (not mini/flash/haiku) for long-form pages like documentation or legal docs.",
                    },
                    "selector": {
                        "type": "string",
                        "description": "Optional CSS selector to target a specific section before extraction.",
                    },
                    "extract_as_list": {
                        "type": "boolean",
                        "description": "If true, extracts ALL matching items on the page as a JSON array. Use for listing pages (product catalogues, article feeds).",
                        "default": False,
                    },
                    "js_render": {
                        "type": "boolean",
                        "description": "Use headless browser to render JS before extracting. Ask the user before enabling.",
                        "default": False,
                    },
                    "click_selector": {
                        "type": "string",
                        "description": "CSS selector for an element to click after page load (tabs, accordions, load-more). Only used when js_render=true.",
                    },
                },
                "required": ["url", "schema", "llm_provider"],
            },
        ),
        # ── extract_crawl ─────────────────────────────────────────────────────
        types.Tool(
            name="extract_crawl",
            description=(
                "Crawl a domain and extract structured data from every page using your LLM. "
                "Each page is processed independently — failed pages return an error without "
                "aborting the batch. Only successfully extracted pages are billed.\n\n"
                "Use this when the user wants structured FIELDS from MULTIPLE pages "
                "(e.g. extract title + price from every product page on a site).\n\n"
                "⚠️ Each page takes 5–15 seconds. Default is 5 pages. For more than 20 pages, "
                "warn the user about wait times and credit usage before proceeding.\n\n"
                "PRE-FLIGHT REQUIRED — before calling:\n"
                "1. Call verify_provider_key(provider, 'llm') → get live model list\n"
                "2. Present models to user, ask them to choose one\n"
                "3. Ask: 'Is this a JavaScript-heavy site?' → js_render (not available for extract_crawl, note this)\n"
                "4. Confirm max_pages with the user\n\n"
                "LLM keys can be omitted if set as environment variables."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The root domain to crawl.",
                    },
                    "schema": {
                        "type": "object",
                        "description": (
                            "Dict mapping field names to description strings. "
                            'Example: {"title": "string — the product name", "price": "number — price in USD"}'
                        ),
                        "additionalProperties": {"type": "string"},
                    },
                    "llm_provider": {
                        "type": "string",
                        "description": "LLM provider. One of: 'openai', 'anthropic', 'gemini'. Call verify_provider_key first.",
                        "enum": ["openai", "anthropic", "gemini"],
                    },
                    "llm_api_key": {
                        "type": "string",
                        "description": "API key for the LLM provider. Can be omitted if set as env var.",
                    },
                    "llm_model": {
                        "type": "string",
                        "description": "LLM model name from verify_provider_key. Do not guess or hardcode. Advanced models (not mini/flash/haiku) use 30k char context — better for long pages.",
                    },
                    "crawl_mode": {
                        "type": "string",
                        "description": "'sitemap': reads sitemap.xml. 'spider': follows links from root URL.",
                        "enum": ["sitemap", "spider"],
                        "default": "sitemap",
                    },
                    "max_pages": {
                        "type": "integer",
                        "description": "Maximum pages to crawl and extract. Default: 5. Maximum: 50. Always confirm with user before setting above 20.",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 50,
                    },
                    "selector": {
                        "type": "string",
                        "description": "Optional CSS selector applied to every page before extraction.",
                    },
                    "include_pattern": {
                        "type": "string",
                        "description": "Only crawl URLs containing this substring (e.g. '/products/').",
                    },
                    "exclude_pattern": {
                        "type": "string",
                        "description": "Skip URLs containing this substring (e.g. '/blog/').",
                    },
                    "extract_as_list": {
                        "type": "boolean",
                        "description": "If true, extracts ALL matching items on each page as a JSON array.",
                        "default": False,
                    },
                },
                "required": ["url", "schema", "llm_provider"],
            },
        ),
        # ── sync_to_vectordb ──────────────────────────────────────────────────
        types.Tool(
            name="sync_to_vectordb",
            description=(
                "Full RAG pipeline: scrape a URL, embed the chunks using your embedding "
                "provider, and inject the vectors into your vector database — all in one call.\n\n"
                "Use this when the user wants to ADD web content to their vector DB for "
                "later retrieval. The user brings their own embedding provider and vector DB.\n\n"
                "PRE-FLIGHT REQUIRED — before calling:\n"
                "1. Call verify_provider_key(embedding_provider, 'embedding') → get live embedding model list\n"
                "2. Present models to user, ask them to choose one\n"
                "3. Call list_vector_db_providers if user is unsure what config fields are needed\n"
                "4. Ask: 'Is this a JavaScript-heavy page or SPA?' → js_render\n"
                "5. Present Contextual Retrieval as a recommended upgrade: 'Would you like "
                "Contextual Retrieval (RAG 2.0)? It enriches each chunk with LLM-generated "
                "context before embedding, improving retrieval accuracy by 35–50%. "
                "Costs ~$0.001/chunk extra. If yes, I'll also need your LLM provider and model.'\n"
                "6. If contextual_retrieval=yes: call verify_provider_key(llm_provider, 'llm') too\n\n"
                "Keys can be omitted if set as environment variables (OPENAI_API_KEY, "
                "PINECONE_API_KEY, etc.)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The web URL to scrape, embed, and inject into the vector DB.",
                    },
                    "embedding_provider": {
                        "type": "string",
                        "description": "Embedding provider. Call verify_provider_key(provider, 'embedding') first to get available models.",
                        "enum": [
                            "openai",
                            "cohere",
                            "gemini",
                            "mistral",
                            "voyage",
                            "ollama",
                        ],
                    },
                    "embedding_model": {
                        "type": "string",
                        "description": "Embedding model name from verify_provider_key. Do not guess or hardcode.",
                    },
                    "embedding_api_key": {
                        "type": "string",
                        "description": "API key for the embedding provider. Can be omitted if set as env var.",
                    },
                    "embedding_endpoint": {
                        "type": "string",
                        "description": "Public HTTPS endpoint for Ollama only (e.g. from ngrok). Not needed for cloud providers.",
                    },
                    "vector_db": {
                        "type": "string",
                        "description": "Vector DB provider. Call list_vector_db_providers to see required config fields for each.",
                        "enum": [
                            "pinecone",
                            "qdrant",
                            "chroma",
                            "supabase",
                            "weaviate",
                            "mongodb",
                            "azure_cosmos",
                            "azure_cosmos_mongo",
                            "lancedb",
                        ],
                    },
                    "vector_db_config": {
                        "type": "object",
                        "description": (
                            "Provider-specific config. Call list_vector_db_providers for required fields. "
                            "API keys within this config can be omitted if set as env vars. "
                            'Examples: pinecone: {"index_host": "https://my-index.svc.pinecone.io"} | '
                            'qdrant: {"url": "https://cluster.qdrant.io", "collection_name": "docs"} | '
                            'supabase: {"connection_string": "postgresql://...", "table_name": "documents"} | '
                            'chroma: {"collection_name": "docs"}'
                        ),
                        "additionalProperties": True,
                    },
                    "selector": {
                        "type": "string",
                        "description": "Optional CSS selector to target a specific page section.",
                    },
                    "chunk_size": {
                        "type": "integer",
                        "description": "Target token count per chunk. Default: 512.",
                        "default": 512,
                        "minimum": 64,
                        "maximum": 4096,
                    },
                    "overlap": {
                        "type": "integer",
                        "description": "Token overlap between consecutive chunks. Default: 50.",
                        "default": 50,
                        "minimum": 0,
                        "maximum": 500,
                    },
                    "js_render": {
                        "type": "boolean",
                        "description": "Use headless browser to render JS before scraping. Ask the user before enabling.",
                        "default": False,
                    },
                    "contextual_retrieval": {
                        "type": "boolean",
                        "description": "Enable RAG 2.0 contextual enrichment before embedding. Present as a recommended upgrade. Requires llm_provider and llm_model.",
                        "default": False,
                    },
                    "llm_provider": {
                        "type": "string",
                        "description": "LLM provider for contextual retrieval. Verify with verify_provider_key(provider, 'llm') first.",
                        "enum": ["openai", "anthropic", "gemini"],
                    },
                    "llm_api_key": {
                        "type": "string",
                        "description": "API key for the LLM provider. Can be omitted if set as env var.",
                    },
                    "llm_model": {
                        "type": "string",
                        "description": "LLM model name from verify_provider_key. Do not guess or hardcode.",
                    },
                },
                "required": [
                    "url",
                    "embedding_provider",
                    "vector_db",
                    "vector_db_config",
                ],
            },
        ),
        # ── chunk_file ────────────────────────────────────────────────────────
        types.Tool(
            name="chunk_file",
            description=(
                "Parse a local file and chunk its content into RAG-ready text segments. "
                "Returns structured JSON chunks. No embedding or vector DB required.\n\n"
                "Supported file formats: .pdf, .md, .txt, .yaml, .yml, .json, .csv, .xlsx, .xls,\n"
                ".docx, .ipynb, .html, .htm, .xml, .toml, .ini, .cfg, and all common code files\n"
                "(.py, .js, .ts, .jsx, .tsx, .sql, .go, .rb, .java, .cs, .cpp, .c, .rs, .php,\n"
                ".sh, .bash, .zsh, .r, .swift, .kt, .scala)\n"
                "Maximum file size: 50 MB\n\n"
                "CODE-AWARE CHUNKING: .py files are split by top-level class and function using "
                "Python's AST parser — each class/function becomes its own chunk with imports "
                "prepended for context. .sql files are split by statement block (CREATE TABLE, "
                "INSERT INTO, SELECT, etc.). Both fall back to plain text if parsing fails.\n\n"
                "FETCH MODE: By default (local-fetch mode), the file is parsed on the "
                "user's machine using their own CPU — no heavy PDF processing on the server. "
                "The extracted text is sent to the server for chunking only. "
                "This is cheaper ($0.0020/file) and keeps large PDF processing off the server.\n\n"
                "For scanned/image-only PDFs that need OCR, the MCP server must be configured "
                "with SCRAPEDATSHI_FETCH_MODE=server in the Claude Desktop config — this routes "
                "the file through the server's OCR pipeline ($0.0040/file).\n\n"
                "PDF DEPENDENCY: Local PDF parsing requires pdfplumber. If not installed, "
                "the tool will raise an ImportError with install instructions. "
                "Install with: pip install pdfplumber\n\n"
                "Use this when the user says 'chunk this PDF', 'process this document', "
                "'read this file', or wants to extract text from a local file.\n\n"
                "Provide the ABSOLUTE path to the file on the user's local machine "
                "(e.g. 'C:/Users/user/Documents/report.pdf' or '/home/user/docs/manual.pdf').\n\n"
                "If contextual_retrieval=true is requested, follow the PRE-FLIGHT sequence:\n"
                "1. Call verify_provider_key(provider, 'llm') → get live model list\n"
                "2. Ask user to choose a model\n"
                "3. Present Contextual Retrieval as a recommended upgrade\n\n"
                "LLM keys can be omitted if set as environment variables."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": (
                            "Absolute path to the local file to chunk. "
                            "Supported: .pdf, .md, .txt, .yaml, .yml, .json, .csv, .xlsx, .xls, "
                            ".docx, .ipynb, .html, .htm, .xml, .toml, .ini, .cfg, "
                            ".py (AST-aware: split by class/function), "
                            ".sql (statement-aware: split by CREATE/INSERT/SELECT blocks), "
                            ".js, .ts, .jsx, .tsx, .go, .rb, .java, .cs, .cpp, .c, .rs, .php, "
                            ".sh, .bash, .zsh, .r, .swift, .kt, .scala. "
                            "Example: 'C:/Users/user/Documents/report.pdf'"
                        ),
                    },
                    "chunk_size": {
                        "type": "integer",
                        "description": "Target token count per chunk. Default: 512. Range: 64–4096.",
                        "default": 512,
                        "minimum": 64,
                        "maximum": 4096,
                    },
                    "overlap": {
                        "type": "integer",
                        "description": "Token overlap between consecutive chunks. Default: 50.",
                        "default": 50,
                        "minimum": 0,
                        "maximum": 500,
                    },
                    "contextual_retrieval": {
                        "type": "boolean",
                        "description": "Enable RAG 2.0 contextual enrichment. Present as a recommended upgrade. Requires llm_provider and llm_model from verify_provider_key.",
                        "default": False,
                    },
                    "llm_provider": {
                        "type": "string",
                        "description": "LLM provider for contextual retrieval. Verify with verify_provider_key first.",
                        "enum": ["openai", "anthropic", "gemini"],
                    },
                    "llm_api_key": {
                        "type": "string",
                        "description": "API key for the LLM provider. Can be omitted if set as env var.",
                    },
                    "llm_model": {
                        "type": "string",
                        "description": "LLM model name from verify_provider_key. Do not guess or hardcode.",
                    },
                },
                "required": ["file_path"],
            },
        ),
        # ── ingest_file ───────────────────────────────────────────────────────
        types.Tool(
            name="ingest_file",
            description=(
                "Full RAG pipeline for local files: upload a file, embed the chunks using "
                "your embedding provider, and inject the vectors into your vector database.\n\n"
                "Supported file formats: .pdf, .md, .txt, .yaml, .yml, .json, .csv, .xlsx, .xls,\n"
                ".docx, .ipynb, .html, .htm, .xml, .toml, .ini, .cfg, and all common code files\n"
                "(.py, .js, .ts, .jsx, .tsx, .sql, .go, .rb, .java, .cs, .cpp, .c, .rs, .php,\n"
                ".sh, .bash, .zsh, .r, .swift, .kt, .scala)\n"
                "Maximum file size: 50 MB\n\n"
                "Use this when the user wants to ADD a local document (PDF, markdown, etc.) "
                "to their vector DB. This is the file-based equivalent of sync_to_vectordb.\n\n"
                "Provide the ABSOLUTE path to the file on the user's local machine.\n\n"
                "PRE-FLIGHT REQUIRED — before calling:\n"
                "1. Call verify_provider_key(embedding_provider, 'embedding') → get live embedding model list\n"
                "2. Present models to user, ask them to choose one\n"
                "3. Call list_vector_db_providers if user is unsure what config fields are needed\n"
                "4. Present Contextual Retrieval as a recommended upgrade: 'Would you like "
                "Contextual Retrieval (RAG 2.0)? It enriches each chunk with LLM-generated "
                "context before embedding, improving retrieval accuracy by 35–50%. "
                "Costs ~$0.001/chunk extra.'\n"
                "5. If contextual_retrieval=yes: call verify_provider_key(llm_provider, 'llm') too\n\n"
                "Keys can be omitted if set as environment variables."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the local file to ingest. Supported: .pdf, .md, .txt, .yaml, .yml, .json. Example: 'C:/Users/user/Documents/report.pdf'",
                    },
                    "embedding_provider": {
                        "type": "string",
                        "description": "Embedding provider. Call verify_provider_key(provider, 'embedding') first to get available models.",
                        "enum": [
                            "openai",
                            "cohere",
                            "gemini",
                            "mistral",
                            "voyage",
                            "ollama",
                        ],
                    },
                    "embedding_model": {
                        "type": "string",
                        "description": "Embedding model name from verify_provider_key. Do not guess or hardcode.",
                    },
                    "embedding_api_key": {
                        "type": "string",
                        "description": "API key for the embedding provider. Can be omitted if set as env var.",
                    },
                    "embedding_endpoint": {
                        "type": "string",
                        "description": "Public HTTPS endpoint for Ollama only (e.g. from ngrok). Not needed for cloud providers.",
                    },
                    "vector_db": {
                        "type": "string",
                        "description": "Vector DB provider. Call list_vector_db_providers to see required config fields for each.",
                        "enum": [
                            "pinecone",
                            "qdrant",
                            "chroma",
                            "supabase",
                            "weaviate",
                            "mongodb",
                            "azure_cosmos",
                            "azure_cosmos_mongo",
                            "lancedb",
                        ],
                    },
                    "vector_db_config": {
                        "type": "object",
                        "description": (
                            "Provider-specific config. Call list_vector_db_providers for required fields. "
                            "API keys within this config can be omitted if set as env vars."
                        ),
                        "additionalProperties": True,
                    },
                    "chunk_size": {
                        "type": "integer",
                        "description": "Target token count per chunk. Default: 512.",
                        "default": 512,
                        "minimum": 64,
                        "maximum": 4096,
                    },
                    "overlap": {
                        "type": "integer",
                        "description": "Token overlap between consecutive chunks. Default: 50.",
                        "default": 50,
                        "minimum": 0,
                        "maximum": 500,
                    },
                    "contextual_retrieval": {
                        "type": "boolean",
                        "description": "Enable RAG 2.0 contextual enrichment before embedding. Present as a recommended upgrade. Requires llm_provider and llm_model.",
                        "default": False,
                    },
                    "llm_provider": {
                        "type": "string",
                        "description": "LLM provider for contextual retrieval. Verify with verify_provider_key(provider, 'llm') first.",
                        "enum": ["openai", "anthropic", "gemini"],
                    },
                    "llm_api_key": {
                        "type": "string",
                        "description": "API key for the LLM provider. Can be omitted if set as env var.",
                    },
                    "llm_model": {
                        "type": "string",
                        "description": "LLM model name from verify_provider_key. Do not guess or hardcode.",
                    },
                },
                "required": [
                    "file_path",
                    "embedding_provider",
                    "vector_db",
                    "vector_db_config",
                ],
            },
        ),
        # ── autorag ───────────────────────────────────────────────────────────
        types.Tool(
            name="autorag",
            description=(
                "Full AutoRAG pipeline: crawl an entire domain, chunk every page, embed all "
                "chunks, and inject into your vector database — all in a single call.\n\n"
                "Use this when the user wants to bulk-ingest an entire website into their "
                "vector DB. This combines crawl_site + sync_to_vectordb into one operation.\n\n"
                "⚠️ ALWAYS confirm the max_pages limit with the user before calling. "
                "Default is 5 pages. Each page is fetched, chunked, embedded, and injected. "
                "For large sites, warn about credit usage and wait times first.\n\n"
                "PRE-FLIGHT REQUIRED — before calling:\n"
                "1. Call verify_provider_key(embedding_provider, 'embedding') → get live embedding model list\n"
                "2. Present models to user, ask them to choose one\n"
                "3. Call list_vector_db_providers if user is unsure what config fields are needed\n"
                "4. Confirm max_pages with the user\n"
                "5. Present Contextual Retrieval as a recommended upgrade: 'Would you like "
                "Contextual Retrieval (RAG 2.0)? It enriches each chunk with LLM-generated "
                "context before embedding, improving retrieval accuracy by 35–50%. "
                "Costs ~$0.001/chunk extra.'\n"
                "6. If contextual_retrieval=yes: call verify_provider_key(llm_provider, 'llm') too\n\n"
                "Keys can be omitted if set as environment variables."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The root domain to crawl (e.g. 'https://docs.example.com').",
                    },
                    "embedding_provider": {
                        "type": "string",
                        "description": "Embedding provider. Call verify_provider_key(provider, 'embedding') first to get available models.",
                        "enum": [
                            "openai",
                            "cohere",
                            "gemini",
                            "mistral",
                            "voyage",
                            "ollama",
                        ],
                    },
                    "embedding_model": {
                        "type": "string",
                        "description": "Embedding model name from verify_provider_key. Do not guess or hardcode.",
                    },
                    "embedding_api_key": {
                        "type": "string",
                        "description": "API key for the embedding provider. Can be omitted if set as env var.",
                    },
                    "vector_db": {
                        "type": "string",
                        "description": "Vector DB provider. Call list_vector_db_providers to see required config fields for each.",
                        "enum": [
                            "pinecone",
                            "qdrant",
                            "chroma",
                            "supabase",
                            "weaviate",
                            "mongodb",
                            "azure_cosmos",
                            "azure_cosmos_mongo",
                            "lancedb",
                        ],
                    },
                    "vector_db_config": {
                        "type": "object",
                        "description": (
                            "Provider-specific config. Call list_vector_db_providers for required fields. "
                            "API keys within this config can be omitted if set as env vars."
                        ),
                        "additionalProperties": True,
                    },
                    "crawl_mode": {
                        "type": "string",
                        "description": "'sitemap': reads sitemap.xml (best for docs/blogs). 'spider': follows links from root URL (works on any site).",
                        "enum": ["sitemap", "spider"],
                        "default": "sitemap",
                    },
                    "max_pages": {
                        "type": "integer",
                        "description": "Maximum pages to crawl and inject. Default: 5. Maximum: 200. Always confirm with user for large sites.",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 200,
                    },
                    "include_pattern": {
                        "type": "string",
                        "description": "Only crawl URLs containing this substring (e.g. '/docs/').",
                    },
                    "exclude_pattern": {
                        "type": "string",
                        "description": "Skip URLs containing this substring (e.g. '/blog/').",
                    },
                    "selector": {
                        "type": "string",
                        "description": "Optional CSS selector applied to every page before chunking.",
                    },
                    "chunk_size": {
                        "type": "integer",
                        "description": "Target token count per chunk. Default: 512.",
                        "default": 512,
                        "minimum": 64,
                        "maximum": 4096,
                    },
                    "overlap": {
                        "type": "integer",
                        "description": "Token overlap between consecutive chunks. Default: 50.",
                        "default": 50,
                        "minimum": 0,
                        "maximum": 500,
                    },
                    "contextual_retrieval": {
                        "type": "boolean",
                        "description": "Enable RAG 2.0 contextual enrichment before embedding. Present as a recommended upgrade. Requires llm_provider and llm_model.",
                        "default": False,
                    },
                    "llm_provider": {
                        "type": "string",
                        "description": "LLM provider for contextual retrieval. Verify with verify_provider_key(provider, 'llm') first.",
                        "enum": ["openai", "anthropic", "gemini"],
                    },
                    "llm_api_key": {
                        "type": "string",
                        "description": "API key for the LLM provider. Can be omitted if set as env var.",
                    },
                    "llm_model": {
                        "type": "string",
                        "description": "LLM model name from verify_provider_key. Do not guess or hardcode.",
                    },
                },
                "required": [
                    "url",
                    "embedding_provider",
                    "vector_db",
                    "vector_db_config",
                ],
            },
        ),
        # ── ingest_scraped ────────────────────────────────────────────────────
        types.Tool(
            name="ingest_scraped",
            description=(
                "Bulk ingest a folder of pre-scraped files — chunk, embed, and inject "
                "into your vector database. Designed for users who have content from external "
                "scrapers (Scrapy, Playwright, wget, etc.) and want to embed it into their vector DB.\n\n"
                "Supports all common file types: .md, .txt, .json, .yaml, .yml, .csv, .xlsx, .xls, "
                ".docx, .ipynb, .html, .htm, .xml, .toml, .ini, .cfg, and all code files "
                "(.py, .js, .ts, .sql, .go, .rb, .java, .cs, .cpp, .c, .rs, .php, .sh, etc.).\n\n"
                "CODE-AWARE CHUNKING: .py files are split by top-level class and function (AST-aware). "
                ".sql files are split by statement block (CREATE TABLE, INSERT INTO, SELECT, etc.).\n\n"
                "For JSON files, automatically detects Scrapy/crawler array exports and extracts "
                "text from each item individually.\n\n"
                "All processing runs locally on the machine running Claude Desktop — no server-side "
                "crawling. Rate limit errors from your embedding provider are handled automatically "
                "with exponential backoff.\n\n"
                "Provide the ABSOLUTE path to the folder on the user's local machine.\n\n"
                "PRE-FLIGHT REQUIRED — before calling:\n"
                "1. Call verify_provider_key(embedding_provider, 'embedding') → get live embedding model list\n"
                "2. Present models to user, ask them to choose one\n"
                "3. Call list_vector_db_providers if user is unsure what config fields are needed\n"
                "4. Confirm max_files with the user for large datasets\n\n"
                "Keys can be omitted if set as environment variables."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "folder_path": {
                        "type": "string",
                        "description": "Absolute path to the folder containing files to ingest. Example: 'C:/Users/user/scrapy_output/' or '/home/user/data/'",
                    },
                    "embedding_provider": {
                        "type": "string",
                        "description": "Embedding provider. Call verify_provider_key(provider, 'embedding') first to get available models.",
                        "enum": [
                            "openai",
                            "cohere",
                            "gemini",
                            "mistral",
                            "voyage",
                            "ollama",
                        ],
                    },
                    "embedding_model": {
                        "type": "string",
                        "description": "Embedding model name from verify_provider_key. Do not guess or hardcode.",
                    },
                    "embedding_api_key": {
                        "type": "string",
                        "description": "API key for the embedding provider. Can be omitted if set as env var.",
                    },
                    "vector_db": {
                        "type": "string",
                        "description": "Vector DB provider. Call list_vector_db_providers to see required config fields for each.",
                        "enum": [
                            "pinecone",
                            "qdrant",
                            "chroma",
                            "supabase",
                            "weaviate",
                            "mongodb",
                            "azure_cosmos",
                            "azure_cosmos_mongo",
                            "lancedb",
                        ],
                    },
                    "vector_db_config": {
                        "type": "object",
                        "description": (
                            "Provider-specific config. Call list_vector_db_providers for required fields. "
                            "API keys within this config can be omitted if set as env vars."
                        ),
                        "additionalProperties": True,
                    },
                    "file_extensions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File extensions to process. Default: ['.md', '.txt', '.json', '.yaml', '.yml']. Example: ['.json'] to process only JSON files.",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "If true, recurse into subdirectories. Default: true.",
                        "default": True,
                    },
                    "max_files": {
                        "type": "integer",
                        "description": "Maximum number of files to process. Default: unlimited. Confirm with user for large datasets.",
                        "minimum": 1,
                    },
                    "batch_delay": {
                        "type": "number",
                        "description": "Seconds to wait between files (default: 0.5). Increase to 1.0–2.0 for large batches to avoid embedding provider rate limits.",
                        "default": 0.5,
                        "minimum": 0,
                    },
                    "json_text_keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Keys to look for when extracting text from JSON items. Default: ['text', 'content', 'html', 'body', 'markdown', 'description']. Used for Scrapy/crawler JSON array exports.",
                    },
                },
                "required": [
                    "folder_path",
                    "embedding_provider",
                    "vector_db",
                    "vector_db_config",
                ],
            },
        ),
        # ── list_embedding_providers ──────────────────────────────────────────
        types.Tool(
            name="list_embedding_providers",
            description=(
                "Returns all supported embedding providers with labels and notes. "
                "Call this to help the user choose an embedding provider before sync_to_vectordb. "
                "After the user chooses, call verify_provider_key to get the live model list."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        # ── list_vector_db_providers ──────────────────────────────────────────
        types.Tool(
            name="list_vector_db_providers",
            description=(
                "Returns all supported vector database providers with required config fields, "
                "optional fields, and setup notes. Call this before sync_to_vectordb to help "
                "the user understand what vector_db_config fields they need to provide."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
    ]


# ── Tool call handlers ────────────────────────────────────────────────────────


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    try:
        if name == "verify_provider_key":
            return await _handle_verify_provider_key(arguments)
        elif name == "get_usage_guide":
            return _handle_get_usage_guide()
        elif name == "scrape_url":
            return await _handle_scrape_url_markdown(arguments)
        elif name == "chunk_url":
            return await _handle_chunk_url(arguments)
        elif name == "crawl_site":
            return await _handle_crawl_site(arguments)
        elif name == "extract_data":
            return await _handle_extract_data(arguments)
        elif name == "extract_crawl":
            return await _handle_extract_crawl(arguments)
        elif name == "sync_to_vectordb":
            return await _handle_sync_to_vectordb(arguments)
        elif name == "chunk_file":
            return await _handle_chunk_file(arguments)
        elif name == "ingest_file":
            return await _handle_ingest_file(arguments)
        elif name == "autorag":
            return await _handle_autorag(arguments)
        elif name == "inspect_vectordb":
            return await _handle_inspect_vectordb(arguments)
        elif name == "query_vectordb":
            return await _handle_query_vectordb(arguments)
        elif name == "rag_chat":
            return await _handle_rag_chat(arguments)
        elif name == "ingest_scraped":
            return await _handle_ingest_folder(arguments)
        elif name == "list_embedding_providers":
            return _handle_list_embedding_providers()
        elif name == "list_vector_db_providers":
            return _handle_list_vector_db_providers()
        else:
            return [types.TextContent(type="text", text=f"❌ Unknown tool: {name}")]
    except Exception as exc:
        return [types.TextContent(type="text", text=_format_error(exc))]


# ── Individual tool handlers ──────────────────────────────────────────────────


async def _handle_verify_provider_key(arguments: dict) -> list[types.TextContent]:
    provider = arguments.get("provider", "").lower().strip()
    provider_type = arguments.get("provider_type", "").lower().strip()

    if not provider:
        return [types.TextContent(type="text", text="❌ 'provider' is required.")]
    if not provider_type:
        return [
            types.TextContent(
                type="text",
                text="❌ 'provider_type' is required. One of: 'llm', 'embedding'.",
            )
        ]

    # Resolve API key
    api_key = arguments.get("api_key")
    if not api_key:
        env_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "cohere": "COHERE_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "voyage": "VOYAGE_API_KEY",
        }
        env_var = env_map.get(provider)
        if env_var:
            api_key = os.environ.get(env_var)
        if not api_key:
            return [
                types.TextContent(
                    type="text",
                    text=(
                        f"❌ No API key found for '{provider}'. "
                        f"Pass api_key as an argument, or set {env_var or 'the corresponding env var'} "
                        "in your MCP environment config."
                    ),
                )
            ]

    result = await _run_discovery(provider, provider_type, api_key)

    if not result["valid"]:
        return [
            types.TextContent(
                type="text",
                text=f"❌ Key verification failed for {provider} ({provider_type}):\n{result['error']}",
            )
        ]

    models = result["models"]
    lines = [
        f"✅ Key verified: {provider} ({provider_type})",
        f"📋 Available models ({len(models)} found):",
    ]
    for m in models:
        lines.append(f"  • {m}")
    lines.append("\nAsk the user to choose a model from this list before proceeding.")

    return [types.TextContent(type="text", text="\n".join(lines))]


def _handle_get_usage_guide() -> list[types.TextContent]:
    guide = """# scrapedatshi MCP — Usage Guide

## Which tool to use?

| Goal | Tool |
|---|---|
| Get clean Markdown from a web page | `scrape_url` |
| Get RAG-ready chunks from a web page | `chunk_url` |
| Chunk a local file (PDF, MD, TXT, etc.) | `chunk_file` |
| Get chunks from multiple pages | `crawl_site` |
| Extract specific fields from one page | `extract_data` |
| Extract specific fields from many pages | `extract_crawl` |
| Add web content to a vector DB | `sync_to_vectordb` |
| Add a local file to a vector DB | `ingest_file` |
| Bulk-ingest an entire website to a vector DB | `autorag` |
| Check what embedding providers are available | `list_embedding_providers` |
| Check what vector DB config fields are needed | `list_vector_db_providers` |
| Verify an API key + get live model list | `verify_provider_key` |

---

## Pre-flight sequence (for any LLM or embedding operation)

**Always follow this sequence before calling scrape_url (with CR), crawl_site (with CR), extract_data, extract_crawl, or sync_to_vectordb:**

1. **Verify the key** → call `verify_provider_key(provider, provider_type)` to get the live model list
2. **Present models** → show the list to the user and ask them to choose one
3. **Ask about JS rendering** → "Is this a JavaScript-heavy page or single-page app (SPA)? If yes, I'll enable JS rendering (adds a small surcharge per page)."
4. **Present Contextual Retrieval as a recommended upgrade** → "Would you like to enable Contextual Retrieval (RAG 2.0)? It uses your LLM to generate a unique context summary for each chunk before embedding, improving retrieval accuracy by 35–50%. Recommended for production RAG pipelines. Costs ~$0.001/chunk extra."
5. **Call the tool** with all confirmed parameters

---

## Key rules

- **Never hardcode or guess model names** — always use `verify_provider_key` to get the live list
- **Always confirm max_pages** with the user before crawl_site or extract_crawl
- **Do not retry on rate limit errors** — inform the user and wait for their instruction
- **Contextual Retrieval is a recommended upgrade** — present it as a quality improvement, not just an option

---

## Supported providers

**LLM** (for extract_data, extract_crawl, contextual_retrieval):
- `openai`, `anthropic`, `gemini`

**Embedding** (for sync_to_vectordb):
- `openai`, `cohere`, `gemini`, `mistral`, `voyage`, `ollama`

**Vector databases** (for sync_to_vectordb):
- `pinecone`, `qdrant`, `chroma`, `supabase`, `weaviate`, `mongodb`, `azure_cosmos`, `azure_cosmos_mongo`, `lancedb`

---

## Environment variables (set in claude_desktop_config.json)

```
SCRAPEDATSHI_API_KEY  — required
OPENAI_API_KEY        — OpenAI LLM + embedding
ANTHROPIC_API_KEY     — Anthropic LLM
GEMINI_API_KEY        — Google Gemini LLM + embedding
COHERE_API_KEY        — Cohere embedding
MISTRAL_API_KEY       — Mistral embedding
VOYAGE_API_KEY        — Voyage AI embedding
PINECONE_API_KEY      — Pinecone vector DB
QDRANT_API_KEY        — Qdrant vector DB (optional)
WEAVIATE_API_KEY      — Weaviate vector DB (optional)
```

When env vars are set, keys are resolved automatically — users don't need to type them in chat.
"""
    return [types.TextContent(type="text", text=guide)]


async def _handle_scrape_url_markdown(arguments: dict) -> list[types.TextContent]:
    """Handler for scrape_url — returns clean Markdown (no chunking)."""
    url = arguments.get("url")
    if not url:
        return [types.TextContent(type="text", text="❌ 'url' is required.")]

    client = _get_client()
    try:
        result = client.pipeline.scrape_url(
            url=url,
            selector=arguments.get("selector"),
            js_render=arguments.get("js_render", False),
            cookies=arguments.get("cookies") or None,
            headers=arguments.get("headers") or None,
            storage_state=arguments.get("storage_state") or None,
        )
    finally:
        client.close()

    lines = [
        f"✅ Scraped: {result.source}",
    ]
    if result.title:
        lines.append(f"📄 Title: {result.title}")
    lines.append(
        f"💳 Credits used: ${result.credits_used:.4f} | Remaining: ${result.credits_remaining:.4f}"
    )
    if result.content_truncated:
        lines.append("⚠️  Content was truncated (exceeded ~75,000 words)")

    lines.append("\n--- Markdown ---")
    lines.append(result.markdown)

    return [types.TextContent(type="text", text="\n".join(lines))]


async def _handle_chunk_url(arguments: dict) -> list[types.TextContent]:
    """Handler for chunk_url — returns RAG-ready chunks."""
    url = arguments.get("url")
    if not url:
        return [types.TextContent(type="text", text="❌ 'url' is required.")]

    contextual_retrieval = arguments.get("contextual_retrieval", False)
    llm_provider = arguments.get("llm_provider")
    llm_api_key = None
    if contextual_retrieval:
        llm_api_key = _resolve_llm_key(arguments, llm_provider)
        if not llm_api_key:
            return [
                types.TextContent(
                    type="text",
                    text=(
                        "❌ contextual_retrieval=true requires an LLM API key. "
                        "Pass llm_api_key as an argument, or set OPENAI_API_KEY / "
                        "ANTHROPIC_API_KEY / GEMINI_API_KEY in your MCP environment config."
                    ),
                )
            ]
        if not arguments.get("llm_model"):
            return [
                types.TextContent(
                    type="text",
                    text=(
                        "❌ llm_model is required when contextual_retrieval=true. "
                        "Call verify_provider_key first to get the available model list, "
                        "then ask the user to choose one."
                    ),
                )
            ]

    client = _get_client()
    try:
        result = client.pipeline.chunk_url(
            url=url,
            selector=arguments.get("selector"),
            chunk_size=arguments.get("chunk_size", 512),
            overlap=arguments.get("overlap", 50),
            js_render=arguments.get("js_render", False),
            contextual_retrieval=contextual_retrieval,
            llm_provider=llm_provider if contextual_retrieval else None,
            llm_api_key=llm_api_key,
            llm_model=arguments.get("llm_model"),
            cookies=arguments.get("cookies") or None,
            headers=arguments.get("headers") or None,
            storage_state=arguments.get("storage_state") or None,
        )
    finally:
        client.close()

    lines = [
        f"✅ Chunked: {result.source}",
        f"📦 Chunks: {result.total_chunks}",
        f"💳 Credits used: ${result.credits_used:.4f} | Remaining: ${result.credits_remaining:.4f}",
    ]
    if result.contextual_retrieval_used:
        lines.append("🧠 Contextual Retrieval: enabled")
    if result.contextual_retrieval_error:
        lines.append(
            f"⚠️  Contextual Retrieval warning: {result.contextual_retrieval_error}"
        )
    if result.content_truncated:
        lines.append("⚠️  Content was truncated (exceeded ~75,000 words)")

    lines.append("\n--- Chunks ---")
    for i, chunk in enumerate(result.chunks, 1):
        preview = chunk.content[:300].replace("\n", " ")
        lines.append(
            f"\n[Chunk {i} | ~{chunk.token_estimate} tokens]\n{preview}{'...' if len(chunk.content) > 300 else ''}"
        )

    return [types.TextContent(type="text", text="\n".join(lines))]


async def _handle_crawl_site(arguments: dict) -> list[types.TextContent]:
    url = arguments.get("url")
    if not url:
        return [types.TextContent(type="text", text="❌ 'url' is required.")]

    contextual_retrieval = arguments.get("contextual_retrieval", False)
    llm_provider = arguments.get("llm_provider")
    llm_api_key = None
    if contextual_retrieval:
        llm_api_key = _resolve_llm_key(arguments, llm_provider)
        if not llm_api_key:
            return [
                types.TextContent(
                    type="text",
                    text=(
                        "❌ contextual_retrieval=true requires an LLM API key. "
                        "Pass llm_api_key as an argument, or set OPENAI_API_KEY / "
                        "ANTHROPIC_API_KEY / GEMINI_API_KEY in your MCP environment config."
                    ),
                )
            ]
        if not arguments.get("llm_model"):
            return [
                types.TextContent(
                    type="text",
                    text=(
                        "❌ llm_model is required when contextual_retrieval=true. "
                        "Call verify_provider_key first to get the available model list."
                    ),
                )
            ]

    client = _get_client()
    try:
        result = client.pipeline.crawl(
            url=url,
            max_pages=arguments.get("max_pages", 10),
            crawl_mode=arguments.get("crawl_mode", "sitemap"),
            selector=arguments.get("selector"),
            include_pattern=arguments.get("include_pattern"),
            exclude_pattern=arguments.get("exclude_pattern"),
            js_render=arguments.get("js_render", False),
            contextual_retrieval=contextual_retrieval,
            llm_provider=llm_provider if contextual_retrieval else None,
            llm_api_key=llm_api_key,
            llm_model=arguments.get("llm_model"),
            cookies=arguments.get("cookies") or None,
            headers=arguments.get("headers") or None,
            allow_subdomains=arguments.get("allow_subdomains", False),
            storage_state=arguments.get("storage_state") or None,
        )
    finally:
        client.close()

    lines = [
        f"✅ Crawled: {result.source_url}",
        f"📄 Pages crawled: {result.pages_crawled}",
        f"📦 Total chunks: {result.total_chunks}",
        f"💳 Credits used: ${result.credits_used:.4f} | Remaining: ${result.credits_remaining:.4f}",
    ]
    if result.contextual_retrieval_used:
        lines.append("🧠 Contextual Retrieval: enabled")
    if result.contextual_retrieval_error:
        lines.append(
            f"⚠️  Contextual Retrieval warning: {result.contextual_retrieval_error}"
        )

    lines.append("\n--- Chunks (first 20 shown) ---")
    for i, chunk in enumerate(result.chunks[:20], 1):
        preview = chunk.content[:200].replace("\n", " ")
        lines.append(
            f"\n[Chunk {i} | ~{chunk.token_estimate} tokens]\n{preview}{'...' if len(chunk.content) > 200 else ''}"
        )
    if result.total_chunks > 20:
        lines.append(f"\n... and {result.total_chunks - 20} more chunks.")

    return [types.TextContent(type="text", text="\n".join(lines))]


async def _handle_extract_data(arguments: dict) -> list[types.TextContent]:
    url = arguments.get("url")
    schema = arguments.get("schema")
    llm_provider = arguments.get("llm_provider")

    if not url:
        return [types.TextContent(type="text", text="❌ 'url' is required.")]
    if not schema:
        return [
            types.TextContent(
                type="text",
                text="❌ 'schema' is required (dict of field_name → description).",
            )
        ]
    if not llm_provider:
        return [
            types.TextContent(
                type="text",
                text="❌ 'llm_provider' is required. One of: 'openai', 'anthropic', 'gemini'.",
            )
        ]
    if not arguments.get("llm_model"):
        return [
            types.TextContent(
                type="text",
                text=(
                    "❌ 'llm_model' is required. Call verify_provider_key(provider, 'llm') first "
                    "to get the live model list, then ask the user to choose one."
                ),
            )
        ]

    llm_api_key = _resolve_llm_key(arguments, llm_provider)
    if not llm_api_key:
        return [
            types.TextContent(
                type="text",
                text=(
                    f"❌ No API key found for LLM provider '{llm_provider}'. "
                    "Pass llm_api_key as an argument, or set the corresponding env var "
                    "(OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY) in your MCP config."
                ),
            )
        ]

    client = _get_client()
    try:
        result = client.pipeline.extract(
            url=url,
            schema=schema,
            llm_provider=llm_provider,
            llm_api_key=llm_api_key,
            llm_model=arguments.get("llm_model"),
            selector=arguments.get("selector"),
            extract_as_list=arguments.get("extract_as_list", False),
            js_render=arguments.get("js_render", False),
            click_selector=arguments.get("click_selector"),
        )
    finally:
        client.close()

    lines = [
        f"✅ Extracted from: {result.url}",
        f"🤖 LLM: {result.llm_provider} / {result.llm_model}",
        f"📋 Fields: {result.field_count}",
    ]
    if result.item_count is not None:
        lines.append(f"📊 Items extracted: {result.item_count}")
    lines.append(
        f"💳 Credits used: ${result.credits_used:.4f} | Remaining: ${result.credits_remaining:.4f}"
    )
    if result.content_warning:
        lines.append(f"⚠️  Content warning: {result.content_warning}")

    lines.append("\n--- Extracted Data ---")
    lines.append(json.dumps(result.extracted, indent=2, ensure_ascii=False))

    return [types.TextContent(type="text", text="\n".join(lines))]


async def _handle_extract_crawl(arguments: dict) -> list[types.TextContent]:
    url = arguments.get("url")
    schema = arguments.get("schema")
    llm_provider = arguments.get("llm_provider")

    if not url:
        return [types.TextContent(type="text", text="❌ 'url' is required.")]
    if not schema:
        return [
            types.TextContent(
                type="text",
                text="❌ 'schema' is required (dict of field_name → description).",
            )
        ]
    if not llm_provider:
        return [
            types.TextContent(
                type="text",
                text="❌ 'llm_provider' is required. One of: 'openai', 'anthropic', 'gemini'.",
            )
        ]
    if not arguments.get("llm_model"):
        return [
            types.TextContent(
                type="text",
                text=(
                    "❌ 'llm_model' is required. Call verify_provider_key(provider, 'llm') first "
                    "to get the live model list, then ask the user to choose one."
                ),
            )
        ]

    llm_api_key = _resolve_llm_key(arguments, llm_provider)
    if not llm_api_key:
        return [
            types.TextContent(
                type="text",
                text=(
                    f"❌ No API key found for LLM provider '{llm_provider}'. "
                    "Pass llm_api_key as an argument, or set the corresponding env var "
                    "(OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY) in your MCP config."
                ),
            )
        ]

    client = _get_client()
    try:
        result = client.pipeline.extract_crawl(
            url=url,
            schema=schema,
            llm_provider=llm_provider,
            llm_api_key=llm_api_key,
            llm_model=arguments.get("llm_model"),
            crawl_mode=arguments.get("crawl_mode", "sitemap"),
            max_pages=arguments.get("max_pages", 5),
            selector=arguments.get("selector"),
            include_pattern=arguments.get("include_pattern"),
            exclude_pattern=arguments.get("exclude_pattern"),
            extract_as_list=arguments.get("extract_as_list", False),
        )
    finally:
        client.close()

    lines = [
        f"✅ Extract crawl complete: {result.root_url}",
        f"📄 Pages extracted: {result.pages_extracted} / {result.pages_attempted} attempted",
        f"🔍 Pages discovered: {result.pages_discovered}",
        f"🤖 LLM: {result.llm_provider} / {result.llm_model}",
        f"📋 Schema fields: {result.field_count}",
        f"💳 Credits used: ${result.credits_used:.4f} | Remaining: ${result.credits_remaining:.4f}",
    ]
    if result.job_id:
        lines.append(f"🆔 Job ID: {result.job_id}")

    lines.append("\n--- Results ---")
    for page in result.results:
        if page.ok:
            lines.append(f"\n✅ {page.url}")
            lines.append(json.dumps(page.extracted, indent=2, ensure_ascii=False))
        else:
            lines.append(f"\n❌ {page.url} — {page.error}")

    return [types.TextContent(type="text", text="\n".join(lines))]


async def _handle_sync_to_vectordb(arguments: dict) -> list[types.TextContent]:
    url = arguments.get("url")
    embedding_provider = arguments.get("embedding_provider")
    vector_db = arguments.get("vector_db")

    if not url:
        return [types.TextContent(type="text", text="❌ 'url' is required.")]
    if not embedding_provider:
        return [
            types.TextContent(
                type="text",
                text="❌ 'embedding_provider' is required. Call list_embedding_providers to see options, then verify_provider_key to get models.",
            )
        ]
    if not vector_db:
        return [
            types.TextContent(
                type="text",
                text="❌ 'vector_db' is required. Call list_vector_db_providers to see options and required config fields.",
            )
        ]
    if not arguments.get("embedding_model") and embedding_provider != "ollama":
        return [
            types.TextContent(
                type="text",
                text=(
                    "❌ 'embedding_model' is required. Call verify_provider_key(embedding_provider, 'embedding') "
                    "to get the live model list, then ask the user to choose one."
                ),
            )
        ]

    embedding_api_key = _resolve_embedding_key(arguments, embedding_provider)
    if embedding_api_key is None:
        return [
            types.TextContent(
                type="text",
                text=(
                    f"❌ No API key found for embedding provider '{embedding_provider}'. "
                    "Pass embedding_api_key as an argument, or set the corresponding env var "
                    "(OPENAI_API_KEY, COHERE_API_KEY, GEMINI_API_KEY, MISTRAL_API_KEY, VOYAGE_API_KEY) "
                    "in your MCP config. For Ollama, pass an empty string."
                ),
            )
        ]

    vector_db_config = _resolve_vector_db_config(arguments, vector_db)
    provider_info = VECTOR_DB_PROVIDERS.get(vector_db, {})
    required_fields = provider_info.get("required_fields", [])
    missing = [f for f in required_fields if not vector_db_config.get(f)]
    if missing:
        return [
            types.TextContent(
                type="text",
                text=(
                    f"❌ Missing required fields for vector DB '{vector_db}': {missing}. "
                    "Call list_vector_db_providers for details on required fields."
                ),
            )
        ]

    contextual_retrieval = arguments.get("contextual_retrieval", False)
    llm_provider = arguments.get("llm_provider")
    llm_api_key = None
    if contextual_retrieval:
        if not arguments.get("llm_model"):
            return [
                types.TextContent(
                    type="text",
                    text=(
                        "❌ llm_model is required when contextual_retrieval=true. "
                        "Call verify_provider_key(llm_provider, 'llm') to get the live model list."
                    ),
                )
            ]
        llm_api_key = _resolve_llm_key(arguments, llm_provider)
        if not llm_api_key:
            return [
                types.TextContent(
                    type="text",
                    text=(
                        "❌ contextual_retrieval=true requires an LLM API key. "
                        "Pass llm_api_key as an argument, or set OPENAI_API_KEY / "
                        "ANTHROPIC_API_KEY / GEMINI_API_KEY in your MCP environment config."
                    ),
                )
            ]

    client = _get_client()
    try:
        result = client.pipeline.sync(
            url=url,
            embedding_provider=embedding_provider,
            embedding_api_key=embedding_api_key,
            embedding_model=arguments.get("embedding_model"),
            embedding_endpoint=arguments.get("embedding_endpoint"),
            vector_db=vector_db,
            vector_db_config=vector_db_config,
            selector=arguments.get("selector"),
            chunk_size=arguments.get("chunk_size", 512),
            overlap=arguments.get("overlap", 50),
            js_render=arguments.get("js_render", False),
            contextual_retrieval=contextual_retrieval,
            llm_provider=llm_provider if contextual_retrieval else None,
            llm_api_key=llm_api_key,
            llm_model=arguments.get("llm_model"),
        )
    finally:
        client.close()

    lines = [
        f"✅ Sync complete: {url}",
        f"📊 Status: {result.status}",
        f"📦 Chunks created: {result.chunks_created}",
        f"🔢 Vectors upserted: {result.vectors_upserted}",
        f"🔤 Total tokens: {result.total_tokens:,}",
        f"🧮 Embedding provider: {result.embedding_provider}",
        f"🗄️  Vector DB: {result.vector_db_provider}",
        f"💳 Credits used: ${result.credits_used:.4f} | Remaining: ${result.credits_remaining:.4f}",
    ]
    if result.contextual_retrieval_used:
        lines.append("🧠 Contextual Retrieval: enabled")
    if result.contextual_retrieval_error:
        lines.append(
            f"⚠️  Contextual Retrieval warning: {result.contextual_retrieval_error}"
        )

    return [types.TextContent(type="text", text="\n".join(lines))]


async def _handle_chunk_file(arguments: dict) -> list[types.TextContent]:
    file_path = arguments.get("file_path")
    if not file_path:
        return [
            types.TextContent(
                type="text",
                text="❌ 'file_path' is required. Provide the absolute path to the local file.",
            )
        ]

    contextual_retrieval = arguments.get("contextual_retrieval", False)
    llm_provider = arguments.get("llm_provider")
    llm_api_key = None
    if contextual_retrieval:
        llm_api_key = _resolve_llm_key(arguments, llm_provider)
        if not llm_api_key:
            return [
                types.TextContent(
                    type="text",
                    text=(
                        "❌ contextual_retrieval=true requires an LLM API key. "
                        "Pass llm_api_key as an argument, or set OPENAI_API_KEY / "
                        "ANTHROPIC_API_KEY / GEMINI_API_KEY in your MCP environment config."
                    ),
                )
            ]
        if not arguments.get("llm_model"):
            return [
                types.TextContent(
                    type="text",
                    text=(
                        "❌ llm_model is required when contextual_retrieval=true. "
                        "Call verify_provider_key first to get the available model list, "
                        "then ask the user to choose one."
                    ),
                )
            ]

    client = _get_client()
    try:
        result = client.pipeline.chunk_file(
            file_path=file_path,
            chunk_size=arguments.get("chunk_size", 512),
            overlap=arguments.get("overlap", 50),
            contextual_retrieval=contextual_retrieval,
            llm_provider=llm_provider if contextual_retrieval else None,
            llm_api_key=llm_api_key,
            llm_model=arguments.get("llm_model"),
        )
    finally:
        client.close()

    lines = [
        f"✅ Chunked file: {result.source}",
        f"📦 Chunks: {result.total_chunks}",
        f"💳 Credits used: ${result.credits_used:.4f} | Remaining: ${result.credits_remaining:.4f}",
    ]
    if result.contextual_retrieval_used:
        lines.append("🧠 Contextual Retrieval: enabled")
    if result.contextual_retrieval_error:
        lines.append(
            f"⚠️  Contextual Retrieval warning: {result.contextual_retrieval_error}"
        )

    lines.append("\n--- Chunks ---")
    for i, chunk in enumerate(result.chunks, 1):
        preview = chunk.content[:300].replace("\n", " ")
        lines.append(
            f"\n[Chunk {i} | ~{chunk.token_estimate} tokens]\n{preview}{'...' if len(chunk.content) > 300 else ''}"
        )

    return [types.TextContent(type="text", text="\n".join(lines))]


async def _handle_ingest_file(arguments: dict) -> list[types.TextContent]:
    file_path = arguments.get("file_path")
    embedding_provider = arguments.get("embedding_provider")
    vector_db = arguments.get("vector_db")

    if not file_path:
        return [
            types.TextContent(
                type="text",
                text="❌ 'file_path' is required. Provide the absolute path to the local file.",
            )
        ]
    if not embedding_provider:
        return [
            types.TextContent(
                type="text",
                text="❌ 'embedding_provider' is required. Call list_embedding_providers to see options, then verify_provider_key to get models.",
            )
        ]
    if not vector_db:
        return [
            types.TextContent(
                type="text",
                text="❌ 'vector_db' is required. Call list_vector_db_providers to see options and required config fields.",
            )
        ]
    if not arguments.get("embedding_model") and embedding_provider != "ollama":
        return [
            types.TextContent(
                type="text",
                text=(
                    "❌ 'embedding_model' is required. Call verify_provider_key(embedding_provider, 'embedding') "
                    "to get the live model list, then ask the user to choose one."
                ),
            )
        ]

    embedding_api_key = _resolve_embedding_key(arguments, embedding_provider)
    if embedding_api_key is None:
        return [
            types.TextContent(
                type="text",
                text=(
                    f"❌ No API key found for embedding provider '{embedding_provider}'. "
                    "Pass embedding_api_key as an argument, or set the corresponding env var "
                    "(OPENAI_API_KEY, COHERE_API_KEY, GEMINI_API_KEY, MISTRAL_API_KEY, VOYAGE_API_KEY) "
                    "in your MCP config. For Ollama, pass an empty string."
                ),
            )
        ]

    vector_db_config = _resolve_vector_db_config(arguments, vector_db)
    provider_info = VECTOR_DB_PROVIDERS.get(vector_db, {})
    required_fields = provider_info.get("required_fields", [])
    missing = [f for f in required_fields if not vector_db_config.get(f)]
    if missing:
        return [
            types.TextContent(
                type="text",
                text=(
                    f"❌ Missing required fields for vector DB '{vector_db}': {missing}. "
                    "Call list_vector_db_providers for details on required fields."
                ),
            )
        ]

    contextual_retrieval = arguments.get("contextual_retrieval", False)
    llm_provider = arguments.get("llm_provider")
    llm_api_key = None
    if contextual_retrieval:
        if not arguments.get("llm_model"):
            return [
                types.TextContent(
                    type="text",
                    text=(
                        "❌ llm_model is required when contextual_retrieval=true. "
                        "Call verify_provider_key(llm_provider, 'llm') to get the live model list."
                    ),
                )
            ]
        llm_api_key = _resolve_llm_key(arguments, llm_provider)
        if not llm_api_key:
            return [
                types.TextContent(
                    type="text",
                    text=(
                        "❌ contextual_retrieval=true requires an LLM API key. "
                        "Pass llm_api_key as an argument, or set OPENAI_API_KEY / "
                        "ANTHROPIC_API_KEY / GEMINI_API_KEY in your MCP environment config."
                    ),
                )
            ]

    client = _get_client()
    try:
        result = client.pipeline.ingest(
            file_path=file_path,
            embedding_provider=embedding_provider,
            embedding_api_key=embedding_api_key,
            embedding_model=arguments.get("embedding_model"),
            embedding_endpoint=arguments.get("embedding_endpoint"),
            vector_db=vector_db,
            vector_db_config=vector_db_config,
            chunk_size=arguments.get("chunk_size", 512),
            overlap=arguments.get("overlap", 50),
            contextual_retrieval=contextual_retrieval,
            llm_provider=llm_provider if contextual_retrieval else None,
            llm_api_key=llm_api_key,
            llm_model=arguments.get("llm_model"),
        )
    finally:
        client.close()

    lines = [
        f"✅ Ingest complete: {result.filename}",
        f"📊 Status: {result.status}",
        f"📦 Chunks created: {result.chunks_created}",
        f"🔢 Vectors upserted: {result.vectors_upserted}",
        f"🔤 Total tokens: {result.total_tokens:,}",
        f"🧮 Embedding provider: {result.embedding_provider}",
        f"🗄️  Vector DB: {result.vector_db_provider}",
        f"💳 Credits used: ${result.credits_used:.4f} | Remaining: ${result.credits_remaining:.4f}",
    ]
    if result.contextual_retrieval_used:
        lines.append("🧠 Contextual Retrieval: enabled")
    if result.contextual_retrieval_error:
        lines.append(
            f"⚠️  Contextual Retrieval warning: {result.contextual_retrieval_error}"
        )

    return [types.TextContent(type="text", text="\n".join(lines))]


async def _handle_autorag(arguments: dict) -> list[types.TextContent]:
    url = arguments.get("url")
    embedding_provider = arguments.get("embedding_provider")
    vector_db = arguments.get("vector_db")

    if not url:
        return [types.TextContent(type="text", text="❌ 'url' is required.")]
    if not embedding_provider:
        return [
            types.TextContent(
                type="text",
                text="❌ 'embedding_provider' is required. Call list_embedding_providers to see options, then verify_provider_key to get models.",
            )
        ]
    if not vector_db:
        return [
            types.TextContent(
                type="text",
                text="❌ 'vector_db' is required. Call list_vector_db_providers to see options and required config fields.",
            )
        ]
    if not arguments.get("embedding_model") and embedding_provider != "ollama":
        return [
            types.TextContent(
                type="text",
                text=(
                    "❌ 'embedding_model' is required. Call verify_provider_key(embedding_provider, 'embedding') "
                    "to get the live model list, then ask the user to choose one."
                ),
            )
        ]

    embedding_api_key = _resolve_embedding_key(arguments, embedding_provider)
    if embedding_api_key is None:
        return [
            types.TextContent(
                type="text",
                text=(
                    f"❌ No API key found for embedding provider '{embedding_provider}'. "
                    "Pass embedding_api_key as an argument, or set the corresponding env var "
                    "(OPENAI_API_KEY, COHERE_API_KEY, GEMINI_API_KEY, MISTRAL_API_KEY, VOYAGE_API_KEY) "
                    "in your MCP config. For Ollama, pass an empty string."
                ),
            )
        ]

    vector_db_config = _resolve_vector_db_config(arguments, vector_db)
    provider_info = VECTOR_DB_PROVIDERS.get(vector_db, {})
    required_fields = provider_info.get("required_fields", [])
    missing = [f for f in required_fields if not vector_db_config.get(f)]
    if missing:
        return [
            types.TextContent(
                type="text",
                text=(
                    f"❌ Missing required fields for vector DB '{vector_db}': {missing}. "
                    "Call list_vector_db_providers for details on required fields."
                ),
            )
        ]

    contextual_retrieval = arguments.get("contextual_retrieval", False)
    llm_provider = arguments.get("llm_provider")
    llm_api_key = None
    if contextual_retrieval:
        if not arguments.get("llm_model"):
            return [
                types.TextContent(
                    type="text",
                    text=(
                        "❌ llm_model is required when contextual_retrieval=true. "
                        "Call verify_provider_key(llm_provider, 'llm') to get the live model list."
                    ),
                )
            ]
        llm_api_key = _resolve_llm_key(arguments, llm_provider)
        if not llm_api_key:
            return [
                types.TextContent(
                    type="text",
                    text=(
                        "❌ contextual_retrieval=true requires an LLM API key. "
                        "Pass llm_api_key as an argument, or set OPENAI_API_KEY / "
                        "ANTHROPIC_API_KEY / GEMINI_API_KEY in your MCP environment config."
                    ),
                )
            ]

    client = _get_client()
    try:
        result = client.pipeline.autorag(
            url=url,
            embedding_provider=embedding_provider,
            embedding_api_key=embedding_api_key,
            embedding_model=arguments.get("embedding_model"),
            vector_db=vector_db,
            vector_db_config=vector_db_config,
            crawl_mode=arguments.get("crawl_mode", "sitemap"),
            max_pages=arguments.get("max_pages", 5),
            include_pattern=arguments.get("include_pattern"),
            exclude_pattern=arguments.get("exclude_pattern"),
            selector=arguments.get("selector"),
            chunk_size=arguments.get("chunk_size", 512),
            overlap=arguments.get("overlap", 50),
            contextual_retrieval=contextual_retrieval,
            llm_provider=llm_provider if contextual_retrieval else None,
            llm_api_key=llm_api_key,
            llm_model=arguments.get("llm_model"),
        )
    finally:
        client.close()

    lines = [
        f"✅ AutoRAG complete: {result.root_url}",
        f"🔍 Pages discovered: {result.pages_discovered}",
        f"📄 Pages crawled: {result.pages_crawled} | Failed: {result.pages_failed}",
        f"📦 Total chunks: {result.total_chunks}",
        f"🔢 Vectors upserted: {result.vectors_upserted}",
        f"🔤 Total tokens: {result.total_tokens:,}",
        f"🧮 Embedding: {result.embedding_provider} / {result.embedding_model}",
        f"🗄️  Vector DB: {result.vector_db_provider}",
        f"💳 Credits used: ${result.credits_used:.4f} | Remaining: ${result.credits_remaining:.4f}",
    ]
    if result.contextual_retrieval_used:
        lines.append("🧠 Contextual Retrieval: enabled")
    if result.contextual_retrieval_error:
        lines.append(
            f"⚠️  Contextual Retrieval warning: {result.contextual_retrieval_error}"
        )

    return [types.TextContent(type="text", text="\n".join(lines))]


async def _handle_inspect_vectordb(arguments: dict) -> list[types.TextContent]:
    vector_db = arguments.get("vector_db")
    if not vector_db:
        return [
            types.TextContent(
                type="text",
                text="❌ 'vector_db' is required. Call list_vector_db_providers to see options.",
            )
        ]

    vector_db_config = _resolve_vector_db_config(arguments, vector_db)
    provider_info = VECTOR_DB_PROVIDERS.get(vector_db, {})
    required_fields = provider_info.get("required_fields", [])
    missing = [f for f in required_fields if not vector_db_config.get(f)]
    if missing:
        return [
            types.TextContent(
                type="text",
                text=(
                    f"❌ Missing required fields for vector DB '{vector_db}': {missing}. "
                    "Call list_vector_db_providers for details on required fields."
                ),
            )
        ]

    client = _get_client()
    try:
        result = client.pipeline.inspect_vectordb(
            vector_db=vector_db,
            vector_db_config=vector_db_config,
        )
    finally:
        client.close()

    lines = [
        f"✅ Inspected vector DB: {result.provider}",
        f"📐 Dimension: {result.dimension}"
        + (" (unknown)" if not result.dimension_known else ""),
        f"🔢 Total vectors: {result.total_vector_count:,}",
    ]
    if result.namespace:
        lines.append(
            f"📁 Namespace: {result.namespace} ({result.namespace_vector_count:,} vectors)"
        )
    if result.suggested_models:
        lines.append("\n🤖 Suggested embedding models (based on dimension):")
        for m in result.suggested_models:
            lines.append(f"  • {m.label} ({m.provider} / {m.model})")
    if result.note:
        lines.append(f"\n💡 {result.note}")
    lines.append(
        "\n⚠️ Use the same embedding model that was used during ingestion. "
        "Using a different model will produce meaningless results."
    )

    return [types.TextContent(type="text", text="\n".join(lines))]


async def _handle_query_vectordb(arguments: dict) -> list[types.TextContent]:
    query = arguments.get("query")
    embedding_provider = arguments.get("embedding_provider")
    vector_db = arguments.get("vector_db")

    if not query:
        return [types.TextContent(type="text", text="❌ 'query' is required.")]
    if not embedding_provider:
        return [
            types.TextContent(
                type="text",
                text="❌ 'embedding_provider' is required. Call list_embedding_providers to see options.",
            )
        ]
    if not arguments.get("embedding_model"):
        return [
            types.TextContent(
                type="text",
                text=(
                    "❌ 'embedding_model' is required. Use inspect_vectordb first to confirm "
                    "which model was used during ingestion, then pass that exact model name."
                ),
            )
        ]
    if not vector_db:
        return [
            types.TextContent(
                type="text",
                text="❌ 'vector_db' is required. Call list_vector_db_providers to see options.",
            )
        ]

    embedding_api_key = _resolve_embedding_key(arguments, embedding_provider)
    if embedding_api_key is None:
        return [
            types.TextContent(
                type="text",
                text=(
                    f"❌ No API key found for embedding provider '{embedding_provider}'. "
                    "Pass embedding_api_key as an argument, or set the corresponding env var."
                ),
            )
        ]

    vector_db_config = _resolve_vector_db_config(arguments, vector_db)
    provider_info = VECTOR_DB_PROVIDERS.get(vector_db, {})
    required_fields = provider_info.get("required_fields", [])
    missing = [f for f in required_fields if not vector_db_config.get(f)]
    if missing:
        return [
            types.TextContent(
                type="text",
                text=(
                    f"❌ Missing required fields for vector DB '{vector_db}': {missing}. "
                    "Call list_vector_db_providers for details."
                ),
            )
        ]

    client = _get_client()
    try:
        result = client.pipeline.query_vectordb(
            query=query,
            embedding_provider=embedding_provider,
            embedding_api_key=embedding_api_key,
            embedding_model=arguments.get("embedding_model"),
            vector_db=vector_db,
            vector_db_config=vector_db_config,
            top_k=arguments.get("top_k", 5),
        )
    finally:
        client.close()

    lines = [
        f"✅ Query complete",
        f"🔍 Query: {result.query}",
        f"🧮 Embedding: {result.embedding_provider} / {result.embedding_model}",
        f"🗄️  Vector DB: {result.vector_db_provider}",
        f"📦 Chunks retrieved: {result.chunks_retrieved} / {result.top_k_requested} requested",
        f"💳 Credits used: ${result.credits_used:.4f} | Remaining: ${result.credits_remaining:.4f}",
    ]

    if result.results:
        lines.append("\n--- Results ---")
        for i, r in enumerate(result.results, 1):
            preview = r.text[:300].replace("\n", " ")
            lines.append(f"\n[{i}] Score: {r.score:.3f}")
            lines.append(preview + ("..." if len(r.text) > 300 else ""))
            if r.metadata:
                lines.append(f"    Metadata: {json.dumps(r.metadata)}")
    else:
        lines.append(
            "\n⚠️ No results found. Check that your embedding model matches the one used during ingestion."
        )

    return [types.TextContent(type="text", text="\n".join(lines))]


async def _handle_rag_chat(arguments: dict) -> list[types.TextContent]:
    query = arguments.get("query")
    embedding_provider = arguments.get("embedding_provider")
    vector_db = arguments.get("vector_db")
    llm_provider = arguments.get("llm_provider")

    if not query:
        return [types.TextContent(type="text", text="❌ 'query' is required.")]
    if not embedding_provider:
        return [
            types.TextContent(
                type="text",
                text="❌ 'embedding_provider' is required.",
            )
        ]
    if not arguments.get("embedding_model"):
        return [
            types.TextContent(
                type="text",
                text=(
                    "❌ 'embedding_model' is required. Use inspect_vectordb first to confirm "
                    "which model was used during ingestion."
                ),
            )
        ]
    if not vector_db:
        return [types.TextContent(type="text", text="❌ 'vector_db' is required.")]
    if not llm_provider:
        return [
            types.TextContent(
                type="text",
                text="❌ 'llm_provider' is required. One of: 'openai', 'anthropic', 'gemini'.",
            )
        ]
    if not arguments.get("llm_model"):
        return [
            types.TextContent(
                type="text",
                text=(
                    "❌ 'llm_model' is required. Call verify_provider_key(llm_provider, 'llm') "
                    "to get the live model list, then ask the user to choose one."
                ),
            )
        ]

    embedding_api_key = _resolve_embedding_key(arguments, embedding_provider)
    if embedding_api_key is None:
        return [
            types.TextContent(
                type="text",
                text=f"❌ No API key found for embedding provider '{embedding_provider}'.",
            )
        ]

    llm_api_key = _resolve_llm_key(arguments, llm_provider)
    if not llm_api_key:
        return [
            types.TextContent(
                type="text",
                text=(
                    f"❌ No API key found for LLM provider '{llm_provider}'. "
                    "Pass llm_api_key as an argument, or set OPENAI_API_KEY / "
                    "ANTHROPIC_API_KEY / GEMINI_API_KEY in your MCP environment config."
                ),
            )
        ]

    vector_db_config = _resolve_vector_db_config(arguments, vector_db)
    provider_info = VECTOR_DB_PROVIDERS.get(vector_db, {})
    required_fields = provider_info.get("required_fields", [])
    missing = [f for f in required_fields if not vector_db_config.get(f)]
    if missing:
        return [
            types.TextContent(
                type="text",
                text=f"❌ Missing required fields for vector DB '{vector_db}': {missing}.",
            )
        ]

    client = _get_client()
    try:
        result = client.pipeline.rag_chat(
            query=query,
            embedding_provider=embedding_provider,
            embedding_api_key=embedding_api_key,
            embedding_model=arguments.get("embedding_model"),
            vector_db=vector_db,
            vector_db_config=vector_db_config,
            llm_provider=llm_provider,
            llm_api_key=llm_api_key,
            llm_model=arguments.get("llm_model"),
            top_k=arguments.get("top_k", 5),
        )
    finally:
        client.close()

    lines = [
        f"💬 RAG Chat Answer",
        f"🔍 Query: {result.query}",
        f"🧮 Embedding: {result.embedding_provider} / {result.embedding_model}",
        f"🤖 LLM: {result.llm_provider} / {result.llm_model}",
        f"📦 Chunks retrieved: {result.chunks_retrieved}",
        f"💳 Credits used: ${result.credits_used:.4f} | Remaining: ${result.credits_remaining:.4f}",
    ]
    if result.llm_error:
        lines.append(f"⚠️  LLM error: {result.llm_error}")

    lines.append("\n--- Answer ---")
    lines.append(result.answer)

    if result.sources:
        lines.append("\n--- Sources ---")
        for i, s in enumerate(result.sources, 1):
            preview = s.text[:200].replace("\n", " ")
            lines.append(f"\n[{i}] Score: {s.score:.3f}")
            lines.append(preview + ("..." if len(s.text) > 200 else ""))

    return [types.TextContent(type="text", text="\n".join(lines))]


async def _handle_ingest_folder(arguments: dict) -> list[types.TextContent]:
    folder_path = arguments.get("folder_path")
    embedding_provider = arguments.get("embedding_provider")
    vector_db = arguments.get("vector_db")

    if not folder_path:
        return [
            types.TextContent(
                type="text",
                text="❌ 'folder_path' is required. Provide the absolute path to the folder.",
            )
        ]
    if not embedding_provider:
        return [
            types.TextContent(
                type="text",
                text="❌ 'embedding_provider' is required. Call list_embedding_providers to see options, then verify_provider_key to get models.",
            )
        ]
    if not vector_db:
        return [
            types.TextContent(
                type="text",
                text="❌ 'vector_db' is required. Call list_vector_db_providers to see options and required config fields.",
            )
        ]
    if not arguments.get("embedding_model") and embedding_provider != "ollama":
        return [
            types.TextContent(
                type="text",
                text=(
                    "❌ 'embedding_model' is required. Call verify_provider_key(embedding_provider, 'embedding') "
                    "to get the live model list, then ask the user to choose one."
                ),
            )
        ]

    embedding_api_key = _resolve_embedding_key(arguments, embedding_provider)
    if embedding_api_key is None:
        return [
            types.TextContent(
                type="text",
                text=(
                    f"❌ No API key found for embedding provider '{embedding_provider}'. "
                    "Pass embedding_api_key as an argument, or set the corresponding env var "
                    "(OPENAI_API_KEY, COHERE_API_KEY, GEMINI_API_KEY, MISTRAL_API_KEY, VOYAGE_API_KEY) "
                    "in your MCP config. For Ollama, pass an empty string."
                ),
            )
        ]

    vector_db_config = _resolve_vector_db_config(arguments, vector_db)
    provider_info = VECTOR_DB_PROVIDERS.get(vector_db, {})
    required_fields = provider_info.get("required_fields", [])
    missing = [f for f in required_fields if not vector_db_config.get(f)]
    if missing:
        return [
            types.TextContent(
                type="text",
                text=(
                    f"❌ Missing required fields for vector DB '{vector_db}': {missing}. "
                    "Call list_vector_db_providers for details on required fields."
                ),
            )
        ]

    # Build optional params
    file_extensions = arguments.get("file_extensions") or None
    json_text_keys = arguments.get("json_text_keys") or None

    client = _get_client()
    try:
        result = client.pipeline.ingest_scraped(
            folder_path=folder_path,
            embedding_provider=embedding_provider,
            embedding_api_key=embedding_api_key,
            embedding_model=arguments.get("embedding_model"),
            embedding_endpoint=arguments.get("embedding_endpoint"),
            vector_db=vector_db,
            vector_db_config=vector_db_config,
            chunk_size=arguments.get("chunk_size", 512),
            overlap=arguments.get("overlap", 50),
            file_extensions=file_extensions,
            recursive=arguments.get("recursive", True),
            max_files=arguments.get("max_files"),
            batch_delay=arguments.get("batch_delay", 0.5),
            json_text_keys=json_text_keys,
        )
    finally:
        client.close()

    lines = [
        f"✅ Folder ingest complete: {folder_path}",
        f"📁 Files processed: {result.files_processed}",
        f"❌ Files failed: {result.files_failed}",
        f"📦 Total chunks: {result.total_chunks}",
        f"🔢 Vectors upserted: {result.vectors_upserted}",
        f"🧮 Embedding: {result.embedding_provider}",
        f"🗄️  Vector DB: {result.vector_db_provider}",
        f"💳 Credits used: ${result.credits_used:.4f} | Remaining: ${result.credits_remaining:.4f}",
    ]

    if result.errors:
        lines.append(f"\n⚠️  {len(result.errors)} file(s) failed:")
        for err in result.errors[:10]:
            lines.append(f"  • {err['file']}: {err['error']}")
        if len(result.errors) > 10:
            lines.append(f"  ... and {len(result.errors) - 10} more errors.")

    return [types.TextContent(type="text", text="\n".join(lines))]


def _handle_list_embedding_providers() -> list[types.TextContent]:
    lines = ["## Supported Embedding Providers\n"]
    for key, info in EMBEDDING_PROVIDERS.items():
        lines.append(f"### `{key}` — {info['label']}")
        lines.append(
            f"- Requires API key: {'Yes' if info['requires_api_key'] else 'No (local)'}"
        )
        lines.append(f"- Local: {'Yes' if info.get('local') else 'No'}")
        lines.append(f"- Notes: {info['notes']}")
        lines.append("")
    lines.append(
        "After choosing a provider, call verify_provider_key(provider, 'embedding') to get the live model list."
    )
    return [types.TextContent(type="text", text="\n".join(lines))]


def _handle_list_vector_db_providers() -> list[types.TextContent]:
    lines = ["## Supported Vector Database Providers\n"]
    for key, info in VECTOR_DB_PROVIDERS.items():
        lines.append(f"### `{key}` — {info['label']}")
        lines.append(f"- Required fields: {info['required_fields']}")
        lines.append(f"- Optional fields: {info.get('optional_fields', [])}")
        lines.append(f"- Local: {'Yes' if info.get('local') else 'No'}")
        lines.append(f"- Notes: {info['notes']}")
        lines.append("")
    return [types.TextContent(type="text", text="\n".join(lines))]


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    """Main entry point — runs the MCP server over stdio."""

    async def _run() -> None:
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    asyncio.run(_run())


if __name__ == "__main__":
    main()
