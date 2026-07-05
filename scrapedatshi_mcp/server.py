"""
scrapedatshi_mcp.server
~~~~~~~~~~~~~~~~~~~~~~~
MCP server exposing scrapedatshi pipeline tools to Claude Desktop and any
MCP-compatible AI client.

Tools exposed:
    scrape_url              — Scrape & chunk a single URL
    crawl_site              — Crawl a whole site (sitemap or spider mode)
    extract_data            — Extract structured schema from a URL using your LLM
    extract_crawl           — Multi-page schema extraction via site crawl
    sync_to_vectordb        — Full pipeline: scrape → embed → inject into vector DB
    list_embedding_providers — Discover supported embedding providers + required fields
    list_vector_db_providers — Discover supported vector DBs + required fields

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

import json
import os
import sys
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

# ── Key resolution helpers ────────────────────────────────────────────────────


def _resolve_scrapedatshi_key() -> str | None:
    """Resolve the scrapedatshi API key from environment."""
    return os.environ.get("SCRAPEDATSHI_API_KEY")


def _resolve_llm_key(arguments: dict, provider: str | None = None) -> str | None:
    """
    Resolve an LLM API key using the fallback chain:
      1. Explicit argument
      2. Provider-specific env var
      3. Generic fallback env vars
    """
    explicit = arguments.get("llm_api_key")
    if explicit:
        return explicit

    # Provider-specific env vars
    provider_env_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    if provider and provider in provider_env_map:
        val = os.environ.get(provider_env_map[provider])
        if val:
            return val

    # Generic fallbacks (try all if provider unknown)
    for env_var in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]:
        val = os.environ.get(env_var)
        if val:
            return val

    return None


def _resolve_embedding_key(arguments: dict, provider: str | None = None) -> str | None:
    """
    Resolve an embedding API key using the fallback chain.
    Returns empty string for Ollama (no key required).
    """
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

    # Generic fallbacks
    for env_var in ["OPENAI_API_KEY", "COHERE_API_KEY", "GEMINI_API_KEY"]:
        val = os.environ.get(env_var)
        if val:
            return val

    return None


def _resolve_vector_db_config(arguments: dict, vector_db: str) -> dict:
    """
    Resolve vector DB config, injecting API keys from env vars where missing.
    The user-provided vector_db_config dict is merged with env-var fallbacks.
    """
    config: dict = {}

    # Parse user-provided config (may be a JSON string or already a dict)
    raw_config = arguments.get("vector_db_config", {})
    if isinstance(raw_config, str):
        try:
            config = json.loads(raw_config)
        except json.JSONDecodeError:
            config = {}
    elif isinstance(raw_config, dict):
        config = dict(raw_config)

    # Inject env-var fallbacks for API keys
    if vector_db == "pinecone":
        if not config.get("api_key"):
            env_key = os.environ.get("PINECONE_API_KEY")
            if env_key:
                config["api_key"] = env_key

    elif vector_db == "qdrant":
        if not config.get("api_key"):
            env_key = os.environ.get("QDRANT_API_KEY")
            if env_key:
                config["api_key"] = env_key

    elif vector_db == "weaviate":
        if not config.get("api_key"):
            env_key = os.environ.get("WEAVIATE_API_KEY")
            if env_key:
                config["api_key"] = env_key

    return config


def _get_client() -> ScrapedatshiClient:
    """Create a ScrapedatshiClient using the resolved API key."""
    api_key = _resolve_scrapedatshi_key()
    if not api_key:
        raise AuthError(
            "No scrapedatshi API key found. Set SCRAPEDATSHI_API_KEY in your MCP "
            "environment config or pass it explicitly."
        )
    return ScrapedatshiClient(api_key=api_key)


def _format_error(exc: Exception) -> str:
    """Format a scrapedatshi exception into a readable error string for Claude."""
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
        return f"❌ Rate limit exceeded: {exc}\nPlease wait a moment and try again."
    if isinstance(exc, ServerBusyError):
        retry = getattr(exc, "retry_after", None)
        wait_msg = f" Retry after {retry} seconds." if retry else ""
        return f"❌ Server temporarily at capacity: {exc}.{wait_msg}"
    if isinstance(exc, ScrapedatshiError):
        return f"❌ scrapedatshi API error: {exc}"
    return f"❌ Unexpected error: {exc}"


# ── Tool definitions ──────────────────────────────────────────────────────────


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        # ── scrape_url ────────────────────────────────────────────────────────
        types.Tool(
            name="scrape_url",
            description=(
                "Scrape a single web URL, chunk its content into RAG-ready text segments, "
                "and return the structured chunks as JSON. No embedding or vector DB required — "
                "this is the fastest and cheapest operation. Use this when the user wants to "
                "read, summarize, or process the content of a specific web page.\n\n"
                "Supports optional CSS selectors to target specific page sections, JavaScript "
                "rendering for SPAs, and Contextual Retrieval (RAG 2.0) for enriched chunks.\n\n"
                "LLM keys (llm_api_key) can be omitted if OPENAI_API_KEY, ANTHROPIC_API_KEY, "
                "or GEMINI_API_KEY is set in the MCP environment config."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The web URL to scrape and chunk (e.g. 'https://docs.example.com/intro').",
                    },
                    "selector": {
                        "type": "string",
                        "description": (
                            "Optional CSS selector to target a specific element on the page "
                            "(e.g. 'article', '.content', 'main'). Omit to scrape the full page."
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
                    "js_render": {
                        "type": "boolean",
                        "description": (
                            "If true, uses a headless Chromium browser to render JavaScript before "
                            "scraping. Required for SPAs and JS-heavy pages. Adds a small surcharge."
                        ),
                        "default": False,
                    },
                    "contextual_retrieval": {
                        "type": "boolean",
                        "description": (
                            "Enable RAG 2.0 contextual enrichment. An LLM generates a unique context "
                            "string for each chunk, boosting retrieval accuracy by 35–50%. "
                            "Requires llm_provider, llm_api_key, and llm_model."
                        ),
                        "default": False,
                    },
                    "llm_provider": {
                        "type": "string",
                        "description": "LLM provider for contextual retrieval. One of: 'openai', 'anthropic', 'gemini'.",
                        "enum": ["openai", "anthropic", "gemini"],
                    },
                    "llm_api_key": {
                        "type": "string",
                        "description": (
                            "API key for the LLM provider. Can be omitted if the corresponding "
                            "env var is set (OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY)."
                        ),
                    },
                    "llm_model": {
                        "type": "string",
                        "description": "LLM model name (e.g. 'gpt-4o-mini', 'claude-3-haiku-20240307', 'gemini-1.5-flash').",
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
                "(follows links — works on any site). Returns chunks from all crawled pages combined.\n\n"
                "⚠️ IMPORTANT: This tool processes multiple pages sequentially. Always confirm the "
                "max_pages limit with the user before calling. Default is 10 pages. For large-scale "
                "crawls, explicitly prompt the user to define a target limit to avoid unexpected "
                "credit usage. Maximum allowed: 200 pages.\n\n"
                "LLM keys can be omitted if set as environment variables in the MCP config."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The root domain or sitemap URL to crawl (e.g. 'https://docs.example.com').",
                    },
                    "max_pages": {
                        "type": "integer",
                        "description": (
                            "Maximum number of pages to crawl. Default: 10. Maximum: 200. "
                            "Always confirm this with the user for large sites."
                        ),
                        "default": 10,
                        "minimum": 1,
                        "maximum": 200,
                    },
                    "crawl_mode": {
                        "type": "string",
                        "description": (
                            "'sitemap' (default): reads sitemap.xml to discover URLs — best for "
                            "documentation sites and blogs. 'spider': follows <a href> links from "
                            "the root URL — works on any site, no sitemap required."
                        ),
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
                        "description": "Use headless browser to render JS before scraping each page. Adds surcharge per page.",
                        "default": False,
                    },
                    "contextual_retrieval": {
                        "type": "boolean",
                        "description": "Enable RAG 2.0 contextual enrichment for all chunks. Requires llm_provider and llm_api_key.",
                        "default": False,
                    },
                    "llm_provider": {
                        "type": "string",
                        "description": "LLM provider for contextual retrieval. One of: 'openai', 'anthropic', 'gemini'.",
                        "enum": ["openai", "anthropic", "gemini"],
                    },
                    "llm_api_key": {
                        "type": "string",
                        "description": "API key for the LLM provider. Can be omitted if set as an env var.",
                    },
                    "llm_model": {
                        "type": "string",
                        "description": "LLM model name for contextual retrieval.",
                    },
                },
                "required": ["url"],
            },
        ),
        # ── extract_data ──────────────────────────────────────────────────────
        types.Tool(
            name="extract_data",
            description=(
                "Scrape a URL and extract structured data matching a user-defined schema using "
                "an LLM. The user brings their own LLM key — scrapedatshi handles the scraping "
                "and orchestration. Returns a JSON object (or array if extract_as_list=true) "
                "matching the schema fields.\n\n"
                "Use this when the user wants to pull specific fields from a web page "
                "(e.g. product title, price, description from an e-commerce page; article "
                "author, date, summary from a news page).\n\n"
                "LLM keys can be omitted if OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY "
                "is set in the MCP environment config."
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
                            "Dict mapping field names to description strings. The LLM uses these "
                            "descriptions to understand what to extract. "
                            'Example: {"title": "string — the product name", "price": "number — price in USD", "in_stock": "boolean — whether in stock"}'
                        ),
                        "additionalProperties": {"type": "string"},
                    },
                    "llm_provider": {
                        "type": "string",
                        "description": "LLM provider to use. One of: 'openai', 'anthropic', 'gemini'.",
                        "enum": ["openai", "anthropic", "gemini"],
                    },
                    "llm_api_key": {
                        "type": "string",
                        "description": (
                            "API key for the LLM provider. Can be omitted if the corresponding "
                            "env var is set (OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY)."
                        ),
                    },
                    "llm_model": {
                        "type": "string",
                        "description": (
                            "Optional model override. Defaults: openai→gpt-4o-mini, "
                            "anthropic→claude-3-haiku-20240307, gemini→gemini-1.5-flash. "
                            "Use an advanced model (gpt-4o, claude-3-5-sonnet, gemini-1.5-pro) "
                            "for long-form pages like documentation or legal docs."
                        ),
                    },
                    "selector": {
                        "type": "string",
                        "description": "Optional CSS selector to target a specific section before extraction.",
                    },
                    "extract_as_list": {
                        "type": "boolean",
                        "description": (
                            "If true, extracts ALL matching items on the page as a JSON array. "
                            "Use for listing pages (product catalogues, article feeds, search results)."
                        ),
                        "default": False,
                    },
                    "js_render": {
                        "type": "boolean",
                        "description": "Use headless browser to render JS before extracting. Required for SPAs.",
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
                "Combines site crawling with schema extraction in a single call. Each page is "
                "processed independently — failed pages return an error without aborting the batch. "
                "Only successfully extracted pages are billed.\n\n"
                "⚠️ IMPORTANT: Each page takes 5–15 seconds to process. Default is 5 pages. "
                "For more than 20 pages, warn the user about potential wait times and credit usage "
                "before proceeding. Always confirm the max_pages limit with the user for large sites. "
                "Maximum: 50 pages per call.\n\n"
                "LLM keys can be omitted if set as environment variables in the MCP config."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The root domain to crawl (e.g. 'https://example.com/products').",
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
                        "description": "LLM provider to use. One of: 'openai', 'anthropic', 'gemini'.",
                        "enum": ["openai", "anthropic", "gemini"],
                    },
                    "llm_api_key": {
                        "type": "string",
                        "description": "API key for the LLM provider. Can be omitted if set as an env var.",
                    },
                    "llm_model": {
                        "type": "string",
                        "description": (
                            "Optional model override. Standard models (mini/flash/haiku) use 8k char context. "
                            "Advanced models use 30k char context — better for long pages."
                        ),
                    },
                    "crawl_mode": {
                        "type": "string",
                        "description": "'sitemap' (default): reads sitemap.xml. 'spider': follows links from root URL.",
                        "enum": ["sitemap", "spider"],
                        "default": "sitemap",
                    },
                    "max_pages": {
                        "type": "integer",
                        "description": (
                            "Maximum pages to crawl and extract from. Default: 5. Maximum: 50. "
                            "Sitemap mode supports up to 100 pages; spider mode up to 25 pages. "
                            "Always confirm with the user before setting above 20."
                        ),
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
                "Full RAG pipeline: scrape a URL, embed the chunks using your embedding provider, "
                "and inject the vectors into your vector database — all in a single call. "
                "Use this when the user wants to add web content to their vector DB for later retrieval.\n\n"
                "The user brings their own embedding provider key and vector DB credentials. "
                "scrapedatshi handles the scraping, chunking, embedding orchestration, and injection.\n\n"
                "To discover supported embedding providers and their required fields, call "
                "list_embedding_providers first. To discover supported vector DBs and their "
                "required config fields, call list_vector_db_providers first.\n\n"
                "Keys can be omitted if set as environment variables (e.g. OPENAI_API_KEY, "
                "PINECONE_API_KEY) in the MCP config."
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
                        "description": (
                            "Embedding provider key. Supported: 'openai', 'cohere', 'gemini', "
                            "'mistral', 'voyage', 'ollama'. Call list_embedding_providers for details."
                        ),
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
                        "description": (
                            "Model name for the embedding provider. Required for all providers. "
                            "Examples: openai→'text-embedding-3-small', cohere→'embed-english-v3.0', "
                            "gemini→'text-embedding-004', mistral→'mistral-embed', "
                            "voyage→'voyage-3', ollama→'nomic-embed-text'."
                        ),
                    },
                    "embedding_api_key": {
                        "type": "string",
                        "description": (
                            "API key for the embedding provider. Can be omitted if the corresponding "
                            "env var is set (OPENAI_API_KEY, COHERE_API_KEY, GEMINI_API_KEY, etc.). "
                            "Pass empty string for Ollama (no key required)."
                        ),
                    },
                    "embedding_endpoint": {
                        "type": "string",
                        "description": (
                            "Public HTTPS endpoint for Ollama only. Must be publicly accessible — "
                            "use ngrok to expose your local Ollama: 'ngrok http 11434'."
                        ),
                    },
                    "vector_db": {
                        "type": "string",
                        "description": (
                            "Vector DB provider key. Supported: 'pinecone', 'qdrant', 'chroma', "
                            "'supabase', 'weaviate', 'mongodb', 'azure_cosmos', 'azure_cosmos_mongo', "
                            "'lancedb'. Call list_vector_db_providers for required config fields."
                        ),
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
                            "Provider-specific configuration dict. Required fields vary by provider. "
                            "Call list_vector_db_providers to see required fields for each provider. "
                            "API keys within this config can be omitted if set as env vars "
                            "(PINECONE_API_KEY, QDRANT_API_KEY, WEAVIATE_API_KEY). "
                            "Examples:\n"
                            '  pinecone: {"index_host": "https://my-index.svc.pinecone.io"}\n'
                            '  qdrant: {"url": "https://cluster.qdrant.io", "collection_name": "docs"}\n'
                            '  supabase: {"connection_string": "postgresql://...", "table_name": "documents"}\n'
                            '  chroma: {"collection_name": "docs"}'
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
                        "description": "Use headless browser to render JS before scraping.",
                        "default": False,
                    },
                    "contextual_retrieval": {
                        "type": "boolean",
                        "description": "Enable RAG 2.0 contextual enrichment. Requires llm_provider and llm_api_key.",
                        "default": False,
                    },
                    "llm_provider": {
                        "type": "string",
                        "description": "LLM provider for contextual retrieval. One of: 'openai', 'anthropic', 'gemini'.",
                        "enum": ["openai", "anthropic", "gemini"],
                    },
                    "llm_api_key": {
                        "type": "string",
                        "description": "API key for the LLM provider. Can be omitted if set as an env var.",
                    },
                    "llm_model": {
                        "type": "string",
                        "description": "LLM model name for contextual retrieval.",
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
        # ── list_embedding_providers ──────────────────────────────────────────
        types.Tool(
            name="list_embedding_providers",
            description=(
                "Returns a list of all supported embedding providers with their labels, "
                "whether they require an API key, and notes on available models. "
                "Call this before sync_to_vectordb to help the user choose an embedding provider "
                "and understand what model names to use."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        # ── list_vector_db_providers ──────────────────────────────────────────
        types.Tool(
            name="list_vector_db_providers",
            description=(
                "Returns a list of all supported vector database providers with their labels, "
                "required config fields, optional fields, and setup notes. "
                "Call this before sync_to_vectordb to help the user understand what "
                "vector_db_config fields they need to provide for their chosen database."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    ]


# ── Tool call handlers ────────────────────────────────────────────────────────


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:

    try:
        if name == "scrape_url":
            return await _handle_scrape_url(arguments)
        elif name == "crawl_site":
            return await _handle_crawl_site(arguments)
        elif name == "extract_data":
            return await _handle_extract_data(arguments)
        elif name == "extract_crawl":
            return await _handle_extract_crawl(arguments)
        elif name == "sync_to_vectordb":
            return await _handle_sync_to_vectordb(arguments)
        elif name == "list_embedding_providers":
            return _handle_list_embedding_providers()
        elif name == "list_vector_db_providers":
            return _handle_list_vector_db_providers()
        else:
            return [types.TextContent(type="text", text=f"❌ Unknown tool: {name}")]
    except Exception as exc:
        return [types.TextContent(type="text", text=_format_error(exc))]


# ── Individual tool handlers ──────────────────────────────────────────────────


async def _handle_scrape_url(arguments: dict) -> list[types.TextContent]:
    url = arguments.get("url")
    if not url:
        return [types.TextContent(type="text", text="❌ 'url' is required.")]

    # Resolve optional LLM key if contextual retrieval is requested
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
        )
    finally:
        client.close()

    # Format output
    lines = [
        f"✅ Scraped: {result.source}",
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

    max_pages = arguments.get("max_pages", 10)

    client = _get_client()
    try:
        result = client.pipeline.crawl(
            url=url,
            max_pages=max_pages,
            crawl_mode=arguments.get("crawl_mode", "sitemap"),
            selector=arguments.get("selector"),
            include_pattern=arguments.get("include_pattern"),
            exclude_pattern=arguments.get("exclude_pattern"),
            js_render=arguments.get("js_render", False),
            contextual_retrieval=contextual_retrieval,
            llm_provider=llm_provider if contextual_retrieval else None,
            llm_api_key=llm_api_key,
            llm_model=arguments.get("llm_model"),
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

    max_pages = arguments.get("max_pages", 5)

    client = _get_client()
    try:
        result = client.pipeline.extract_crawl(
            url=url,
            schema=schema,
            llm_provider=llm_provider,
            llm_api_key=llm_api_key,
            llm_model=arguments.get("llm_model"),
            crawl_mode=arguments.get("crawl_mode", "sitemap"),
            max_pages=max_pages,
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
                text="❌ 'embedding_provider' is required. Call list_embedding_providers to see options.",
            )
        ]
    if not vector_db:
        return [
            types.TextContent(
                type="text",
                text="❌ 'vector_db' is required. Call list_vector_db_providers to see options.",
            )
        ]

    # Resolve embedding key
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

    # Resolve vector DB config with env-var key injection
    vector_db_config = _resolve_vector_db_config(arguments, vector_db)

    # Validate required fields for the chosen vector DB
    provider_info = VECTOR_DB_PROVIDERS.get(vector_db, {})
    required_fields = provider_info.get("required_fields", [])
    missing = [f for f in required_fields if not vector_db_config.get(f)]
    if missing:
        return [
            types.TextContent(
                type="text",
                text=(
                    f"❌ Missing required fields for vector DB '{vector_db}': {missing}. "
                    f"Call list_vector_db_providers for details on required fields."
                ),
            )
        ]

    # Resolve optional LLM key for contextual retrieval
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
    import asyncio

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
