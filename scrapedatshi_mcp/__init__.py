"""
scrapedatshi-mcp
~~~~~~~~~~~~~~~~
MCP (Model Context Protocol) server for the scrapedatshi RAG pipeline API.

Exposes scrapedatshi pipeline tools to Claude Desktop and any MCP-compatible
AI client, allowing conversational access to scraping, crawling, extraction,
and vector DB sync operations.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("scrapedatshi-mcp")
except PackageNotFoundError:
    __version__ = "unknown"
