# scrapedatshi-mcp

MCP (Model Context Protocol) server for the [scrapedatshi](https://scrapedatshi.com) RAG pipeline API.

Use scrapedatshi's scraping, crawling, extraction, and vector DB sync tools directly from **Claude Desktop** — no code required.

---

## What you can do

Just talk to Claude naturally:

- *"Scrape https://docs.example.com and give me the chunks"*
- *"Extract the text from this PDF: https://example.com/annual-report.pdf"*
- *"Extract all tables from this local PDF: C:/Users/me/Documents/financials.pdf"*
- *"Chunk this PDF URL: https://my-bucket.s3.amazonaws.com/report.pdf"* — PDF URLs are automatically detected and extracted
- *"Crawl https://example.com/products and extract the title and price from every page"*
- *"Sync https://docs.example.com to my Pinecone index using OpenAI embeddings"*
- *"Crawl the entire docs.stripe.com site (all 800 pages) and inject it into my Pinecone index"* — large sites are auto-batched server-side, no manual pagination needed
- *"What embedding providers does scrapedatshi support?"*
- *"Inspect my Pinecone index and tell me what embedding model was used"*
- *"Query my Pinecone index for information about API authentication"*
- *"Query my LanceDB with hybrid search — I need to find exact IDs and names, not just semantic matches"*
- *"Chunk https://docs.example.com using hierarchical chunking so the LLM gets full context on retrieval"*
- *"Ingest all the JSON files in my ./scrapy_output/ folder into my Pinecone index"*

---

## Tools exposed

| Tool | What it does |
|---|---|
| `verify_provider_key` | Verify an LLM or embedding API key + get live model list |
| `get_usage_guide` | Returns the guided wizard flow and tool selection reference |
| `scrape_url` | Scrape a URL and return clean Markdown — no chunking, just the raw text |
| `pdf_extract` | Extract text or tables from a PDF (URL or local file) — no chunking, no embedding needed |
| `chunk_url` | Scrape & chunk a single URL into RAG-ready text segments |
| `chunk_file` | Upload a local file (PDF, MD, TXT, CSV, XLSX, DOCX, IPYNB, HTML, XML, code files, etc.) and chunk it into RAG-ready segments |
| `crawl_site` | Crawl an entire site (sitemap or spider mode) and return all chunks |
| `extract_data` | Extract structured schema fields from a URL using your LLM |
| `extract_crawl` | Multi-page schema extraction via site crawl |
| `sync_to_vectordb` | Full pipeline: scrape URL → embed → inject into your vector DB |
| `ingest_file` | Full pipeline: upload local file → embed → inject into your vector DB |
| `ingest_scraped` | Full pipeline: bulk-ingest a folder of pre-scraped files → embed → inject into your vector DB |
| `autorag` | Full pipeline: crawl entire site → chunk → embed → inject into your vector DB (large sites auto-batched) |
| `inspect_vectordb` | Read vector DB metadata: dimension, vector count, suggested embedding models (free) |
| `query_vectordb` | Semantic search: embed a query and retrieve the most relevant chunks from your vector DB. Supports `hybrid_search=true` (vector + BM25 + RRF) for cross-referencing accuracy |
| `rag_chat` | RAG Chat: retrieve top-N chunks from your vector DB and generate a grounded LLM answer |
| `list_embedding_providers` | Discover supported embedding providers + model notes |
| `list_vector_db_providers` | Discover supported vector DBs + required config fields |

---

## Prerequisites

1. **scrapedatshi account** — [Sign up at scrapedatshi.com](https://scrapedatshi.com)
2. **Add credits** — [Billing portal](https://scrapedatshi.com/portal/billing)
3. **Get your API key** — starts with `sds_...`
4. **Claude Desktop** — [Download here](https://claude.ai/download)
5. **Python 3.10+** — [python.org](https://python.org)

---

## Installation

### Option A — Install from PyPI (recommended, works with `uvx`)

```bash
pip install scrapedatshi-mcp
```

Or use [uv](https://docs.astral.sh/uv/) for isolated installs:

```bash
uv tool install scrapedatshi-mcp
```

### Option B — Install from source (local development)

```bash
git clone https://github.com/scrapedatshi/scrapedatshi-mcp.git
cd scrapedatshi-mcp
pip install -e .
```

---

## Claude Desktop configuration

**Easiest way to find your config file:** Open Claude Desktop → **Settings** → **Developer** → **Edit Config**

Alternatively, the file is located at:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

### Recommended — `uvx` with all provider SDKs (auto-updates on restart)

```json
{
  "mcpServers": {
    "scrapedatshi": {
      "command": "uvx",
      "args": [
        "--from", "scrapedatshi-mcp[all]",
        "--refresh",
        "scrapedatshi-mcp"
      ],
      "env": {
        "SCRAPEDATSHI_API_KEY": "sds_your_key_here"
      }
    }
  }
}
```

- `[all]` installs all provider SDKs (OpenAI, Anthropic, Gemini, Voyage AI) so `verify_provider_key` works for any provider
- `--refresh` checks PyPI for updates every time Claude Desktop starts — no manual reinstalls needed

### If installed via pip (using `python`)

```json
{
  "mcpServers": {
    "scrapedatshi": {
      "command": "python",
      "args": ["-m", "scrapedatshi_mcp.server"],
      "env": {
        "SCRAPEDATSHI_API_KEY": "sds_your_key_here"
      }
    }
  }
}
```

### If cloned from source (absolute path)

```json
{
  "mcpServers": {
    "scrapedatshi": {
      "command": "python",
      "args": ["/absolute/path/to/scrapedatshi-mcp/scrapedatshi_mcp/server.py"],
      "env": {
        "SCRAPEDATSHI_API_KEY": "sds_your_key_here"
      }
    }
  }
}
```

Restart Claude Desktop after saving the config.

---

## Secure key configuration (BYOK)

You bring your own LLM, embedding, and vector DB keys. The server resolves keys in this priority order:

1. **Argument passed in the tool call** — explicit override
2. **Environment variable in the MCP config** — preferred secure path (keys never appear in chat)
3. **Clear error message** if neither is found

Add your provider keys to the `env` block in `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "scrapedatshi": {
      "command": "uvx",
      "args": [
        "--from", "scrapedatshi-mcp[all]",
        "--refresh",
        "scrapedatshi-mcp"
      ],
      "env": {
        "SCRAPEDATSHI_API_KEY": "sds_your_key_here",

        "OPENAI_API_KEY": "sk-...",
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "GEMINI_API_KEY": "AIza...",

        "COHERE_API_KEY": "...",
        "MISTRAL_API_KEY": "...",
        "VOYAGE_API_KEY": "...",

        "PINECONE_API_KEY": "pc-...",
        "QDRANT_API_KEY": "...",
        "WEAVIATE_API_KEY": "..."
      }
    }
  }
}
```

Once set, Claude will automatically use these keys without asking you to type them in chat.

---

## Fetch Mode

Starting in v0.5.0, the MCP server uses **local-fetch mode by default** — URLs are fetched on the machine running Claude Desktop (your IP), and only the HTML processing runs on our server. This is cheaper and keeps your IP off our server.

### `SCRAPEDATSHI_FETCH_MODE=local` (default)

The MCP server fetches URLs using the machine's own IP address, then submits the raw HTML to our server for processing.

- ✅ Your IP is used — not our server's
- ✅ Billed at the standard per-URL rate ($0.0020)
- ✅ Faster — no double-hop latency

### `SCRAPEDATSHI_FETCH_MODE=server`

Our server fetches the URL. Use this if Claude Desktop is running in a restricted environment without outbound HTTP access, or if you need server-managed IP rotation.

- ⚠️ Our server's IP is used
- ⚠️ Billed at 2× the standard rate ($0.0040 / URL)
- ✅ Works from restricted environments

To enable server fetch, add `SCRAPEDATSHI_FETCH_MODE` to your MCP config:

```json
{
  "mcpServers": {
    "scrapedatshi": {
      "command": "uvx",
      "args": ["--from", "scrapedatshi-mcp[all]", "--refresh", "scrapedatshi-mcp"],
      "env": {
        "SCRAPEDATSHI_API_KEY": "sds_your_key_here",
        "SCRAPEDATSHI_FETCH_MODE": "server"
      }
    }
  }
}
```

---

## Supported environment variables

| Variable | Used for |
|---|---|
| `SCRAPEDATSHI_API_KEY` | scrapedatshi API key (**required**) |
| `SCRAPEDATSHI_FETCH_MODE` | `local` (default) or `server` — see Fetch Mode above |
| `OPENAI_API_KEY` | OpenAI LLM + embedding |
| `ANTHROPIC_API_KEY` | Anthropic LLM (Claude) |
| `GEMINI_API_KEY` | Google Gemini LLM + embedding |
| `COHERE_API_KEY` | Cohere embedding |
| `MISTRAL_API_KEY` | Mistral embedding |
| `VOYAGE_API_KEY` | Voyage AI embedding |
| `PINECONE_API_KEY` | Pinecone vector DB |
| `QDRANT_API_KEY` | Qdrant vector DB (optional for local) |
| `WEAVIATE_API_KEY` | Weaviate vector DB (optional for local) |

---

## Authenticated Scraping (v0.5.1+)

For pages behind a login wall, you can pass your session cookies and/or custom headers to `scrape_url` and `crawl_site`. Credentials are **only sent to URLs within the permitted domain scope** — they are never leaked to external domains.

### Scrape a login-walled page

Just tell Claude:

> *"Scrape https://internal.company.com/wiki/api-docs — use my session cookie: abc123"*

Claude will call `scrape_url` with:

```json
{
  "url": "https://internal.company.com/wiki/api-docs",
  "cookies": {"session": "abc123"},
  "headers": {"Authorization": "Bearer eyJ..."}
}
```

### Authenticated crawl with subdomain scope

> *"Crawl https://company.com including wiki.company.com and docs.company.com — use session cookie abc123"*

Claude will call `crawl_site` with:

```json
{
  "url": "https://company.com",
  "cookies": {"session": "abc123"},
  "allow_subdomains": true,
  "max_pages": 20
}
```

**Security model:**
- Cookies and headers are **only sent to URLs within the permitted domain scope** — never to external domains discovered during crawling
- `allow_subdomains: false` (default): only the exact hostname receives credentials
- `allow_subdomains: true`: credentials are shared with subdomains of the root domain (e.g. `wiki.company.com` when root is `company.com`). Multi-part TLDs (`.co.uk`, `.com.br`) are handled safely.
- Credentials are **never forwarded to the scrapedatshi server** — they stay on the machine running Claude Desktop

### Enterprise SSO / MFA — Session Capture (v0.6.4+)

For enterprise portals protected by Okta, Duo, or any SSO/MFA flow that blocks automated login, use the SDK's `capture_session()` utility to authenticate manually in a real browser, then pass the captured session state to Claude via the `storage_state` parameter.

**Step 1 — Capture the session locally (run once):**

```bash
pip install scrapedatshi[auth]
playwright install chromium
```

```python
from scrapedatshi.auth import capture_session
import json

state = capture_session(
    "https://internal.company.com/login",
    save_to="session.auth.json",   # gitignored automatically
)
```

This opens a real browser window. Log in manually (including any MFA prompts), then press Enter. The session state is saved to `session.auth.json`.

**Step 2 — Tell Claude to use the saved session:**

> *"Crawl https://internal.company.com using the session state in session.auth.json"*

Claude will call `crawl_site` with the `storage_state` parameter containing the captured session.

> **⚠ Security:** `session.auth.json` contains live authentication tokens. Never commit it to version control. The SDK's `.gitignore` template automatically filters `*.auth.json` files.

---

## Example conversations

### Get clean Markdown from a page

> **You:** Scrape https://docs.example.com/getting-started and show me the content.

Claude calls `scrape_url` and returns the full page as clean Markdown — title, credits used, and the raw text in one piece.

The response also includes `selectors_found` — a list of CSS selectors for the main content sections detected on the page. Claude can use these to re-scrape just a specific section:

> **You:** Now scrape just the pricing section.

Claude calls `scrape_url` again with `selector="section#pricing"` (from `selectors_found`).

---

### Chunk a page for RAG

> **You:** Chunk https://docs.example.com/getting-started into RAG-ready segments.

Claude calls `chunk_url` and returns the structured chunks with token counts and credit usage.

---

### Crawl a documentation site

> **You:** Crawl https://docs.example.com — just the first 5 pages.

Claude calls `crawl_site` with `max_pages=5` and returns all chunks from all pages.

---

### Extract structured data from a product page

> **You:** Extract the product name, price, and whether it's in stock from https://example.com/products/widget-pro

Claude calls `extract_data` with a schema it constructs from your request, using your OpenAI key from the env config.

---

### Extract data from an entire product catalogue

> **You:** Crawl https://example.com/products and extract the title and price from every product page. Limit to 10 pages.

Claude calls `extract_crawl` with `max_pages=10` and returns per-page extraction results.

---

### Sync a page to your vector DB

> **You:** Sync https://docs.example.com to my Pinecone index. The index host is https://my-index-abc123.svc.pinecone.io. Use OpenAI text-embedding-3-small.

Claude calls `sync_to_vectordb`. If `OPENAI_API_KEY` and `PINECONE_API_KEY` are set in your env config, no keys need to be typed in chat.

---

### Discover what's supported

> **You:** What embedding providers does scrapedatshi support?

Claude calls `list_embedding_providers` and returns a formatted list with model notes.

> **You:** What fields do I need to configure for Qdrant?

Claude calls `list_vector_db_providers` and returns the required and optional fields for each provider.

---

## Supported providers

### Embedding providers

| Key | Provider |
|---|---|
| `openai` | OpenAI (text-embedding-3-small, text-embedding-3-large, ada-002) |
| `cohere` | Cohere (embed-english-v3.0, embed-multilingual-v3.0) |
| `gemini` | Google Gemini (text-embedding-004, gemini-embedding-001) |
| `mistral` | Mistral (mistral-embed) |
| `voyage` | Voyage AI (voyage-3, voyage-3-lite, voyage-code-3) |
| `ollama` | Ollama local (nomic-embed-text, mxbai-embed-large, etc.) |

### Vector databases

| Key | Provider |
|---|---|
| `pinecone` | Pinecone |
| `qdrant` | Qdrant |
| `chroma` | ChromaDB (local) |
| `supabase` | Supabase (pgvector) |
| `weaviate` | Weaviate |
| `mongodb` | MongoDB Atlas |
| `azure_cosmos` | Azure Cosmos DB (NoSQL) |
| `azure_cosmos_mongo` | Azure Cosmos DB (MongoDB API) |
| `lancedb` | LanceDB (local) |

### LLM providers (for extraction + contextual retrieval)

| Key | Provider |
|---|---|
| `openai` | OpenAI (gpt-4o-mini, gpt-4o, etc.) |
| `anthropic` | Anthropic (claude-3-haiku, claude-3-5-sonnet, etc.) |
| `gemini` | Google Gemini (gemini-1.5-flash, gemini-1.5-pro, etc.) |

---

## Billing

- Credits are deducted from your scrapedatshi account after each successful API call
- Failed requests are not charged
- Every tool response includes `credits_used` and `credits_remaining`
- LLM, embedding, and vector DB costs are billed directly by your chosen providers — scrapedatshi only charges for scraping and orchestration
- Top up at [scrapedatshi.com/portal/billing](https://scrapedatshi.com/portal/billing)

### Per-URL rates

| Mode | Rate | When |
|---|---|---|
| Local fetch (default) | $0.0020 / URL | `SCRAPEDATSHI_FETCH_MODE=local` (default) |
| Server fetch | $0.0040 / URL | `SCRAPEDATSHI_FETCH_MODE=server` |
| Spider crawl (server) | $0.0050 / URL | `/v1/spider` — server-side link-following |
| Chunk fee | $0.0005 / chunk | All routes |
| Injection fee | $0.0030 / chunk | sync_to_vectordb, ingest_file, autorag |
| Contextual Retrieval | $0.0010 / chunk | When `contextual_retrieval=true` |
| Vector query | $0.0002 / chunk | query_vectordb, rag_chat |

---

## Auto-Batching for Large Sites

When you ask Claude to crawl a large site (more than 200 pages), the `autorag` and `crawl_site` tools automatically split the job into sequential batches server-side. You don't need to do anything special — just ask Claude to crawl the site and it handles the rest.

> **You:** Crawl the entire docs.stripe.com site and inject everything into my Pinecone index.

Claude calls `autorag` with a high `max_pages` value. If the site has 600 pages, the server processes it as 3 batches of 200 pages each and returns the combined result.

The response will include `auto_batched: true` and `batches_processed: N` when batching occurred.

---

## Safety limits

To prevent runaway credit usage and client timeouts:

- `crawl_site`: defaults to **10 pages**, maximum 200 per batch (auto-batched for larger jobs)
- `autorag`: defaults to **5 pages**, no hard upper limit — large jobs are auto-batched
- `extract_crawl`: defaults to **5 pages**, maximum 50 per call

Claude will always confirm page limits with you before calling multi-page tools.

---

## Troubleshooting

### Contextual Retrieval fails — "model no longer available"

LLM providers periodically deprecate older models. If you see an error like *"This model is no longer available"*, run `verify_provider_key` again to get the current list of available models for your key, then select a current model.

**Current recommended models for contextual retrieval:**
- **Gemini**: `gemini-2.5-flash` or `gemini-2.0-flash-001` (not `gemini-2.0-flash` — deprecated)
- **OpenAI**: any current `gpt-4o` or `gpt-4.1` series model
- **Anthropic**: any current `claude-3-5` or `claude-3-7` series model

**Provider model & deprecation pages:**
- OpenAI: <a href="https://platform.openai.com/docs/deprecations" target="_blank" rel="noopener noreferrer">platform.openai.com/docs/deprecations</a>
- Anthropic: <a href="https://docs.anthropic.com/en/docs/about-claude/models" target="_blank" rel="noopener noreferrer">docs.anthropic.com/en/docs/about-claude/models</a>
- Google Gemini: <a href="https://ai.google.dev/gemini-api/docs/models" target="_blank" rel="noopener noreferrer">ai.google.dev/gemini-api/docs/models</a>
- Cohere: <a href="https://docs.cohere.com/docs/models" target="_blank" rel="noopener noreferrer">docs.cohere.com/docs/models</a>
- Mistral: <a href="https://docs.mistral.ai/getting-started/models/" target="_blank" rel="noopener noreferrer">docs.mistral.ai/getting-started/models</a>
- Voyage AI: <a href="https://docs.voyageai.com/docs/embeddings" target="_blank" rel="noopener noreferrer">docs.voyageai.com/docs/embeddings</a>

---

### Contextual Retrieval fails — "quota exceeded"

Your LLM provider API key has no remaining credits. Add credits at your provider's billing page. Note that **scrapedatshi credits are separate from your LLM provider credits** — you need both.

---

### `verify_provider_key` returns no models

If key verification succeeds but returns an empty model list, your API key may be restricted to specific model families or your account may have limited access. Check your provider's dashboard for account restrictions.

---

### Claude Desktop doesn't show scrapedatshi tools

1. Make sure you saved `claude_desktop_config.json` correctly (valid JSON, no trailing commas)
2. Fully quit and reopen Claude Desktop — a simple window close is not enough
3. Check that `uvx` is installed: run `uvx --version` in your terminal
4. If using `--refresh`, the first startup may take a few seconds to download the package

---

## License

MIT — see [LICENSE](LICENSE)
