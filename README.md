# scrapedatshi-mcp

MCP (Model Context Protocol) server for the [scrapedatshi](https://scrapedatshi.com) RAG pipeline API.

Use scrapedatshi's scraping, crawling, extraction, and vector DB sync tools directly from **Claude Desktop** — no code required.

---

## What you can do

Just talk to Claude naturally:

- *"Scrape https://docs.example.com and give me the chunks"*
- *"Crawl https://example.com/products and extract the title and price from every page"*
- *"Sync https://docs.example.com to my Pinecone index using OpenAI embeddings"*
- *"What embedding providers does scrapedatshi support?"*

---

## Tools exposed

| Tool | What it does |
|---|---|
| `verify_provider_key` | Verify an LLM or embedding API key + get live model list |
| `get_usage_guide` | Returns the guided wizard flow and tool selection reference |
| `scrape_url` | Scrape & chunk a single URL into RAG-ready text segments |
| `chunk_file` | Upload a local file (PDF, MD, TXT, etc.) and chunk it into RAG-ready segments |
| `crawl_site` | Crawl an entire site (sitemap or spider mode) and return all chunks |
| `extract_data` | Extract structured schema fields from a URL using your LLM |
| `extract_crawl` | Multi-page schema extraction via site crawl |
| `sync_to_vectordb` | Full pipeline: scrape URL → embed → inject into your vector DB |
| `ingest_file` | Full pipeline: upload local file → embed → inject into your vector DB |
| `autorag` | Full pipeline: crawl entire site → chunk → embed → inject into your vector DB |
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

Open your Claude Desktop config file:

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

## Supported environment variables

| Variable | Used for |
|---|---|
| `SCRAPEDATSHI_API_KEY` | scrapedatshi API key (**required**) |
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

## Example conversations

### Scrape a single page

> **You:** Scrape https://docs.example.com/getting-started and show me the chunks.

Claude calls `scrape_url` and returns the chunked content with token counts and credit usage.

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

---

## Safety limits

To prevent runaway credit usage and client timeouts:

- `crawl_site`: defaults to **10 pages**, maximum 200
- `extract_crawl`: defaults to **5 pages**, maximum 50 per call

Claude will always confirm page limits with you before calling multi-page tools.

---

## License

MIT — see [LICENSE](LICENSE)
