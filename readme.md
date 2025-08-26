# Filesystem Finder Agent

An AI agent that takes a **keyword, natural-language query, or file**, then **searches across all folders/drives you point it to** and returns:

* Exact locations (paths) that match
* Ranked results (keyword + semantic)
* A **markdown report** (auto-generated) summarizing what it found
* Optional **JSON** with structured hits (path, snippet, score, tags)

> Goal: Make “where is X in my mess of files?” effortless—whether X is a filename, phrase inside documents, a particular code construct, or a concept.

---

## Core Features

1. **Hybrid Search**

   * *Literal*: filename, extension, exact-phrase, regex
   * *Semantic*: embeddings + vector search for concepts (e.g., "performance tuning" -> finds `perf_notes.md`, relevant PRs)
   * *Metadata filters*: size, modified date, owner, extension, repo name

2. **Multi-Location Scanning**

   * Local paths (multiple roots)
   * External mounts (e.g., `/Volumes/*`)
   * (Optional later) Cloud connectors (S3, GDrive) via provider SDKs

3. **Agentic Planning (Claude)**

   * Planner agent (interprets the user’s intent, decides strategies: literal, semantic, code-aware)
   * Executor tools (filesystem walker, indexer, vector DB, previewer)
   * Synthesizer (writes a readable **report.md** and a machine-friendly **results.json**)

4. **Code-Aware Search (optional)**

   * Language-aware heuristics (function/class names, imports, dependency graphs)
   * Parse ASTs for precise matches (e.g., TypeScript, Python, Java)

5. **Auto Markdown Report**

   * For each query, produce `reports/<timestamp>_<slug>.md` with:

     * Summary of intent
     * What was searched (locations, filters)
     * Top N hits with context snippets
     * Reasoning notes from the agent
     * Next-step suggestions

6. **Privacy & Safety**

   * Explicit allowlist of roots
   * Ignore patterns (`.gitignore`-style)
   * Dry-run & preview modes

---

## Architecture (High-level)

```
User -> CLI/UI -> Orchestrator -> Planner (Claude) -> Tool Router -> {Indexer, FS Walker, VectorSearch, Preview}
                                                    -> Synthesizer (Claude) -> report.md + results.json
```

* **Orchestrator**: Manages a run; feeds context to Claude; caches partial results
* **Planner (Claude)**: Converts intent to steps & tool calls
* **Tools**:

  * `FS Walker`: walks directories, respects ignore rules, yields file paths, basic metadata
  * `Content Extractor`: text from files (txt, md, code, pdf via OCR/lib later)
  * `Indexer`: builds an index (keyword + embeddings) and stores in a local vector DB (e.g., LanceDB, SQLite+FAISS, Chroma)
  * `VectorSearch`: nearest-neighbor semantic lookup
  * `Previewer`: gathers snippets around matches
* **Synthesizer (Claude)**: Given raw hits + user prompt, writes a clean markdown summary and optional JSON

---

## Data Flow

1. **Ingest** (one-time or on demand)

   * Walk roots -> extract text -> chunk -> embed -> upsert to vector DB
2. **Plan**

   * Claude decides which search modes to use based on the prompt/file
3. **Search**

   * Literal + semantic queries
   * Combine & rerank
4. **Synthesize**

   * Claude writes `report.md` and saves `results.json`

---

## Example Prompts

* "Find any mention of `JWT` secrets or `.env` files modified last month in /Users/sneh/projects and /Volumes/WorkArchive."
* "Where did I store the doc about Stripe webhook retries?"
* "Given this PDF (upload), find related meeting notes and summarize decisions."

---

## CLI Sketch

```bash
finder \
  --roots "/Users/sneh/projects,/Volumes/WorkArchive" \
  --query "stripe webhook retries" \
  --filters "ext:md,txt;mtime:30d" \
  --out ./runs

# Or with a seed file:
finder --roots "/data" --file ./docs/contract.pdf --out ./runs
```

### Sample Output Tree

```
runs/
  2025-08-26_stripe-webhook-retries/
    report.md
    results.json
    logs/
```

---

##  Tech Stack

* **Language**: Python or TypeScript (choose your comfort)
* **LLM**: Claude (Anthropic API)
* **Embeddings**: Claude embeddings or open-source (e.g., BGE-small) if offline is needed
* **Vector DB**: LanceDB / FAISS / Chroma (local)
* **Parsing**: `textract`/`pypdf` (pdf), `python-magic` (mime), `tree-sitter` (code, optional)
* **Config**: `.finderagent.yaml` for roots, ignores, db path

