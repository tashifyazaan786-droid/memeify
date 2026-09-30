# Meme Knowledge Agent

**Wildcard Track — Agentic AI Hackathon**

A single agentic pipeline — **Research → Understand → Retrieve → Analyze →
Verify → Store → Generate** — that turns a static meme encyclopedia into a
self-growing, evidence-backed knowledge base, and reuses that same verified
memory to power both a meme-culture explainer and a meme-based marketing
tool. One system, two outputs — not two disconnected chatbots.

This is a rebuild of an earlier project (`Meme Historian + Marketing
Agent`) that had the same two capabilities but as separate tabs, backed by
a static hand-written knowledge base with no way to grow, and no visible
reasoning or verification step.

---

## What's actually new here

| | Previous version | This version |
|---|---|---|
| Knowledge base | Static file, grows only if a human edits it | Grows itself: the agent researches, verifies, and saves new entries during normal use |
| Trust | LLM's claims taken at face value | A deterministic (non-LLM) verification gate scores every new claim before it can be saved or used |
| Historian / Marketing | Two separate agents, two tabs, no shared story | One shared research+verify pipeline; the two modes are just different final-generation steps over the same evidence |
| Visibility | Black-box final answer only | A live "agent activity" panel shows what it searched, what it found, and why it trusted (or didn't trust) the result |
| Persistence safety | N/A | The LLM can *read* memory via tools, but cannot write to it directly — only the deterministic verification gate (or an explicit human click) can persist a new fact |

---

## Architecture

```
                     ┌─────────────────────────────┐
   User query   ───▶ │   PHASE 1 — Research Agent   │   (agentic, tool-using)
 (topic or biz)      │   langchain create_agent     │
                     │   tools:                      │
                     │    • search_knowledge_base     │
                     │    • search_web_for_context     │
                     │    • get_live_trending_topics    │
                     └─────────────┬────────────────┘
                                   │ structured JSON candidate(s)
                                   ▼
                     ┌─────────────────────────────┐
                     │  PHASE 2 — Verification gate │   (deterministic, code)
                     │  verification.score_confidence│
                     │   HIGH   → auto-save to KB     │
                     │   MEDIUM → held as draft        │
                     │   LOW    → held back, not used   │
                     └─────────────┬────────────────┘
                                   │ verified/usable context
                                   ▼
                     ┌─────────────────────────────┐
                     │  PHASE 3 — Generation (plain  │   (single LLM call,
                     │  LLM call, not agentic)       │    no tools)
                     │   Historian explanation   OR   │
                     │   Marketing ad concepts         │
                     └─────────────────────────────┘

Knowledge base: data/memes.md (source of truth, versionable) +
                Chroma vector store (chroma_db/, retrievable memory)
```

**Why writing isn't an agent tool.** The LLM agent can call
`search_knowledge_base` to *read* memory, but there is no `save_to_kb`
tool available to it. Persisting a new fact only ever happens through
`verification.score_confidence` (automatic, for well-corroborated
findings) or `agent.approve_and_save` (triggered by a human clicking
Approve in the UI). This means a confused or adversarially-prompted model
can't corrupt the permanent knowledge base on its own — a deliberate
reliability/security choice, not an oversight.

**Why verification is deterministic, not another LLM call.** Asking a
model to self-rate its own confidence is a well-known reliability trap —
and a live demo is the worst place for that trap to spring. Instead,
`verification.py` scores confidence from things that can actually be
counted: how many independent sources were found, whether the extraction
step reported agreement or conflict between them, and whether the
extracted content is substantive rather than empty/hedged.

**Honest limitation:** there is no ground-truth oracle for meme folklore.
"Verified" here means *corroborated by multiple independent live sources
and internally consistent* — not "certified factually true." The agent's
own output says this explicitly; it doesn't oversell its certainty.

---

## Project layout

| File | Purpose |
|---|---|
| `config.py` | Paths, models, thresholds, languages, fallbacks |
| `knowledge_store.py` | RAG retrieval + the only code path that can write a new verified entry |
| `research_tools.py` | The 3 agent-facing tools (KB search, web search, live trends) |
| `verification.py` | The deterministic confidence-scoring gate |
| `agent.py` | Orchestrator: research agent → extraction → verify/gate → generation |
| `app.py` | Unified Gradio UI (one page, mode toggle, activity log, approval flow) |
| `data/memes.md` | Seed knowledge base (curated, pre-marked `Verified: Yes`) |

---

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env → GROQ_API_KEY=gsk_...
```

Free Groq key: https://console.groq.com/keys

```bash
python app.py
```

- First launch builds the Chroma store from `data/memes.md` (a few seconds,
  downloads the `BAAI/bge-small-en-v1.5` embedding model, ~130 MB, once).
- Later launches load the persisted store from `chroma_db/`.
- New verified entries are appended to `data/memes.md` directly, so the
  knowledge base's growth is visible in plain text / version control, not
  hidden inside a binary vector store.

## Sanity-check pieces independently

```bash
python verification.py    # (run the __main__ block, or see the pytest-style checks in this README)
python research_tools.py  # trends + KB search + web research
python agent.py           # full pipeline on one historian query
```

---

## Demo flow (~4 minutes)

1. **Historian, known topic**: ask about something already in the seed KB
   (e.g. *"What is aura farming?"*) — show the activity log: knowledge base
   hit, no live research needed, instant grounded answer.
2. **Historian, unknown topic**: ask about something NOT in the KB, or a
   very recent trend — show the activity log doing live research, the
   confidence tier it lands on, and (if high confidence) the knowledge
   base panel now containing that new entry it just taught itself.
3. **Marketing mode**: describe a real business, watch it pull live
   trends, verify 1–3 of them, and generate ad concepts grounded only in
   what passed verification — call out that this reuses the exact same
   pipeline from steps 1–2, not a separate system.
4. **Trust story**: point at a medium/low-confidence draft sitting in the
   "awaiting approval" panel — explain that the agent chose *not* to trust
   it automatically, and that a human has to vouch for it before it
   becomes permanent memory.

---

## What was verified vs. what needs a live run

I don't have network access to Groq, Reddit, or DuckDuckGo from the
environment this was built in, so I could not run a full live pipeline
end-to-end myself. What I did verify directly, with the actual installed
packages:

- ✅ All files parse/compile with no syntax errors.
- ✅ Every third-party import used (`langchain.agents.create_agent`,
  `langchain_groq.ChatGroq`, `langchain_community.utilities.DuckDuckGoSearchAPIWrapper`,
  `langchain_text_splitters.MarkdownHeaderTextSplitter`, `gradio`) resolves
  to a real class/function at that path — none of the API is guessed.
- ✅ `MarkdownHeaderTextSplitter` run against the actual `data/memes.md` —
  correctly produces one chunk per meme with `meme_name` metadata.
- ✅ `verification.score_confidence` unit-tested across high/medium/low/
  conflicting/empty-content cases (this caught and fixed a real bug: the
  original substantive-content check was miscalibrated and misclassified
  valid short content as low-confidence).
- ✅ The JSON-extraction logic that parses the research agent's structured
  output tested against fenced, unfenced, multi-candidate, and malformed
  inputs.
- ⚠️ Not run end-to-end: the actual Groq LLM calls, live Reddit/web
  search, and the Chroma + sentence-transformers embedding pipeline
  (installing `sentence-transformers`/`torch` exceeded this sandbox's
  available disk space). These are standard, widely-used integrations and
  the previous project used the same Chroma/embedding stack successfully,
  but you should do one full local run before the demo to confirm.

---

## Must-have vs. nice-to-have (if you're short on time before the hackathon)

**Must-have (already built):** the full pipeline above, both modes, the
approval UI, the activity log.

**High-impact, not yet built — worth adding if you have a day or two:**
- Show retrieved-source links (not just domain names) in the UI so judges
  can click through and check a claim themselves.
- A "regenerate with more research" button when confidence lands medium.

**Nice-to-have:**
- Export the knowledge base growth as a shareable timeline ("started with
  30 entries, learned 6 more live during this demo").
- A second verification signal: LLM self-consistency (ask the same
  question twice, check if the answer changes) alongside the source-count
  heuristic.
