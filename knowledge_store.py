"""
The agent's long-term memory.

Two responsibilities, kept in one module because they're really one
concern (the knowledge base), not two:

1. RETRIEVE — semantic search over `data/memes.md`, split one `##` heading
   per chunk so slang/partial/fuzzy queries still land on the right entry.
2. STORE     — append a newly *verified* entry to `data/memes.md` (the
   durable source of truth) AND add it to the live Chroma collection so it
   is retrievable immediately, without a restart or rebuild.

Deliberate design choice: writing to this store is NOT exposed to the LLM
agent as a callable tool. The agent can only *read* memory. Persisting a
new fact is only ever triggered by the deterministic verification gate in
`verification.py` (auto-save on HIGH confidence) or by an explicit human
click in the UI (MEDIUM/LOW confidence drafts). This keeps an LLM — which
can be steered by adversarial or just plain wrong input — from being able
to corrupt the knowledge base on its own say-so.
"""

from __future__ import annotations

import logging
import re
import shutil
from datetime import datetime, timezone
from typing import Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter

import config

logger = logging.getLogger(__name__)

_embeddings = None
_vectorstore: Optional[Chroma] = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def _load_and_split() -> list:
    """Load data/memes.md and split it so each ## heading is one chunk."""
    text = config.DATA_PATH.read_text(encoding="utf-8")
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", "meme_name")], strip_headers=False
    )
    docs = splitter.split_text(text)

    for doc in docs:
        name = doc.metadata.get("meme_name", "Unknown")
        doc.metadata["meme_name"] = name
        era_match = re.search(r"\*\*Era:\*\*\s*(.+)", doc.page_content)
        if era_match:
            doc.metadata["era"] = era_match.group(1).strip()
        verified_match = re.search(r"\*\*Verified:\*\*\s*(.+)", doc.page_content)
        doc.metadata["verified"] = bool(
            verified_match and verified_match.group(1).lower().startswith("yes")
        )
    return docs


def build_vectorstore(force_rebuild: bool = False) -> Chroma:
    global _vectorstore
    if _vectorstore is not None and not force_rebuild:
        return _vectorstore

    persist = str(config.PERSIST_DIR)

    if force_rebuild and config.PERSIST_DIR.exists():
        shutil.rmtree(config.PERSIST_DIR)

    if not force_rebuild and config.PERSIST_DIR.exists() and any(config.PERSIST_DIR.iterdir()):
        logger.info("Loading existing vector store from %s", persist)
        _vectorstore = Chroma(
            collection_name="meme_knowledge",
            persist_directory=persist,
            embedding_function=_get_embeddings(),
        )
        return _vectorstore

    docs = _load_and_split()
    if not docs:
        raise RuntimeError("No meme chunks found — check data/memes.md formatting.")

    _vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=_get_embeddings(),
        collection_name="meme_knowledge",
        persist_directory=persist,
    )
    logger.info("Knowledge base built: %d entries", len(docs))
    return _vectorstore


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        return build_vectorstore()
    return _vectorstore


def search(query: str, k: int = config.RETRIEVER_K) -> list:
    """Return (doc, relevance_score) pairs, best first.

    Chroma's relevance-score API returns higher scores for stronger matches.
    """
    vs = get_vectorstore()
    return vs.similarity_search_with_relevance_scores(query, k=k)


def add_verified_entry(
    meme_name: str,
    meaning: str,
    origin: str,
    aliases: str,
    era: str,
    sources: list,
    confidence: str,
) -> None:
    """Persist a newly verified entry."""

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if sources:
        source_parts = []

        for source in sources:
            if isinstance(source, dict):
                title = source.get("title", "Source")
                url = source.get("url", "")
                domain = source.get("domain", "")

                if url:
                    source_parts.append(
                        f"{title} | {domain} | {url}"
                    )
                elif domain:
                    source_parts.append(
                        f"{title} | {domain}"
                    )
                else:
                    source_parts.append(title)
            else:
                source_parts.append(str(source))

        sources_str = "; ".join(source_parts)
    else:
        sources_str = "Live research (see agent activity log)"

    block = f"""
## {meme_name}

**Aliases:** {aliases or "Not specified"}
**Verified:** Yes (auto-verified {confidence} confidence, {timestamp})
**Sources:** {sources_str}
**Era:** {era or "Unknown"}
**Origin:** {origin}
**Meaning:** {meaning}
**Cultural Context:** Added automatically by the agent after live research and verification on {timestamp}.

---

"""

    with config.DATA_PATH.open("a", encoding="utf-8") as f:
        f.write(block)

    vs = get_vectorstore()

    from langchain_core.documents import Document

    doc = Document(
        page_content=block.strip(),
        metadata={
            "meme_name": meme_name,
            "era": era or "Unknown",
            "verified": True,
            "confidence": confidence,
            "added": timestamp,
        },
    )

    vs.add_documents([doc])

    logger.info(
        "Stored new verified entry: %s (%s confidence)",
        meme_name,
        confidence,
    )
def list_verified_entries(limit: int = 50) -> list[dict]:
    """For the UI's 'knowledge base' transparency panel."""
    docs = _load_and_split()
    out = []
    for d in docs:
        out.append(
            {
                "name": d.metadata.get("meme_name", "Unknown"),
                "era": d.metadata.get("era", "Unknown"),
                "verified": d.metadata.get("verified", False),
            }
        )
    return out[:limit]
