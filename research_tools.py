"""
Tools the research agent can call. All three are READ-ONLY with respect to
the permanent knowledge base — writing is handled separately (see
knowledge_store.add_verified_entry), never as an agent-callable tool.

1. search_knowledge_base    - check what we already know (RAG)
2. get_live_trending_topics - what's hot right now (Reddit + fallback)
3. search_web_for_context   - general live research for anything not
                               already in the knowledge base
"""

from __future__ import annotations

import logging
import time

import requests
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import tool

import config
import knowledge_store

logger = logging.getLogger(__name__)

_trends_cache: dict = {"ts": 0.0, "data": ""}
TRENDS_CACHE_TTL = 120  # seconds - avoid hammering Reddit during a demo

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the agent's own verified knowledge base for a meme, slang term,
    or internet-culture reference. ALWAYS call this first before doing any
    live web research — if we already have a good, verified answer there
    is no need to re-research it.

    Args:
        query: the meme name, slang term, or free-text question.

    Returns:
        The best-matching entries with a relevance score, or a clear
        statement that nothing strong was found.
    """
    if not query or not query.strip():
        return "No query provided."

    try:
        results = knowledge_store.search(query.strip(), k=config.RETRIEVER_K)
    except Exception as exc:
        logger.exception("Knowledge base search failed")
        return f"Knowledge base is unavailable right now ({exc}). Proceed to live research."

    if not results:
        return "NO_MATCH: nothing found in the knowledge base for this query."

    parts = []
    for doc, score in results:
        verified = doc.metadata.get("verified", False)
        parts.append(
            f"[relevance={score:.2f}, verified={verified}]\n{doc.page_content.strip()}"
        )
    return "\n\n---\n\n".join(parts)


@tool
def get_live_trending_topics(limit: int = config.TREND_LIMIT_DEFAULT) -> str:
    """
    Fetch currently trending meme post titles from Reddit (r/memes,
    r/IndianDankMemes, r/dankmemes, r/memetemplatesofficial). Use this to
    discover what is trending RIGHT NOW, e.g. for a digest or when
    grounding a marketing concept in current culture. Falls back to a
    curated cached list if Reddit is unreachable, so this never hard-fails.

    Args:
        limit: how many trending titles to return.

    Returns:
        Newline-separated "subreddit: title (score upvotes)" lines.
    """
    now = time.time()
    if _trends_cache["data"] and (now - _trends_cache["ts"]) < TRENDS_CACHE_TTL:
        return "\n".join(_trends_cache["data"].splitlines()[:limit])

    lines: list[str] = []
    failures = 0
    per_sub = max(2, limit // len(config.SUBREDDITS) + 1)

    for sub in config.SUBREDDITS:
        try:
            resp = requests.get(
                f"https://www.reddit.com/r/{sub}/hot.json",
                params={"limit": per_sub},
                headers=HEADERS,
                timeout=config.REDDIT_TIMEOUT,
            )
            resp.raise_for_status()
            posts = resp.json()["data"]["children"]
            for post in posts:
                d = post["data"]
                if d.get("stickied") or d.get("over_18"):
                    continue
                title = (d.get("title") or "").strip()
                if not title:
                    continue
                lines.append(f"r/{sub}: {title} ({d.get('score', 0)} upvotes)")
        except Exception as exc:
            failures += 1
            logger.warning("Reddit fetch failed for r/%s: %s", sub, exc)

    if failures == len(config.SUBREDDITS) or not lines:
        result = "\n".join(config.FALLBACK_TRENDS[:limit])
        logger.info("Using fallback trends")
    else:
        seen, unique = set(), []
        for line in lines:
            key = line.split(":", 1)[-1].strip().lower()[:80]
            if key not in seen:
                seen.add(key)
                unique.append(line)
        result = "\n".join(unique[:limit])
        _trends_cache["ts"] = now
        _trends_cache["data"] = result

    return result


@tool
def search_web_for_context(query: str) -> str:
    """
    Search the general web for factual context on a meme, slang term, or
    trend that is NOT already in the knowledge base. Use this to research
    origin, meaning, and history before drafting a new knowledge base
    entry. Returns multiple independent results so their agreement (or
    disagreement) can be assessed.

    Args:
        query: what to research, e.g. "origin of skibidi toilet meme".

    Returns:
        Up to a few results as "[domain] title -- snippet", or a message
        that no results were found.
    """
    if not query or not query.strip():
        return "No query provided."

    try:
        wrapper = DuckDuckGoSearchAPIWrapper()
        raw_results = wrapper.results(query.strip(), max_results=config.WEB_SEARCH_MAX_RESULTS)
    except Exception as exc:
        logger.exception("Web search failed")
        return f"WEB_SEARCH_FAILED: {exc}. Treat this topic as unresearched/low confidence."

    if not raw_results:
        return "NO_RESULTS: web search returned nothing for this query."

    # Preserve the exact URL from the search provider. The agent is instructed
    # to pass these URLs through unchanged so the UI can expose real evidence
    # links instead of domain-only labels.
    import json

    results = []
    for r in raw_results:
        link = (r.get("link") or "").strip()
        domain = link.split("/")[2] if "://" in link and len(link.split("/")) > 2 else link
        title = (r.get("title") or "").strip()
        snippet = (r.get("snippet") or "").strip()
        results.append({
            "title": title or "Source",
            "domain": domain,
            "url": link,
            "snippet": snippet,
        })
    return json.dumps({"results": results}, ensure_ascii=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("--- KB search ---")
    print(search_knowledge_base.invoke({"query": "aura farming"}))
    print("\n--- Trends ---")
    print(get_live_trending_topics.invoke({"limit": 5}))
    print("\n--- Web research ---")
    print(search_web_for_context.invoke({"query": "origin of skibidi toilet meme"}))

