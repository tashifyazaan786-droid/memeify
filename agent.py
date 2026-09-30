"""
The orchestrator. Two phases, deliberately kept separate:

PHASE 1 (agentic, non-deterministic) - the LLM autonomously decides which
tools to call and in what order (check memory -> research live -> check
trends) via langchain's `create_agent`. This is where genuine agentic
behaviour lives: intent understanding, planning, tool selection.

PHASE 2 (deterministic, code-controlled) - parsing the agent's structured
output, scoring confidence with verification.py, and deciding what gets
persisted. This is deliberately NOT left to the LLM, for reliability and
safety (see knowledge_store.py's docstring for why).

A third, plain (non-agentic) LLM call handles final answer generation,
grounded only in what phase 2 approved.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import logging
import re

import requests
from typing import Any

from langchain.agents import create_agent
from langchain_groq import ChatGroq

import config
import knowledge_store
import verification
from research_tools import get_live_trending_topics, search_knowledge_base, search_web_for_context

logger = logging.getLogger(__name__)

TOOLS = [search_knowledge_base, search_web_for_context, get_live_trending_topics]

RESEARCH_SYSTEM_PROMPT = """You are the research module of a meme & internet-culture
knowledge agent. Your job is ONLY to gather and structure evidence — you do not
write the final user-facing answer.

Process, every time:
1. ALWAYS call search_knowledge_base first for any specific meme/topic you're
   investigating. If it returns a strong, verified match, you can reuse that
   content directly (set already_verified_in_kb=true) without further research.
2. If the knowledge base has no strong match, call search_web_for_context to
   research it live. Use a focused query (the meme/slang name + "meme origin
   meaning").
3. For MARKETING mode only: call get_live_trending_topics first to see what's
   currently trending, then pick 1-3 trends that plausibly connect to the
   business, and research each with search_knowledge_base /
   search_web_for_context.
4. Assess whether your sources agree with each other. Never invent or smooth
   over disagreement — report it honestly.
5. When using web research, preserve the actual URLs returned by
   search_web_for_context. Do NOT reduce sources to domains only.
6. Never invent, guess, or construct URLs. Only include URLs that actually
   appeared in the tool results.

At the very end of your response, output ONLY a fenced json code block
(nothing after it) containing a JSON array. HISTORIAN mode: exactly one
object about the requested topic. MARKETING mode: one object per candidate
trend (1-3 total).

Each object must have exactly these fields:

```json
[
  {
    "meme_name": "string",
    "aliases": "string or empty",
    "era": "string or empty",
    "origin": "1-3 sentences, empty string if truly nothing found",
    "meaning": "1-3 sentences, empty string if truly nothing found",
    "already_verified_in_kb": true or false,
    "agreement": "consistent" or "conflicting" or "insufficient",
    "sources": [
      {
        "title": "exact source title from tool result",
        "domain": "source domain",
        "url": "exact URL returned by tool"
      }
    ]
  }
]
If already_verified_in_kb is true, "sources" can be ["existing knowledge base"]
and agreement should be "consistent". Do not fabricate sources you didn't
actually see in tool results.
"""

GENERATION_SYSTEM_PROMPT_HISTORIAN = """You are the Meme Historian, a friendly,
sharp-witted expert on internet culture (global and Indian). You have been
given already-researched, verified context — do not contradict it and do not
invent facts beyond it. Always mention the confidence level and (briefly) what
it's based on, in plain language a normal user understands (not raw scores).
If confidence is low/nothing solid was found, say so honestly instead of
guessing. Keep it tight: a few short paragraphs, not an essay. Follow the
requested response language exactly."""

GENERATION_SYSTEM_PROMPT_MARKETING = """You are the Meme Marketing Agent. Turn a
business description into 2-3 ready-to-post ad concepts, built ONLY around the
verified/researched meme context you are given below — never a meme you
weren't given context for. For EACH concept use exactly this structure:

**Suggested Meme:** <name>  (confidence: <level>)
**Why it fits:** <1-2 sentences linking the meme's meaning/vibe to the business>
**Caption:** <ready-to-post caption, witty, platform-native>
**Reel/Shorts Script:** <3-5 short beats, easy to film on a phone>
**Hashtags:** <5-8 relevant hashtags>
**Best Platforms:** <e.g. Instagram Reels, YouTube Shorts, WhatsApp Status>

If a concept is based on medium-confidence context, add one honest caveat
sentence after "Why it fits" noting it's a newer/less-corroborated reference.
Never use a meme marked low confidence. Be specific to the business, not
generic. Follow the requested response language exactly."""


IMAGE_ANALYSIS_PROMPT = """You are the image-understanding stage of a meme knowledge agent.
Analyze the supplied meme image conservatively. Do not claim certainty. Return ONLY
valid JSON with these keys: likely_template, text_detected, visual_description,
meaning_context, origin_background, common_use, cultural_context, confidence,
research_required, research_query.

Rules:
- likely_template must be a best-effort identification, or "Unknown".
- text_detected should contain visible text you can actually read; use an empty
  string if none is legible.
- confidence must be one of high, medium, low.
- research_required must be true when the identification/origin is uncertain.
- Never invent an origin.
- If the image is not clearly a meme, say so in visual_description and use
  Unknown/low/research_required=true.
"""

def _parse_json_object(text: str) -> dict:
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    raw = match.group(1) if match else None
    if raw is None:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        raw = match.group(1) if match else None
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _image_data_url(image_path: str) -> str:
    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/png"
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def analyze_meme_image(image_path: str) -> dict[str, Any]:
    """Best-effort multimodal analysis; it never writes to the knowledge base."""
    if not image_path:
        return {"error": "No image was supplied."}

    api_key = config.get_groq_api_key()
    payload = {
        "model": config.GROQ_VISION_MODEL,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": IMAGE_ANALYSIS_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this meme image."},
                    {"type": "image_url", "image_url": {"url": _image_data_url(image_path)}},
                ],
            },
        ],
    }
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=90,
    )
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    analysis = _parse_json_object(content)
    if not analysis:
        raise RuntimeError("The image-analysis model returned an unreadable response.")

    analysis.setdefault("likely_template", "Unknown")
    analysis.setdefault("text_detected", "")
    analysis.setdefault("visual_description", "")
    analysis.setdefault("meaning_context", "")
    analysis.setdefault("origin_background", "")
    analysis.setdefault("common_use", "")
    analysis.setdefault("cultural_context", "")
    analysis.setdefault("confidence", "low")
    analysis.setdefault("research_required", True)
    analysis.setdefault("research_query", analysis.get("likely_template", "Unknown"))
    return analysis


def _sources_from_doc(page_content: str) -> list[dict]:
    """Recover only URLs that were actually stored in a verified KB entry."""
    sources = []
    match = re.search(r"\*\*Sources:\*\*\s*(.+)", page_content or "")
    if not match:
        return sources
    for part in match.group(1).split(";"):
        pieces = [p.strip() for p in part.split("|")]
        if len(pieces) >= 3 and pieces[-1].startswith(("http://", "https://")):
            sources.append({"title": pieces[0], "domain": pieces[1], "url": pieces[-1]})
    return sources



def _get_llm(model: str) -> ChatGroq:
    return ChatGroq(
        model=model,
        temperature=config.LLM_TEMPERATURE,
        api_key=config.get_groq_api_key(),
        max_retries=2,
    )


def get_research_agent():
    llm = _get_llm(config.GROQ_MODEL)
    return create_agent(model=llm, tools=TOOLS, system_prompt=RESEARCH_SYSTEM_PROMPT)


def _extract_activity_log(messages: list) -> list[str]:
    """Turn the agent's raw message trace into short, user-facing status
    lines -- tool calls and result previews, never raw model reasoning."""
    log: list[str] = []
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                name = tc.get("name", "tool")
                args = tc.get("args", {})
                arg_preview = ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:2])
                log.append(f"🔎 Calling `{name}`({arg_preview})")
            continue
        msg_type = getattr(msg, "type", "")
        if msg_type == "tool":
            content = getattr(msg, "content", "") or ""
            preview = content.strip().replace("\n", " ")[:140]
            log.append(f"   ↳ {preview}{'...' if len(content) > 140 else ''}")
    return log


def _extract_json_array(text: str) -> list[dict]:
    match = re.search(r"```json\s*(\[.*?\])\s*```", text, re.DOTALL)
    raw = match.group(1) if match else None
    if raw is None:
        match = re.search(r"(\[.*\])", text, re.DOTALL)
        raw = match.group(1) if match else None
    if raw is None:
        logger.warning("No JSON array found in agent output")
        return []
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON from agent output: %s", raw[:200])
        return []


def _run_research(user_prompt: str) -> tuple[list[dict], list[str]]:
    agent = get_research_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
    messages = result.get("messages", [])
    activity_log = _extract_activity_log(messages)

    final_text = ""
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "ai" and getattr(msg, "content", ""):
            final_text = msg.content
            break

    candidates = _extract_json_array(final_text)
    return candidates, activity_log


def _verify_and_gate(candidates: list[dict]) -> tuple[list[dict], list[dict], list[str]]:
    """Returns (usable_for_generation, pending_review, extra_log_lines)."""
    usable: list[dict] = []
    pending: list[dict] = []
    log_lines: list[str] = []

    for c in candidates:
        name = c.get("meme_name", "Unknown")

        if c.get("already_verified_in_kb"):
            c["tier"] = "existing_verified"
            usable.append(c)
            log_lines.append(f"✅ '{name}' already verified in knowledge base — reused directly.")
            continue

        result = verification.score_confidence(
            origin=c.get("origin", ""),
            meaning=c.get("meaning", ""),
            agreement=c.get("agreement", "insufficient"),
            sources=c.get("sources", []),
        )
        c["tier"] = result.tier
        c["verification_reasoning"] = result.reasoning
        c["source_count"] = result.source_count

        if result.tier == "high":
            try:
                knowledge_store.add_verified_entry(
                    meme_name=name,
                    meaning=c.get("meaning", ""),
                    origin=c.get("origin", ""),
                    aliases=c.get("aliases", ""),
                    era=c.get("era", ""),
                    sources=c.get("sources", []),
                    confidence="high",
                )
                log_lines.append(
                    f"✅ '{name}' verified HIGH confidence ({result.reasoning}) — auto-saved to knowledge base."
                )
            except Exception as exc:
                logger.exception("Failed to auto-save verified entry")
                log_lines.append(f"⚠️ '{name}' verified HIGH but saving failed: {exc}")
            usable.append(c)
        elif result.tier == "medium":
            log_lines.append(
                f"🟡 '{name}' verified MEDIUM confidence ({result.reasoning}) — held as draft, needs your approval to save."
            )
            pending.append(c)
            usable.append(c)  # medium can still be used in generation, with a caveat
        else:
            log_lines.append(
                f"🔴 '{name}' LOW confidence ({result.reasoning}) — not saved, not used in output."
            )
            pending.append(c)

    return usable, pending, log_lines


def _generate_historian_answer(topic: str, usable: list[dict], language: str) -> str:
    llm = _get_llm(config.GROQ_MODEL)
    lang_instruction = config.LANGUAGE_INSTRUCTIONS.get(language, config.LANGUAGE_INSTRUCTIONS["English"])
    if not usable:
        context = "No verified or corroborated context was found for this topic."
    else:
        context = json.dumps(usable, indent=2)

    prompt = (
        f"User asked about: {topic}\n\n"
        f"Verified research context (JSON):\n{context}\n\n"
        f"({lang_instruction})"
    )
    response = llm.invoke(
        [
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT_HISTORIAN},
            {"role": "user", "content": prompt},
        ]
    )
    return response.content


def _generate_marketing_answer(business_description: str, usable: list[dict], language: str) -> str:
    llm = _get_llm(config.GROQ_MODEL)
    lang_instruction = config.LANGUAGE_INSTRUCTIONS.get(language, config.LANGUAGE_INSTRUCTIONS["English"])
    grounded = [c for c in usable if c.get("tier") != "low"]

    if not grounded:
        return (
            "⚠️ No trend could be verified with enough confidence to safely ground an ad "
            "concept right now. Try again in a bit, or describe the business differently "
            "so I can search more specifically."
        )

    context = json.dumps(grounded, indent=2)
    prompt = (
        f"Business description: {business_description}\n\n"
        f"Verified/researched trend context (JSON):\n{context}\n\n"
        f"({lang_instruction})"
    )
    response = llm.invoke(
        [
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT_MARKETING},
            {"role": "user", "content": prompt},
        ]
    )
    return response.content


def run_pipeline(
    user_input: str,
    mode: str,
    language: str = "English",
) -> dict[str, Any]:
    """Run the meme research pipeline with a KB-first strategy."""

    # 1. Search our verified knowledge base first.
    try:
        kb_results = knowledge_store.search(
            user_input.strip(),
            k=config.RETRIEVER_K,
        )
    except Exception as exc:
        logger.exception("Knowledge base lookup failed")
        kb_results = []
        logger.warning("Continuing to live research: %s", exc)

    verified_matches = []

    for doc, score in kb_results:
        verified = doc.metadata.get("verified", False)

        if verified and score >= config.KB_MATCH_SCORE_THRESHOLD:
            verified_matches.append((doc, score))

    # 2. Strong verified match -> skip web research.
    if verified_matches:
        doc, score = verified_matches[0]

        name = doc.metadata.get("meme_name", user_input)

        candidate = {
            "meme_name": name,
            "meaning": doc.page_content,
            "origin": "",
            "aliases": "",
            "era": doc.metadata.get("era", "Unknown"),
            "sources": _sources_from_doc(doc.page_content),
            "agreement": "existing_verified",
            "already_verified_in_kb": True,
            "tier": "existing_verified",
        }

        activity_log = [
            f"🧠 Knowledge base match: '{name}' "
            f"(relevance={score:.2f})",
            f"✅ '{name}' is already verified — "
            "skipping live web research.",
        ]

        usable = [candidate]
        pending = []

    # 3. No strong match -> research the internet.
    else:
        if mode == "marketing":
            user_prompt = (
                f"Mode: MARKETING\nBusiness description: {user_input}\n\n"
                "Find 1-3 currently trending memes/topics that plausibly "
                "connect to this business, research each, and follow the "
                "required JSON output format."
            )
        else:
            user_prompt = (
                f"Mode: HISTORIAN\nTopic: {user_input}\n\n"
                "Research this one topic and follow the required JSON "
                "output format (a single-item array)."
            )

        candidates, activity_log = _run_research(user_prompt)

        if not candidates:
            activity_log.append(
                "Could not extract structured findings from research — "
                "treating as unverified."
            )

        usable, pending, gate_log = _verify_and_gate(candidates)
        activity_log.extend(gate_log)

    # 4. Generate the final answer.
    if mode == "marketing":
        answer = _generate_marketing_answer(
            user_input,
            usable,
            language,
        )
    else:
        answer = _generate_historian_answer(
            user_input,
            usable,
            language,
        )

    return {
        "activity_log": activity_log,
        "answer": answer,
        "pending_review": pending,
        "mode": mode,
        "candidates": usable,
    }

def run_image_pipeline(image_path: str, language: str = "English") -> dict[str, Any]:
    """Image -> understand -> existing KB/web research -> verify -> generate."""
    analysis = analyze_meme_image(image_path)
    if analysis.get("error"):
        return {"analysis": analysis, "activity_log": [f"⚠️ Image analysis failed: {analysis['error']}"], "answer": "", "candidates": [], "pending_review": []}

    template = str(analysis.get("likely_template") or "Unknown")
    detected = str(analysis.get("text_detected") or "")
    research_query = str(analysis.get("research_query") or template or detected or "meme image")
    prompt = (
        f"Image-derived meme identification: {template}.\n"
        f"Visible text: {detected}\n"
        f"Visual description: {analysis.get('visual_description', '')}\n"
        f"Meaning/context hypothesis: {analysis.get('meaning_context', '')}\n"
        f"Image confidence: {analysis.get('confidence', 'low')}.\n"
        f"Research required: {analysis.get('research_required', True)}.\n"
        f"Research query: {research_query}"
    )
    result = run_pipeline(prompt, mode="historian", language=language)
    result["image_analysis"] = analysis
    result["activity_log"] = ["🖼️ Image understood (best-effort; not treated as certain)."] + result.get("activity_log", [])
    return result


def approve_and_save(entry: dict) -> str:
    """Called when the user clicks 'Approve' on a medium/low-confidence draft."""
    try:
        knowledge_store.add_verified_entry(
            meme_name=entry.get("meme_name", "Unknown"),
            meaning=entry.get("meaning", ""),
            origin=entry.get("origin", ""),
            aliases=entry.get("aliases", ""),
            era=entry.get("era", ""),
            sources=entry.get("sources", []),
            confidence=f"{entry.get('tier', 'medium')} (human-approved)",
        )
        return f"✅ Saved '{entry.get('meme_name')}' to the knowledge base."
    except Exception as exc:
        logger.exception("Manual save failed")
        return f"⚠️ Failed to save: {exc}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    user_input = input("Enter your meme question: ").strip()

    if not user_input:
        print("Please enter a question.")
        raise SystemExit(1)

    result = run_pipeline(user_input, mode="historian")

    for line in result["activity_log"]:
        print(line)

    print("\n--- ANSWER ---\n")
    print(result["answer"])

    if result.get("pending_review"):
        print("\n--- PENDING REVIEW ---\n")
        for item in result["pending_review"]:
            print(item)