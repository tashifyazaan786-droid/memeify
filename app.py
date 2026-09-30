
"""
Unified Gradio UI for the Meme Knowledge Agent.

Pipeline:
Research -> Verify -> Store -> Generate

Run:
python app.py
"""

from __future__ import annotations

import html
import os
import logging
from urllib.parse import urlparse

import gradio as gr

import agent
import config
import knowledge_store


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


HISTORIAN_EXAMPLES = [
    "What is aura farming and where did it come from?",
    "Explain the Ravi Kishan 'jaldi the late' meme",
    "What does 'bro thought he cooked' mean?",
    "What's the story behind Gucci Morty?",
]

MARKETING_EXAMPLES = [
    "A small chai and snacks stall near a college campus, open till midnight",
    "Affordable online coaching for Class 10-12 board exams, WhatsApp doubt support",
    "A local gym in a tier-2 city targeting beginners and working professionals",
]


def build_activity_panel(result: dict) -> str:
    """Create a judge-friendly, high-level agent activity panel."""

    activity = result.get("activity_log", [])
    pending = result.get("pending_review", [])

    lines = [
        "### 🔎 Agent Activity",
        "",
        "The agent exposes its **observable actions and decisions**, "
        "not private chain-of-thought.",
        "",
    ]

    if activity:
        for item in activity:
            text = str(item)
            lower = text.lower()

            if "knowledge base match" in lower:
                lines.append(f"🧠 **MEMORY RETRIEVAL** — {text}")

            elif "skipping live web research" in lower:
                lines.append(f"⚡ **EFFICIENCY** — {text}")

            elif "calling `search_knowledge_base" in lower:
                lines.append(f"🧠 **KNOWLEDGE SEARCH** — {text}")

            elif "calling `search_web_for_context" in lower:
                lines.append(f"🌐 **LIVE RESEARCH** — {text}")

            elif "verified high" in lower or "verified" in lower:
                lines.append(f"🛡️ **VERIFICATION** — {text}")

            elif "saved" in lower or "stored" in lower:
                lines.append(f"💾 **MEMORY UPDATE** — {text}")

            elif "research" in lower:
                lines.append(f"🔎 **RESEARCH** — {text}")

            else:
                lines.append(f"• {text}")
    else:
        lines.append("_No agent activity recorded._")

    lines.extend(
        [
            "",
            "---",
            "",
            "### 🛡️ Verification Gate",
            "",
        ]
    )

    if pending:
        lines.append(
            f"🟡 **{len(pending)} finding(s) awaiting human approval**"
        )
    else:
        lines.append("🟢 **No pending findings**")

    lines.extend(
        [
            "",
            "---",
            "",
            "### 🧠 Memory",
            "",
        ]
    )

    try:
        entries = knowledge_store.list_verified_entries()
        lines.append(
            f"📚 **{len(entries)} verified entries currently stored**"
        )
    except Exception:
        lines.append("📚 Knowledge-base count unavailable")

    return "\n".join(lines)

def build_sources_panel(result: dict) -> str:
    """Render only real source URLs exposed by the research pipeline."""
    candidates = list(result.get("candidates", [])) + list(result.get("pending_review", []))
    cards = []
    for candidate in candidates:
        for source in candidate.get("sources", []) or []:
            if not isinstance(source, dict):
                continue
            url = (source.get("url") or "").strip()
            if not url.startswith(("http://", "https://")):
                continue
            title = source.get("title") or "Source"
            domain = source.get("domain") or urlparse(url).netloc
            cards.append(
                '<a class="mk-source-card" href="' + html.escape(url, quote=True) + '" target="_blank" rel="noopener noreferrer">'
                '<div><div class="mk-source-kicker">SOURCE</div>'
                '<strong>' + html.escape(title) + '</strong>'
                '<span>' + html.escape(domain) + '</span></div>'
                '<span class="mk-source-open">↗</span></a>'
            )
    if not cards:
        return '<div class="mk-source-empty"><b>Evidence &amp; Sources</b><span>No research URLs were exposed for this run.</span></div>'
    return '<div class="mk-source-list">' + "".join(cards) + '</div>'


def build_image_analysis_panel(result: dict) -> str:
    analysis = result.get("image_analysis") or {}
    if not analysis:
        return ""
    confidence = str(analysis.get("confidence", "low")).upper()
    research = "Research required" if analysis.get("research_required", True) else "Research not required"
    rows = [
        ("Likely identification", analysis.get("likely_template") or "Unknown"),
        ("Text detected", analysis.get("text_detected") or "No legible text detected"),
        ("Meaning / context", analysis.get("meaning_context") or "Not established"),
        ("Origin / background", analysis.get("origin_background") or "Not established"),
        ("Common use", analysis.get("common_use") or "Not established"),
        ("Cultural context", analysis.get("cultural_context") or "Not established"),
    ]
    body = "".join('<div class="mk-analysis-row"><b>' + html.escape(label) + '</b><span>' + html.escape(str(value)) + '</span></div>' for label, value in rows)
    return ('<div class="mk-image-analysis"><div class="mk-image-analysis-head"><div><span>IMAGE ANALYSIS</span><h3>Best-effort identification</h3></div>'
            '<strong>' + html.escape(confidence) + ' · ' + html.escape(research) + '</strong></div>' + body + '</div>')


def build_marketing_summary(result: dict) -> str:
    """Show verified/researched marketing context without duplicating the answer."""
    if result.get("mode") != "marketing":
        return '<div class="mk-source-empty"><b>Marketing workspace</b><span>Select Marketing mode to generate a grounded campaign concept.</span></div>'
    candidates = result.get("candidates", []) or []
    if not candidates:
        return '<div class="mk-source-empty"><b>No grounded meme context yet</b><span>The agent did not find usable evidence for a marketing concept.</span></div>'
    rows = []
    for c in candidates:
        tier = str(c.get("tier", "unknown")).replace("_", " ").title()
        name = c.get("meme_name", "Unknown meme")
        meaning = c.get("meaning", "")
        rows.append(
            '<div class="mk-concept-row"><b>' + html.escape(str(name)) + '</b>'
            '<span><strong>' + html.escape(tier) + '</strong> · ' + html.escape(str(meaning)[:300]) + '</span></div>'
        )
    return '<div class="mk-marketing-card"><div class="mk-feature-label">Verified meme context</div>' + ''.join(rows) + '</div>'


def run(mode_label: str, query: str, language: str):
    logger.info("RUN INPUT TYPES: mode=%s, query=%s, language=%s",
                type(mode_label), type(query), type(language))

    if not isinstance(query, str):
        return (
            "⚠️ **Invalid topic input. Please enter your topic as text.**",
            "_Pipeline stopped._",
            "_No source information available._",
            [],
            gr.update(choices=[], value=None),
            gr.update(),
            "",
        )

    if not query.strip():
        return (
            "⚠️ **Enter a topic or business description first.**",
            "*No activity.*",
            "*No source information available.*",
            [],
            gr.update(choices=[], value=None),
            gr.update(),
            "",
        )

    mode = "marketing" if mode_label == "📢 Marketing" else "historian"

    try:
        result = agent.run_pipeline(
            query.strip(),
            mode=mode,
            language=language,
        )

    except RuntimeError as exc:
        return (
            f"⚠️ **{exc}**",
            "_Pipeline stopped._",
            "_No source information available._",
            [],
            gr.update(choices=[], value=None),
            gr.update(),
            "",
        )

    except Exception as exc:
        logger.exception("Pipeline failed")

        return (
            f"⚠️ **Something went wrong:** {exc}",
            "_Pipeline failed. Check the terminal for details._",
            "_No source information available._",
            [],
            gr.update(choices=[], value=None),
            gr.update(),
            "",
        )

    activity_md = build_activity_panel(result)
    sources_md = build_sources_panel(result)

    pending = result.get("pending_review", [])

    pending_names = [
        p.get("meme_name", "Unknown")
        for p in pending
    ]

    return (
        result.get("answer", "_No answer generated._"),
        activity_md,
        sources_md,
        pending,
        gr.update(
            choices=pending_names,
            value=pending_names[0] if pending_names else None,
        ),
        gr.update(),
        build_marketing_summary(result),
    )


def run_image(image_path: str, language: str):
    if not image_path:
        return "⚠️ Upload a meme image first.", "", "", [], gr.update(choices=[], value=None)
    try:
        result = agent.run_image_pipeline(image_path, language=language)
    except Exception as exc:
        logger.exception("Image pipeline failed")
        return f"⚠️ Image analysis failed: {exc}", "", "", [], gr.update(choices=[], value=None)
    pending = result.get("pending_review", [])
    names = [p.get("meme_name", "Unknown") for p in pending]
    return (result.get("answer", "_No answer generated._"), build_image_analysis_panel(result), build_sources_panel(result), pending, gr.update(choices=names, value=names[0] if names else None))


def approve_selected(
    selected_name: str,
    pending_state: list,
):
    if not selected_name:
        return "*Nothing selected to approve.*", gr.update()

    match = next(
        (
            p
            for p in pending_state
            if p.get("meme_name") == selected_name
        ),
        None,
    )

    if not match:
        return "*Couldn't find that draft anymore.*", gr.update()

    msg = agent.approve_and_save(match)

    remaining = [
        p
        for p in pending_state
        if p.get("meme_name") != selected_name
    ]

    remaining_names = [
        p.get("meme_name", "Unknown")
        for p in remaining
    ]

    return (
        msg,
        gr.update(
            choices=remaining_names,
            value=remaining_names[0] if remaining_names else None,
        ),
    )


def refresh_kb_view():
    try:
        entries = knowledge_store.list_verified_entries()
    except Exception as exc:
        logger.exception("Could not load knowledge base")
        return f"⚠️ Could not load knowledge base: {exc}"

    if not entries:
        return "*Knowledge base is empty.*"

    lines = []

    for entry in entries:
        verified = "✅" if entry.get("verified") else "🟡"

        lines.append(
            f"- **{entry.get('name', 'Unknown')}** "
            f"({entry.get('era', 'Unknown era')}) {verified}"
        )

    return (
        f"### 📚 {len(entries)} verified entries\n\n"
        + "\n".join(lines)
    )


CUSTOM_CSS = """
:root {
    --mk-bg: #0f1012; --mk-surface: #151619; --mk-surface-2: #1b1c20;
    --mk-border: rgba(255,255,255,.13); --mk-border-soft: rgba(255,255,255,.08);
    --mk-text: #f7f7f5; --mk-muted: #a5a6aa; --mk-dim: #777980;
    --mk-accent: #6f8cff; --mk-accent-hover: #8099ff; --mk-shadow: 0 24px 70px rgba(0,0,0,.28);
}
html, body, .gradio-container { background:var(--mk-bg)!important; color:var(--mk-text)!important; font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif!important; }
.gradio-container { max-width:100%!important; padding:0!important; } footer { display:none!important; }
/* LOCKED HEADER — keep these rules unchanged. */
#mk-header { margin:0!important; padding:0!important; }
.mk-header-shell { width:100%; border-bottom:1px solid var(--mk-border); background:radial-gradient(circle at 50% -20%,rgba(111,140,255,.10),transparent 42%),#101113; }
.mk-nav { height:68px; display:flex; align-items:center; gap:28px; padding:0 clamp(20px,4vw,64px); border-bottom:1px solid var(--mk-border-soft); }
.mk-brand { display:inline-flex; align-items:center; gap:12px; min-width:max-content; color:var(--mk-text); font-size:18px; font-weight:720; letter-spacing:-.03em; }
.mk-brand-name { white-space:nowrap; }
.mk-nav-links { display:flex; align-items:center; gap:26px; margin-left:auto; }
.mk-nav-link { color:#d2d3d6; font-size:13px; font-weight:560; text-decoration:none; opacity:.88; transition:color 180ms ease,opacity 180ms ease; }
.mk-nav-link:hover { color:#fff; opacity:1; }
.mk-nav-dot { width:5px; height:5px; border-radius:999px; background:#5b5d63; margin-left:-16px; }
.mk-nav-status { display:inline-flex; align-items:center; gap:7px; margin-left:4px; padding:7px 10px; border:1px solid var(--mk-border); border-radius:999px; color:#bfc1c6; font-size:11px; letter-spacing:.01em; }
.mk-live-dot { width:6px; height:6px; border-radius:50%; background:#8d94a3; box-shadow:0 0 0 3px rgba(141,148,163,.10); }
.mk-orb { position:relative; width:42px; height:42px; flex:0 0 42px; overflow:hidden; border-radius:50%; background:#151619; box-shadow:inset 0 0 0 1px rgba(255,255,255,.08),0 0 24px rgba(0,0,0,.28); isolation:isolate; }
.mk-orb::before { content:""; position:absolute; inset:-12%; border-radius:50%; background:repeating-linear-gradient(90deg,rgba(245,245,242,.86) 0 2px,rgba(245,245,242,.10) 2px 5px); opacity:.72; transform-origin:center; animation:mk-orb-rotate 5.2s cubic-bezier(.55,.05,.45,.95) infinite; }
.mk-orb::after { content:""; position:absolute; inset:0; border-radius:50%; background:radial-gradient(circle at 68% 32%,rgba(255,255,255,.86) 0 2px,transparent 2.5px),radial-gradient(circle at 48% 56%,rgba(255,255,255,.70) 0 2px,transparent 2.5px),radial-gradient(circle at 31% 44%,rgba(255,255,255,.62) 0 2px,transparent 2.5px); opacity:0; animation:mk-orb-points 5.2s ease-in-out infinite; }
@keyframes mk-orb-rotate { 0%,14%{transform:rotate(0) scale(1.03)} 28%,39%{transform:rotate(90deg) scale(1.03)} 53%,64%{transform:rotate(180deg) scale(1.03)} 78%,89%{transform:rotate(270deg) scale(1.03)} 100%{transform:rotate(360deg) scale(1.03)} }
@keyframes mk-orb-points { 0%,20%,82%,100%{opacity:0;transform:translateY(0)} 30%,52%{opacity:.95;transform:translateY(0)} 62%{opacity:.35;transform:translateY(1px)} }
.mk-hero { max-width:980px; margin:0 auto; padding:clamp(72px,9vw,118px) 24px clamp(58px,7vw,92px); text-align:center; }
.mk-eyebrow { display:inline-flex; align-items:center; gap:9px; margin-bottom:20px; color:#b9bbc1; font-size:12px; font-weight:650; letter-spacing:.10em; text-transform:uppercase; }
.mk-eyebrow-line { width:28px; height:1px; background:#555861; }
.mk-hero-title { margin:0; color:#f6f6f3; font-size:clamp(48px,8vw,94px); line-height:.98; letter-spacing:-.065em; font-weight:760; }
.mk-hero-title span { color:#aeb0b5; font-weight:660; }
.mk-hero-subtitle { max-width:700px; margin:24px auto 0; color:#b0b1b6; font-size:clamp(16px,2vw,19px); line-height:1.55; letter-spacing:-.015em; }
.mk-hero-actions { display:flex; justify-content:center; align-items:center; gap:12px; margin-top:32px; }
.mk-hero-note { margin-top:17px; color:#777980; font-size:11px; }
.mk-theme-toggle{width:44px!important;height:28px!important;min-width:44px!important;padding:0!important;border-radius:999px!important;border:1px solid var(--mk-border)!important;background:var(--mk-surface-2)!important;color:var(--mk-text)!important}.mk-theme-toggle::before{content:'☾';font-size:13px}
body.mk-light .mk-header-shell{background:radial-gradient(circle at 50% -20%,rgba(80,104,217,.08),transparent 42%),#f7f7f4}
body.mk-light .mk-nav-link{color:#44474e}
body.mk-light .mk-hero-title{color:#17181b}
body.mk-light .mk-hero-title span{color:#65686f}
body.mk-light .mk-hero-subtitle{color:#64676e}

@media (max-width:820px){
    .mk-nav{height:62px;gap:16px;padding:0 18px}
    .mk-nav-links{display:none}
    .mk-nav-status{margin-left:auto}
    .mk-hero{padding-top:64px;padding-bottom:54px}
    .mk-hero-title{font-size:clamp(46px,15vw,70px)}
    .mk-hero-subtitle{font-size:15px}
    .mk-hero-actions{flex-direction:column}
}

/* Everything below the locked header. */
#mk-main-content { width:min(1180px,calc(100% - 40px)); margin:0 auto; padding:76px 0 0; }
.mk-section { padding:0 0 78px; }
.mk-section + .mk-section { padding-top:76px; border-top:1px solid var(--mk-border-soft); }
.mk-section-kicker { color:var(--mk-dim); font-size:11px; font-weight:700; letter-spacing:.15em; text-transform:uppercase; margin-bottom:14px; }
.mk-section-title { margin:0; font-size:clamp(30px,4.5vw,56px); line-height:1.02; letter-spacing:-.055em; font-weight:700; color:var(--mk-text); }
.mk-section-copy { max-width:690px; margin:16px 0 0; color:var(--mk-muted); font-size:16px; line-height:1.7; }
.mk-surface { background:linear-gradient(180deg,rgba(255,255,255,.035),rgba(255,255,255,.015)); border:1px solid var(--mk-border); box-shadow:var(--mk-shadow); }
.mk-panel { border-radius:3px; padding:24px; margin-top:34px; }
.mk-panel-head { display:flex; align-items:flex-start; justify-content:space-between; gap:18px; padding-bottom:18px; margin-bottom:20px; border-bottom:1px solid var(--mk-border-soft); }
.mk-panel-title { margin:0; color:var(--mk-text); font-size:19px; font-weight:650; letter-spacing:-.025em; }
.mk-panel-meta { color:var(--mk-dim); font-size:11px; letter-spacing:.08em; text-transform:uppercase; }
#mk-main-content .block,#mk-main-content .form,#mk-main-content .panel,#mk-main-content .wrap { border-color:var(--mk-border)!important; box-shadow:none!important; }
#mk-main-content .label,#mk-main-content label,#mk-main-content .prose,#mk-main-content .markdown { color:var(--mk-text)!important; }
#mk-main-content textarea,#mk-main-content input,#mk-main-content select { background:var(--mk-surface)!important; color:var(--mk-text)!important; border-color:var(--mk-border)!important; border-radius:2px!important; }
#mk-main-content textarea::placeholder,#mk-main-content input::placeholder { color:var(--mk-dim)!important; }
#mk-main-content button { transition:transform 180ms ease,border-color 180ms ease,background 180ms ease; border-radius:2px!important; }
#mk-main-content button:hover { transform:translateY(-1px); }
#mk-run-agent { min-height:50px; border-radius:2px!important; box-shadow:0 14px 34px rgba(0,0,0,.20); }
.mk-mode-hint { color:var(--mk-dim); font-size:12px; margin-top:10px; }
.mk-pipeline { display:grid; grid-template-columns:repeat(7,1fr); border:1px solid var(--mk-border); background:var(--mk-surface); margin-top:18px; }
.mk-step { min-height:88px; padding:16px 12px; border-right:1px solid var(--mk-border-soft); }
.mk-step:last-child { border-right:0; }
.mk-step-num { color:var(--mk-accent); font-size:10px; letter-spacing:.12em; font-weight:700; }
.mk-step-name { margin-top:8px; font-size:13px; color:var(--mk-text); font-weight:620; }
.mk-step-desc { margin-top:4px; color:var(--mk-dim); font-size:10px; line-height:1.4; }
.mk-answer-layout { display:grid; grid-template-columns:minmax(0,1.35fr) minmax(290px,.65fr); gap:18px; margin-top:34px; }
.mk-output,.mk-activity { min-height:260px; margin-top:0; }
.mk-output .prose { font-size:15px!important; line-height:1.72!important; }
.mk-source-list { display:grid; gap:10px; margin-top:16px; }
.mk-source-card { display:flex; justify-content:space-between; align-items:center; gap:18px; padding:16px 17px; border:1px solid var(--mk-border); background:var(--mk-surface); color:var(--mk-text)!important; text-decoration:none!important; transition:transform 180ms ease,border-color 180ms ease,background 180ms ease; }
.mk-source-card:hover { transform:translateX(3px); border-color:rgba(71,121,238,.65); background:var(--mk-surface-2); }
.mk-source-kicker { color:var(--mk-accent); font-size:9px; font-weight:750; letter-spacing:.14em; margin-bottom:6px; }
.mk-source-card strong { display:block; font-size:14px; font-weight:620; }
.mk-source-card span:not(.mk-source-open) { display:block; margin-top:4px; color:var(--mk-dim); font-size:11px; }
.mk-source-open { color:var(--mk-accent); font-size:17px; }
.mk-source-empty { padding:24px 0; border-top:1px solid var(--mk-border-soft); color:var(--mk-muted); }
.mk-source-empty b { display:block; color:var(--mk-text); margin-bottom:5px; }
.mk-image-stage { display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-top:34px; }
.mk-image-analysis { border:1px solid var(--mk-border); background:var(--mk-surface); padding:20px; }
.mk-image-analysis-head { display:flex; justify-content:space-between; gap:16px; align-items:flex-start; padding-bottom:14px; border-bottom:1px solid var(--mk-border-soft); margin-bottom:8px; }
.mk-image-analysis-head span { color:var(--mk-dim); font-size:9px; letter-spacing:.13em; }
.mk-image-analysis-head h3 { margin:5px 0 0; font-size:19px; color:var(--mk-text); }
.mk-image-analysis-head strong { color:var(--mk-accent); font-size:10px; letter-spacing:.08em; text-transform:uppercase; }
.mk-analysis-row { display:grid; grid-template-columns:150px 1fr; gap:16px; padding:12px 0; border-bottom:1px solid var(--mk-border-soft); font-size:13px; line-height:1.55; }
.mk-analysis-row b { color:var(--mk-dim); font-weight:550; }
.mk-analysis-row span { color:var(--mk-text); }
.mk-marketing-grid,.mk-feature-grid { display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-top:34px; }
.mk-marketing-card,.mk-feature-card { border:1px solid var(--mk-border); background:var(--mk-surface); padding:22px; min-height:210px; }
.mk-feature-card h3 { margin:0 0 8px; font-size:18px; letter-spacing:-.025em; }
.mk-feature-card p { color:var(--mk-muted); font-size:13px; line-height:1.65; margin:0 0 18px; }
.mk-feature-label { color:var(--mk-dim); font-size:10px; letter-spacing:.13em; text-transform:uppercase; margin-bottom:10px; }
.mk-concept-row { display:grid; grid-template-columns:145px 1fr; gap:14px; padding:12px 0; border-bottom:1px solid var(--mk-border-soft); font-size:13px; line-height:1.55; }
.mk-concept-row b { color:var(--mk-dim); font-weight:550; }
.mk-concept-row span { color:var(--mk-text); }
.mk-approval { border-left:2px solid var(--mk-accent); }
.mk-knowledge-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:1px; border:1px solid var(--mk-border); background:var(--mk-border); margin-top:18px; }
.mk-knowledge-stat { background:var(--mk-surface); padding:24px; min-height:125px; }
.mk-knowledge-stat .value { font-size:32px; letter-spacing:-.05em; font-weight:680; }
.mk-knowledge-stat .label { margin-top:7px; color:var(--mk-dim); font-size:11px; text-transform:uppercase; letter-spacing:.11em; }
.mk-cta { margin-top:0; padding:70px clamp(24px,6vw,72px); text-align:center; border:1px solid var(--mk-border); background:repeating-linear-gradient(90deg,rgba(255,255,255,.035) 0 1px,transparent 1px 10px),linear-gradient(180deg,#171819,#111213); }
.mk-cta h2 { margin:0; font-size:clamp(32px,5vw,58px); line-height:1.02; letter-spacing:-.055em; }
.mk-cta p { max-width:650px; margin:16px auto 0; color:var(--mk-muted); line-height:1.65; }
.mk-cta-rule { width:64px; height:2px; background:var(--mk-accent); margin:24px auto 0; }
.mk-footer { border-top:1px solid var(--mk-border-soft); padding:28px 0 42px; display:flex; justify-content:space-between; gap:20px; color:var(--mk-dim); font-size:11px; }
.mk-footer strong { color:var(--mk-muted); font-weight:600; }
.gr-accordion { background:transparent!important; border-color:var(--mk-border)!important; }
body.mk-light { --mk-bg:#f5f5f2; --mk-surface:#fff; --mk-surface-2:#f0f1ed; --mk-border:rgba(20,22,25,.13); --mk-border-soft:rgba(20,22,25,.08); --mk-text:#17181b; --mk-muted:#65686f; --mk-dim:#76797c; --mk-accent:#5068d9; --mk-accent-hover:#4058c8; --mk-shadow:0 24px 70px rgba(20,22,25,.10); }
body.mk-light #mk-main-content textarea,body.mk-light #mk-main-content input,body.mk-light #mk-main-content select { background:#fff!important; color:#171819!important; }
body.mk-light #mk-main-content .label,body.mk-light #mk-main-content label,body.mk-light #mk-main-content .prose,body.mk-light #mk-main-content .markdown { color:#171819!important; }
body.mk-light .mk-cta { background:repeating-linear-gradient(90deg,rgba(20,22,25,.035) 0 1px,transparent 1px 10px),#f8f8f5; }
@media (max-width:900px) { #mk-main-content{width:min(100% - 28px,760px);padding-top:56px}.mk-answer-layout,.mk-marketing-grid,.mk-feature-grid,.mk-image-stage{grid-template-columns:1fr}.mk-pipeline{grid-template-columns:repeat(4,1fr)}.mk-step:nth-child(4){border-right:0}.mk-step:nth-child(n+5){border-top:1px solid var(--mk-border-soft)} }
@media (max-width:640px) { #mk-main-content{width:calc(100% - 20px);padding-top:40px}.mk-section,.mk-section + .mk-section{padding-bottom:52px;padding-top:52px}.mk-panel,.mk-feature-card,.mk-marketing-card{padding:17px}.mk-pipeline{grid-template-columns:repeat(2,1fr)}.mk-step{border-right:0;border-bottom:1px solid var(--mk-border-soft)}.mk-analysis-row,.mk-concept-row{grid-template-columns:1fr;gap:4px}.mk-knowledge-grid{grid-template-columns:1fr}.mk-footer{flex-direction:column}.mk-nav-links{display:none} }
@media (prefers-reduced-motion:reduce) { .mk-orb::before,.mk-orb::after{animation:none!important} #mk-main-content *{transition:none!important} }
"""

with gr.Blocks(
    title="Meme Knowledge Agent",
) as demo:
    # LOCKED HEADER — intentionally preserved.
    gr.HTML("""
        <header id="mk-header" class="mk-header-shell">
            <div class="mk-nav">
                <div class="mk-brand">
                    <span class="mk-orb" aria-hidden="true"></span>
                    <span class="mk-brand-name">Meme Knowledge Agent</span>
                </div>
                <nav class="mk-nav-links" aria-label="Product navigation">
                    <span class="mk-nav-link">How It Works</span><span class="mk-nav-dot"></span>
                    <span class="mk-nav-link">Why It Matters</span><span class="mk-nav-dot"></span>
                    <span class="mk-nav-link">Use Cases</span><span class="mk-nav-dot"></span>
                    <span class="mk-nav-link">Knowledge</span>
                </nav>
                <button id="mk-theme-toggle" class="mk-theme-toggle" aria-label="Toggle dark and light mode" title="Toggle theme"></button>
                <div class="mk-nav-status"><span class="mk-live-dot" aria-hidden="true"></span>Agent online</div>
            </div>
            <section class="mk-hero" aria-labelledby="mk-hero-title">
                <div class="mk-eyebrow"><span class="mk-eyebrow-line"></span>Agentic Internet Culture Intelligence<span class="mk-eyebrow-line"></span></div>
                <h1 id="mk-hero-title" class="mk-hero-title">Meme <span>Knowledge</span> Agent</h1>
                <p class="mk-hero-subtitle">A self-growing, evidence-backed agent that retrieves memory, researches live culture, verifies findings, and remembers what it learns.</p>
                <div class="mk-hero-actions"><span class="mk-hero-note">Research → Verify → Remember → Generate</span></div>
            </section>
        </header>
    """)
    pending_state = gr.State([])
    with gr.Column(elem_id="mk-main-content"):
        with gr.Column(elem_classes=["mk-section"]):
            gr.HTML("""
                <div class="mk-section-kicker">01 / Intelligence workspace</div>
                <h2 class="mk-section-title">Understand internet culture<br>without losing the trail.</h2>
                <p class="mk-section-copy">Ask about a meme, upload an image, or describe a business. The same evidence-backed system retrieves what it already knows, researches what is missing, verifies new findings, and turns the result into useful output.</p>
            """)
            with gr.Column(elem_classes=["mk-surface", "mk-panel"]):
                gr.HTML('<div class="mk-panel-head"><div><div class="mk-panel-meta">Agent workspace</div><h3 class="mk-panel-title">Start with a question or a visual</h3></div><div class="mk-panel-meta">Live · grounded</div></div>')
                with gr.Row():
                    with gr.Column(scale=2):
                        mode = gr.Radio(["📜 Historian", "📢 Marketing"], value="📜 Historian", label="Mode")
                    with gr.Column(scale=1):
                        language = gr.Dropdown(choices=list(config.LANGUAGE_INSTRUCTIONS.keys()), value="English", label="Response language")
                query = gr.Textbox(label="Question or marketing brief", placeholder=HISTORIAN_EXAMPLES[0], lines=4)
                gr.Examples(HISTORIAN_EXAMPLES, inputs=[query], label="Historian prompts")
                gr.Examples(MARKETING_EXAMPLES, inputs=[query], label="Marketing briefs")
                submit_btn = gr.Button("Run Agent", variant="primary", size="lg", elem_id="mk-run-agent")
                gr.HTML('<div class="mk-mode-hint">Historian explains. Marketing turns verified meme context into grounded campaign concepts.</div>')
            gr.HTML("""
                <div class="mk-pipeline" aria-label="Agent pipeline">
                    <div class="mk-step"><div class="mk-step-num">01</div><div class="mk-step-name">Research</div><div class="mk-step-desc">Find live evidence.</div></div>
                    <div class="mk-step"><div class="mk-step-num">02</div><div class="mk-step-name">Understand</div><div class="mk-step-desc">Interpret intent or image.</div></div>
                    <div class="mk-step"><div class="mk-step-num">03</div><div class="mk-step-name">Retrieve</div><div class="mk-step-desc">Search verified memory.</div></div>
                    <div class="mk-step"><div class="mk-step-num">04</div><div class="mk-step-name">Analyze</div><div class="mk-step-desc">Compare context.</div></div>
                    <div class="mk-step"><div class="mk-step-num">05</div><div class="mk-step-name">Verify</div><div class="mk-step-desc">Gate new claims.</div></div>
                    <div class="mk-step"><div class="mk-step-num">06</div><div class="mk-step-name">Store</div><div class="mk-step-desc">Save only verified facts.</div></div>
                    <div class="mk-step"><div class="mk-step-num">07</div><div class="mk-step-name">Generate</div><div class="mk-step-desc">Produce the answer.</div></div>
                </div>
            """)
        with gr.Column(elem_classes=["mk-section"]):
            gr.HTML('<div class="mk-section-kicker">02 / Visual understanding</div><h2 class="mk-section-title">Bring the meme itself.</h2><p class="mk-section-copy">Upload a meme image when the text alone is not enough. The model makes a best-effort identification, extracts visible text, then feeds the result into the existing retrieval, research, and verification pipeline.</p>')
            with gr.Column(elem_classes=["mk-surface", "mk-panel"]):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(label="Meme image", type="filepath", sources=["upload", "clipboard"])
                        with gr.Row():
                            analyze_image_btn = gr.Button("Analyze Meme", variant="primary")
                            clear_image_btn = gr.Button("Remove", variant="secondary")
                    with gr.Column(scale=1):
                        image_analysis_out = gr.HTML('<div class="mk-source-empty"><b>Image understanding</b><span>Upload an image and run analysis to see a likely template, detected text, context, confidence, and research status.</span></div>')
        with gr.Column(elem_classes=["mk-section"]):
            gr.HTML('<div class="mk-section-kicker">03 / Grounded output</div><h2 class="mk-section-title">Evidence first. Answer second.</h2><p class="mk-section-copy">The output area keeps the generated answer beside observable activity and real research sources, so the provenance of a claim stays visible.</p>')
            with gr.Row(elem_classes=["mk-answer-layout"]):
                with gr.Column(elem_classes=["mk-surface", "mk-panel", "mk-output"]):
                    gr.HTML('<div class="mk-panel-head"><div><div class="mk-panel-meta">Generated intelligence</div><h3 class="mk-panel-title">Answer</h3></div></div>')
                    answer_out = gr.Markdown(value="_Your grounded answer will appear here._")
                with gr.Column(elem_classes=["mk-surface", "mk-panel", "mk-activity"]):
                    gr.HTML('<div class="mk-panel-head"><div><div class="mk-panel-meta">Observable decisions</div><h3 class="mk-panel-title">Agent activity</h3></div></div>')
                    activity_out = gr.Markdown(value="_Run the agent to see retrieval, research, verification, and storage activity._")
            with gr.Column(elem_classes=["mk-surface", "mk-panel"]):
                gr.HTML('<div class="mk-panel-head"><div><div class="mk-panel-meta">Traceable evidence</div><h3 class="mk-panel-title">Sources</h3></div></div>')
                sources_out = gr.HTML(value='<div class="mk-source-empty"><b>No sources yet</b><span>Real source URLs will appear here after research.</span></div>')
        with gr.Column(elem_classes=["mk-section"]):
            gr.HTML('<div class="mk-section-kicker">04 / Marketing intelligence</div><h2 class="mk-section-title">Turn verified culture into a campaign idea.</h2><p class="mk-section-copy">Marketing mode uses the same knowledge retrieval, live research, and deterministic verification gate. It produces grounded meme recommendations and copy without requiring a paid image-generation service.</p>')
            with gr.Row(elem_classes=["mk-marketing-grid"]):
                with gr.Column(elem_classes=["mk-surface", "mk-panel"]):
                    gr.HTML('<div class="mk-panel-head"><div><div class="mk-panel-meta">Creative brief</div><h3 class="mk-panel-title">Marketing workspace</h3></div><div class="mk-panel-meta">Text only · verified</div></div>')
                    gr.Markdown("Use the Marketing mode above, then click **Run Agent**. The generated answer contains the recommended meme/template, why it fits, caption/copy, target audience, and platform suggestions.")
                with gr.Column(elem_classes=["mk-surface", "mk-panel"]):
                    gr.HTML('<div class="mk-panel-head"><div><div class="mk-panel-meta">Campaign intelligence</div><h3 class="mk-panel-title">Grounded concept</h3></div></div>')
                    marketing_concept_out = gr.HTML('<div class="mk-source-empty"><b>Marketing output will appear in the Answer panel</b><span>Evidence and source cards remain available above.</span></div>')
        with gr.Column(elem_classes=["mk-section"]):
            gr.HTML('<div class="mk-section-kicker">05 / Trust & memory</div><h2 class="mk-section-title">A memory that earns its way into storage.</h2><p class="mk-section-copy">New claims are not written by the model directly. The deterministic verification gate decides whether evidence is strong enough to store, or whether a human needs to approve the draft.</p>')
            with gr.Row(elem_classes=["mk-feature-grid"]):
                with gr.Column(elem_classes=["mk-surface", "mk-feature-card", "mk-approval"]):
                    gr.HTML('<div class="mk-feature-label">Human approval</div><h3>Drafts stay drafts.</h3><p>Medium-confidence findings can be used with a caveat, but they remain outside permanent memory until you explicitly approve them.</p>')
                    pending_dropdown = gr.Dropdown(choices=[], label="Pending drafts", interactive=True)
                    approve_btn = gr.Button("Approve & Save Draft")
                    approve_status = gr.Markdown()
                with gr.Column(elem_classes=["mk-surface", "mk-feature-card"]):
                    gr.HTML('<div class="mk-feature-label">Persistent knowledge</div><h3>Searchable, verified, growing.</h3><p>The source-of-truth markdown file and Chroma vector store preserve verified entries for future retrieval.</p>')
                    kb_view = gr.Markdown(value="_Click refresh to inspect the current memory._")
                    refresh_btn = gr.Button("Refresh Knowledge")
            gr.HTML('<div class="mk-knowledge-grid"><div class="mk-knowledge-stat"><div class="value">Verified</div><div class="label">Permanent memory gate</div></div><div class="mk-knowledge-stat"><div class="value">Chroma</div><div class="label">Semantic retrieval</div></div><div class="mk-knowledge-stat"><div class="value">Live</div><div class="label">Research when memory is thin</div></div></div>')
        with gr.Column(elem_classes=["mk-section"]):
            gr.HTML('<div class="mk-cta"><div class="mk-section-kicker">Keep the context connected</div><h2>Move from a meme<br>to an evidence trail.</h2><p>One workspace for remembering internet culture, researching what changed, and turning verified context into useful explanations and campaign ideas.</p><div class="mk-cta-rule"></div></div>')
            gr.HTML('<footer class="mk-footer"><div><strong>Meme Knowledge Agent</strong><br>Research · Verify · Remember · Generate</div><div>Knowledge base: <code>data/memes.md</code><br>Powered by Groq · LangChain · Chroma · Gradio</div></footer>')

        submit_btn.click(fn=run, inputs=[mode, query, language], outputs=[answer_out, activity_out, sources_out, pending_state, pending_dropdown, image_analysis_out, marketing_concept_out])
        analyze_image_btn.click(fn=run_image, inputs=[image_input, language], outputs=[answer_out, image_analysis_out, sources_out, pending_state, pending_dropdown])
        clear_image_btn.click(fn=lambda: (None, '<div class="mk-source-empty"><b>Image cleared</b><span>Upload another meme image when ready.</span></div>'), outputs=[image_input, image_analysis_out])
        refresh_btn.click(fn=refresh_kb_view, outputs=[kb_view])
        approve_btn.click(fn=approve_selected, inputs=[pending_dropdown, pending_state], outputs=[approve_status, pending_dropdown]).then(fn=refresh_kb_view, outputs=[kb_view])

    demo.load(fn=None, js="""() => { const saved=localStorage.getItem('mk-theme'); if(saved==='light') document.body.classList.add('mk-light'); }""")
    demo.load(fn=None, js="""() => { const btn=document.getElementById('mk-theme-toggle'); if(!btn||btn.dataset.bound)return; btn.dataset.bound='1'; btn.addEventListener('click',()=>{const light=document.body.classList.toggle('mk-light'); localStorage.setItem('mk-theme',light?'light':'dark');}); }""")

if __name__ == "__main__":
    import socket

    def _find_free_port(start=7860, end=7870):
        for candidate in range(start, end + 1):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    sock.bind(("127.0.0.1", candidate))
                    return candidate
                except OSError:
                    continue
        raise OSError(f"No free local port found in range {start}-{end}.")

    configured_port = int(os.environ.get("GRADIO_SERVER_PORT", "0"))
    port = configured_port if configured_port > 0 else _find_free_port()
    print(f"Meme Knowledge Agent running at http://127.0.0.1:{port}")
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=port,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate", neutral_hue="slate"),
        css=CUSTOM_CSS,
    )
