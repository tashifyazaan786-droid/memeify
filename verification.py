"""
The verification gate.

Deliberately NOT another LLM call asked to "rate your own confidence 1-10"
— models are unreliable judges of their own accuracy, and for a live demo
that unreliability is exactly the kind of thing that fails in front of
judges. Instead this is a small set of deterministic, explainable rules
over things we can actually count: how many independent sources were
found, whether they agree, and whether the extracted content is
substantive rather than empty/hedged.

Honest limitation: there is no ground-truth oracle for meme folklore, so
"verified" here means "corroborated by multiple independent live sources
and internally consistent" — not "certified factually true." The agent
says this explicitly in its output; it does not claim more certainty than
this process can actually provide.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VerificationResult:
    tier: str  # "high" | "medium" | "low"
    reasoning: str
    source_count: int
    domains: list[str] = field(default_factory=list)


def _distinct_domains(sources) -> list[str]:
    seen = []

    for source in sources:
        if isinstance(source, dict):
            domain = source.get("domain", "")
        else:
            domain = source

        if isinstance(domain, str):
            domain = domain.strip().lower()

        if domain and domain not in seen:
            seen.append(domain)

    return seen
def score_confidence(
    *,
    origin: str,
    meaning: str,
    agreement: str,
    sources: list[str],
) -> VerificationResult:
    """
    agreement: the extraction step's own read of whether its sources agreed
        with each other -- one of "consistent" | "conflicting" | "insufficient".
    sources: raw source identifiers (domains/titles) the extraction step says
        it drew from.
    """
    domains = _distinct_domains(sources)
    source_count = len(domains)

    has_substantive_content = (
        bool(origin and origin.strip()) and bool(meaning and meaning.strip())
        and len(origin.strip()) >= 10 and len(meaning.strip()) >= 10
    )

    if not has_substantive_content:
        return VerificationResult(
            tier="low",
            reasoning="No substantive origin/meaning content was extracted — nothing solid to verify.",
            source_count=source_count,
            domains=domains,
        )

    if agreement == "conflicting":
        return VerificationResult(
            tier="low",
            reasoning="Sources disagreed with each other on core facts — held back rather than guessing.",
            source_count=source_count,
            domains=domains,
        )

    if agreement == "consistent" and source_count >= 2:
        return VerificationResult(
            tier="high",
            reasoning=f"{source_count} independent sources found and they agree on the core facts.",
            source_count=source_count,
            domains=domains,
        )

    if agreement == "consistent" and source_count == 1:
        return VerificationResult(
            tier="medium",
            reasoning="Only one independent source found — content looks coherent but isn't cross-corroborated yet.",
            source_count=source_count,
            domains=domains,
        )

    return VerificationResult(
        tier="medium" if source_count >= 1 else "low",
        reasoning="Evidence was thin or the extraction step wasn't confident in agreement across sources.",
        source_count=source_count,
        domains=domains,
    )
