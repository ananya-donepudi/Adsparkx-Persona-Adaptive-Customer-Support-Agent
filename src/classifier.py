"""Persona detection logic."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PersonaResult:
    tag: str
    confidence: float
    rationale: str


TECH_TERMS = {
    "api",
    "sdk",
    "endpoint",
    "http",
    "webhook",
    "payload",
    "token",
    "error code",
    "stack trace",
    "latency",
    "integration",
    "json",
}

FRUSTRATED_TERMS = {
    "angry",
    "annoyed",
    "frustrated",
    "again",
    "broken",
    "terrible",
    "unacceptable",
    "not working",
    "waste",
    "ridiculous",
    "urgent",
}

EXEC_TERMS = {
    "roi",
    "sla",
    "business impact",
    "timeline",
    "risk",
    "priority",
    "executive",
    "leadership",
    "contract",
    "account",
    "renewal",
}


def classify_persona(message: str) -> PersonaResult:
    """Classify a support message into Tech, Frustrated, or Exec."""
    text = message.lower()
    scores = {
        "Tech": _score_terms(text, TECH_TERMS),
        "Frustrated": _score_terms(text, FRUSTRATED_TERMS),
        "Exec": _score_terms(text, EXEC_TERMS),
    }

    if any(mark in message for mark in ("!!", "???")):
        scores["Frustrated"] += 1.5
    if len(message) < 90 and any(term in text for term in ("status", "eta", "impact")):
        scores["Exec"] += 1.0

    tag, score = max(scores.items(), key=lambda item: item[1])
    if score <= 0:
        return PersonaResult("Exec", 0.45, "Defaulted to concise business-style support.")

    total = sum(scores.values()) or 1.0
    confidence = min(0.95, 0.45 + (score / total) * 0.5)
    matched = _matched_terms(text, _terms_for(tag))
    rationale = f"Detected {tag.lower()} cues" + (f": {', '.join(matched[:3])}" if matched else ".")
    return PersonaResult(tag, round(confidence, 2), rationale)


def _score_terms(text: str, terms: set[str]) -> float:
    return sum(1.0 for term in terms if term in text)


def _matched_terms(text: str, terms: set[str]) -> list[str]:
    return [term for term in sorted(terms) if term in text]


def _terms_for(tag: str) -> set[str]:
    return {
        "Tech": TECH_TERMS,
        "Frustrated": FRUSTRATED_TERMS,
        "Exec": EXEC_TERMS,
    }[tag]
