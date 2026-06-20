"""Confidence thresholds and escalation handoff generation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from src.config import AppConfig, SENSITIVE_KEYWORDS


def should_escalate(
    message: str,
    persona_tag: str,
    retrieval_score: float,
    config: AppConfig,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    text = message.lower()

    if retrieval_score < config.escalation_score:
        reasons.append("retrieval_confidence_low")
    if any(keyword in text for keyword in SENSITIVE_KEYWORDS):
        reasons.append("sensitive_issue")
    if persona_tag == "Frustrated" and any(term in text for term in ("cancel", "refund", "lawsuit")):
        reasons.append("high_emotion_sensitive_request")

    return bool(reasons), reasons


def retrieval_is_sufficient(retrieval_score: float, config: AppConfig) -> bool:
    return retrieval_score >= config.min_retrieval_score


def generate_handoff_json(
    message: str,
    persona_tag: str,
    reasons: list[str],
    retrieved_chunks: list[dict[str, Any]],
) -> str:
    payload = {
        "handoff_type": "human_agent",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "persona": persona_tag,
        "priority": "high" if "sensitive_issue" in reasons else "normal",
        "reasons": reasons,
        "customer_message": message,
        "retrieval_evidence": [
            {
                "source": chunk.get("source"),
                "score": round(float(chunk.get("score", 0.0)), 3),
                "preview": chunk.get("text", "")[:260],
            }
            for chunk in retrieved_chunks
        ],
    }
    return json.dumps(payload, indent=2)
