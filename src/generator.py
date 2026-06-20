"""Persona-based prompt compiler and LLM caller."""

from __future__ import annotations

from typing import Any

from src.config import AppConfig


PERSONA_STYLE = {
    "Tech": "Use precise technical language, include concrete steps, and cite config or API details when relevant.",
    "Frustrated": "Start with empathy, acknowledge friction directly, keep steps short, and avoid sounding defensive.",
    "Exec": "Be concise, lead with impact and decision-ready next steps, and avoid implementation noise unless requested.",
}


def generate_adaptive_response(
    message: str,
    persona_tag: str,
    chunks: list[dict[str, Any]],
    config: AppConfig,
) -> str:
    """Generate a persona-adaptive response.

    Gemini is used when available. If the API key, internet access, or SDK call fails,
    the app falls back to a local retrieved-context response so the assignment demo
    remains stable.
    """
    prompt = compile_prompt(message, persona_tag, chunks, config)
    if config.gemini_api_key:
        try:
            from google import genai

            client = genai.Client(api_key=config.gemini_api_key)
            response = client.models.generate_content(
                model=config.chat_model,
                contents=prompt,
            )
            if response.text:
                return response.text.strip()
        except Exception:
            pass

    return _fallback_response(persona_tag, chunks)


def compile_prompt(
    message: str,
    persona_tag: str,
    chunks: list[dict[str, Any]],
    config: AppConfig,
) -> str:
    context = "\n\n".join(
        f"Source: {chunk['source']} | Score: {chunk['score']:.2f}\n{chunk['text']}"
        for chunk in chunks
    )[: config.max_context_chars]

    return f"""You are a persona-adaptive customer support agent.

Persona: {persona_tag}
Style guide: {PERSONA_STYLE.get(persona_tag, PERSONA_STYLE["Exec"])}

Rules:
- Answer only from the provided support knowledge.
- If a policy, security, billing, or account-sensitive issue is unclear, recommend human follow-up.
- Keep the answer aligned to the detected persona.
- Cite source filenames in a short "Sources" line.

Support knowledge:
{context}

Customer message:
{message}

Adaptive response:"""


def _fallback_response(persona_tag: str, chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "I could not find relevant knowledge-base material for this request. This should be escalated to a human agent."

    top = chunks[0]
    sources = ", ".join(sorted({chunk["source"] for chunk in chunks}))

    if persona_tag == "Frustrated":
        opener = "I'm sorry this has been painful. Here's the fastest path I found:"
    elif persona_tag == "Tech":
        opener = "Based on the retrieved support docs, the likely next technical steps are:"
    else:
        opener = "Here's the concise support position and next step:"

    return (
        f"{opener}\n\n"
        f"{top['text'][:900].strip()}\n\n"
        f"Sources: {sources}"
    )
