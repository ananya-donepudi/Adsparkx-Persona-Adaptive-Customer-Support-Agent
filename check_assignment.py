"""Small assignment checklist runner for the persona support agent."""

from __future__ import annotations

from src.classifier import classify_persona
from src.config import DATA_DIR, get_config
from src.escalator import generate_handoff_json, retrieval_is_sufficient, should_escalate
from src.rag_pipeline import load_and_chunk_documents, retrieve


def main() -> None:
    config = get_config()

    print("Assignment requirement check")
    print("============================")
    print(f"Gemini API key configured: {'yes' if config.gemini_api_key else 'no'}")

    required_docs = [
        DATA_DIR / "api_troubleshooting.md",
        DATA_DIR / "billing_policy.txt",
        DATA_DIR / "password_reset_guide.pdf",
    ]
    for path in required_docs:
        print(f"Knowledge document exists: {path.name}: {'yes' if path.exists() else 'no'}")

    chunks = load_and_chunk_documents(config)
    print(f"Document chunks created: {len(chunks)}")

    tech_message = "Our API token keeps failing with a 401 during webhook setup. What should I check?"
    persona = classify_persona(tech_message)
    retrieved_chunks, score = retrieve(tech_message, config)
    print(f"Persona classifier output: {persona.tag}")
    print(f"Top retrieval source: {retrieved_chunks[0]['source'] if retrieved_chunks else 'none'}")
    print(f"Retrieval score: {score:.3f}")
    print(f"Retrieval quality sufficient: {'yes' if retrieval_is_sufficient(score, config) else 'no'}")

    sensitive_message = "I want a refund for an unauthorized charge. This is unacceptable."
    sensitive_persona = classify_persona(sensitive_message)
    escalate, reasons = should_escalate(sensitive_message, sensitive_persona.tag, 0.5, config)
    handoff_json = generate_handoff_json(sensitive_message, sensitive_persona.tag, reasons, retrieved_chunks)
    print(f"Sensitive issue escalates: {'yes' if escalate else 'no'}")
    print(f"Handoff JSON generated: {'yes' if handoff_json.startswith('{') else 'no'}")


if __name__ == "__main__":
    main()
