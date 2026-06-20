"""Main Streamlit web UI for the persona-adaptive support agent."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.classifier import classify_persona
from src.config import AppConfig, get_config
from src.escalator import generate_handoff_json, retrieval_is_sufficient, should_escalate
from src.generator import generate_adaptive_response
from src.rag_pipeline import build_or_load_collection, retrieve


st.set_page_config(page_title="Persona Support Agent", page_icon=":speech_balloon:", layout="wide")


def main() -> None:
    config = get_config()

    st.title("Persona-Adaptive Customer Support Agent")
    st.caption("Classifies persona, retrieves support context, adapts the response, and escalates risky cases.")

    with st.sidebar:
        st.subheader("Knowledge Base")
        if st.button("Rebuild Vector Index", use_container_width=True):
            with st.spinner("Rebuilding Chroma index..."):
                build_or_load_collection(config, rebuild=True)
            st.success("Vector index rebuilt.")

        st.subheader("Runtime")
        st.write("Gemini key:", "configured" if config.gemini_api_key else "missing")
        st.write("Top-K:", config.top_k)
        st.write("Minimum score:", config.min_retrieval_score)
        st.write("Escalation score:", config.escalation_score)

    if "history" not in st.session_state:
        st.session_state.history = []

    default_message = "Our API token keeps failing with a 401 during webhook setup. What should I check?"
    message = st.chat_input("Ask a customer support question")

    if message:
        st.session_state.history.append(("user", message))
        with st.spinner("Retrieving support context..."):
            st.session_state.history.append(("assistant", process_message(message, config)))

    if not st.session_state.history:
        st.info("Try this sample: " + default_message)

    for role, payload in st.session_state.history:
        with st.chat_message(role):
            if role == "user":
                st.write(payload)
            else:
                render_assistant_payload(payload)


def process_message(message: str, config: AppConfig) -> dict[str, Any]:
    persona = classify_persona(message)
    chunks: list[dict[str, Any]] = []
    retrieval_score = 0.0

    try:
        chunks, retrieval_score = retrieve(message, config)
        escalate, reasons = should_escalate(message, persona.tag, retrieval_score, config)

        if escalate or not retrieval_is_sufficient(retrieval_score, config):
            answer = "I'll connect you to a human agent please be patient."
            handoff = generate_handoff_json(message, persona.tag, reasons or ["retrieval_quality_low"], chunks)
        else:
            answer = generate_adaptive_response(message, persona.tag, chunks, config)
            handoff = None
        error = None
    except Exception as exc:
        answer = "I could not complete retrieval for this message."
        handoff = generate_handoff_json(message, persona.tag, ["application_error"], chunks)
        error = str(exc)

    return {
        "answer": answer,
        "persona": persona,
        "retrieval_score": retrieval_score,
        "chunks": chunks,
        "handoff": handoff,
        "error": error,
    }


def render_assistant_payload(payload: dict[str, Any]) -> None:
    st.write(payload["answer"])
    if payload.get("error"):
        st.error(payload["error"])


if __name__ == "__main__":
    main()
