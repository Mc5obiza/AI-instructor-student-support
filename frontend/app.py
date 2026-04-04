from __future__ import annotations

import json
import uuid
from typing import Any, Iterator

import requests
import streamlit as st


DEFAULT_API_URL = "http://127.0.0.1:8000/ask/stream"


def stream_events(url: str, prompt: str, session_id: str) -> Iterator[dict[str, Any]]:
    """Yield parsed SSE events from the backend stream endpoint."""
    payload = {"prompt": prompt, "session_id": session_id}
    with requests.post(url, json=payload, stream=True, timeout=(5, 600)) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines(decode_unicode=True):
            if raw_line is None:
                continue

            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue

            event_json = line[len("data:") :].strip()
            if not event_json:
                continue

            try:
                yield json.loads(event_json)
            except json.JSONDecodeError:
                continue


def main() -> None:
    st.set_page_config(page_title="RAG Stream Test", layout="centered")
    st.title("RAG Stream Test")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    api_url = st.text_input("Backend SSE URL", value=DEFAULT_API_URL)

    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"Session: {st.session_state.session_id}")
    with col2:
        if st.button("New Session"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Type your question...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    answer = ""
    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            for event in stream_events(api_url, prompt, st.session_state.session_id):
                event_type = str(event.get("type", ""))
                if event_type == "token":
                    answer += str(event.get("token", ""))
                    placeholder.markdown(answer)
                elif event_type == "error":
                    answer = str(event.get("error", "Unknown backend error"))
                    placeholder.error(answer)
                    break
                elif event_type == "done":
                    break
        except requests.RequestException as exc:
            answer = f"Request failed: {exc}"
            placeholder.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
