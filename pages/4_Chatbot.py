"""Project assistant powered by the local Ollama service."""

import httpx
import streamlit as st

from app import (
    REMOTE_CHAT_DISABLED_MESSAGE,
    check_chatbot,
    configure_page,
    request_chat_answer,
)
from modules.chatbot import DEFAULT_OLLAMA_MODEL, MAX_QUERY_LENGTH

configure_page("Chatbot | SecurePulse")
st.title("SecurePulse Assistant")
st.caption(
    "Ask anything about SecurePulse: architecture, code, synthetic data, model "
    "training, predictions, API, tests, setup, or privacy. "
    "Do not enter patient names, readings, or personal information. This assistant "
    "cannot provide medical advice. Answers are grounded in the project README "
    "and source files."
)
with st.expander("Example questions"):
    st.markdown(
        """
        - How does `generate_data.py` create the synthetic risk labels?
        - What happens from clicking Estimate risk to seeing the result?
        - Which metrics does `train_model.py` calculate?
        - How does the backend ensure patient measurements are not written to history?
        - Which tests cover invalid blood-pressure readings?
        """
    )

chat_status = check_chatbot()
if chat_status["available"]:
    st.success(f"Local Ollama connected — model {chat_status['model']}.")
else:
    st.warning(f"Local AI chatbot unavailable: {chat_status['detail']}")
    if chat_status["detail"] != REMOTE_CHAT_DISABLED_MESSAGE:
        st.info("Install Ollama, start its local service, and pull the model to enable answers.")
        st.code(f"ollama pull {DEFAULT_OLLAMA_MODEL}")

for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input(
    "Ask about SecurePulse",
    max_chars=MAX_QUERY_LENGTH,
    disabled=not bool(chat_status["available"]),
)
if prompt:
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Asking the local Ollama model..."):
            try:
                answer = request_chat_answer(prompt)
            except httpx.HTTPStatusError as error:
                st.error(f"Chat backend error: {error.response.text}")
            except httpx.HTTPError as error:
                st.error(f"Could not reach the chat backend: {error}")
            else:
                st.markdown(answer["answer"])
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": answer["answer"]}
                )

if st.session_state.chat_messages and st.button("Clear chat"):
    st.session_state.chat_messages.clear()
    st.rerun()
