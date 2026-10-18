"""Session-only prediction history page."""

import pandas as pd
import streamlit as st

from app import configure_page

configure_page("Patient History | SecurePulse")
st.title("Patient History")
st.caption(
    "This page shows synthetic prediction summaries from the current browser "
    "session only. It does not store patient records or raw readings."
)

history = st.session_state.get("prediction_history", [])
if not history:
    st.info("No synthetic prediction results in this browser session yet.")
else:
    history_data = pd.DataFrame(
        [
            {
                "Submission": index,
                "Submitted (UTC)": result["submitted_at"],
                "Risk band": result["risk_band"],
                "Synthetic estimate": result["risk_probability"],
                "Anomaly flagged": result["is_anomaly"],
            }
            for index, result in enumerate(history, start=1)
        ]
    )
    st.dataframe(history_data, use_container_width=True, hide_index=True)
    if len(history_data) > 1:
        st.line_chart(
            history_data,
            x="Submission",
            y="Synthetic estimate",
            height=240,
        )
    else:
        st.info("Submit another synthetic reading to see the estimate trend.")
    st.caption(
        "History contains prediction summaries only and stays in memory for this "
        "browser session. Raw readings are not copied into history or the audit chain."
    )
