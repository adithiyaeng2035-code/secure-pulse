"""Browse the generated synthetic patient profiles."""

import pandas as pd
import streamlit as st

from app import DATA_PATH, configure_page
from modules.data_processor import FEATURE_COLUMNS

configure_page("Synthetic Patients | SecurePulse")
st.title("Synthetic Patient Profiles")
st.caption("Generated demo profiles only — these are not real people or patient records.")

if not DATA_PATH.is_file():
    st.error(f"Synthetic dataset not found at {DATA_PATH}. Run `python generate_data.py`.")
else:
    try:
        patient_data = pd.read_csv(DATA_PATH, usecols=["patient_id", *FEATURE_COLUMNS])
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.error(f"Could not load the synthetic patient sample: {error}")
    else:
        st.metric("Synthetic profiles", f"{len(patient_data):,}")
        st.dataframe(
            patient_data.head(100).rename(
                columns={"patient_id": "Synthetic demo ID"}
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            f"Dataset contains {len(patient_data):,} synthetic profiles; "
            "showing up to the first 100. These are not real people."
        )
