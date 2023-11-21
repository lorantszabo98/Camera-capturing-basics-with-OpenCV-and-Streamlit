import streamlit as st


def add_property_to_session(entry):
    if "entries" not in st.session_state:
        st.session_state.entries = []
    st.session_state.entries.append(entry)


def get_entries_from_session():
    if "entries" not in st.session_state:
        st.session_state.entries = []

    entries = st.session_state.entries.copy()
    st.session_state.entries = []
    return entries

