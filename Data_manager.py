import streamlit as st


def add_property_to_session(entry):
    if "entries" not in st.session_state:
        st.session_state.entries = []
    st.session_state.entries.append(entry)


def get_entries_from_session():
    if "entries" not in st.session_state:
        st.session_state.entries = []
    return st.session_state.entries

