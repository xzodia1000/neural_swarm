import streamlit as st


def home_page():
    st.subheader("Upload a dataset.")
    dataset = st.file_uploader("")
    return dataset
