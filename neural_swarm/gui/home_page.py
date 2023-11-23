import streamlit as st
import pandas as pd


def home_page(ann_main):
    st.subheader("Upload Dataset")

    if "file" not in st.session_state:
        st.session_state["file"] = None

    if "checkbox_state" not in st.session_state:
        st.session_state["checkbox_state"] = False

    dataset = st.file_uploader("Upload a dataset")

    if dataset is not None:
        header = st.checkbox(
            "Dataset has headers by default.", value=st.session_state["checkbox_state"]
        )

        if header != st.session_state["checkbox_state"]:
            st.session_state["checkbox_state"] = header
            process_file(dataset, header)
            st.rerun()

        if dataset != st.session_state["file"]:
            st.session_state["file"] = dataset
            st.session_state["checkbox_state"] = False
            process_file(dataset, False)

        if "df" in st.session_state:
            st.dataframe(st.session_state["df"])
            st.write(st.session_state["df"].shape)

        if st.button("Next", key="next1"):
            ann_main.set_dataset(st.session_state["df"])
            st.session_state["page"] = "view_dataset"
            st.rerun()


def process_file(file, header):
    """Process the file based on the header state"""
    if header:
        st.session_state["df"] = pd.read_csv(file)
    else:
        df = pd.read_csv(file, header=None)
        df.columns = [f"feature_{i}" for i in range(df.shape[1] - 1)] + ["target"]
        st.session_state["df"] = df
