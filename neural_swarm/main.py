import streamlit as st
from gui.home_page import home_page
from gui.create_ann_page import create_ann_page
from ann_main import ANNMain
from gui.view_dataset_page import view_dataset_page


def main():
    st.title("Neural Swarm")

    if "ann_main" not in st.session_state:
        st.session_state["ann_main"] = ANNMain()

    if "page" not in st.session_state:
        st.session_state["page"] = "home"

    if st.session_state["page"] == "home":
        home_page(st.session_state["ann_main"])

    if st.session_state["page"] == "view_dataset":
        view_dataset_page(st.session_state["ann_main"])

    if st.session_state["page"] == "create_ann":
        create_ann_page(st.session_state["ann_main"])


if __name__ == "__main__":
    main()
