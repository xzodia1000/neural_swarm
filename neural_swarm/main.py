import streamlit as st
from neural_swarm.ann_main import ANNMain
from neural_swarm.gui import (
    home_page,
    create_ann_page,
    view_dataset_page,
    show_ann_page,
    configure_pso_page,
    train_ann_page,
    view_results_page,
)


def main():
    st.title("Neural Swarm")

    if "ann_main" not in st.session_state:
        st.session_state["ann_main"] = ANNMain()

    if "page" not in st.session_state:
        st.session_state["page"] = "home"

    if st.session_state["page"] == "home":
        home_page.home_page(st.session_state["ann_main"])

    if st.session_state["page"] == "view_dataset":
        view_dataset_page.view_dataset_page(st.session_state["ann_main"])

    if st.session_state["page"] == "create_ann":
        create_ann_page.create_ann_page(st.session_state["ann_main"])

    if st.session_state["page"] == "show_ann":
        show_ann_page.show_ann_page(st.session_state["ann_main"])

    if st.session_state["page"] == "configure_pso":
        configure_pso_page.configure_pso_page(st.session_state["ann_main"])

    if st.session_state["page"] == "train_ann":
        train_ann_page.train_ann_page(st.session_state["ann_main"])

    if st.session_state["page"] == "view_results":
        view_results_page.view_results_page(st.session_state["ann_main"])


if __name__ == "__main__":
    main()
