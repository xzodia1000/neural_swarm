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
    """
    Main function for the Streamlit UI of the Neural Swarm application.

    This function orchestrates the user interface, managing different pages
    and their states using Streamlit.
    """
    st.title("Neural Swarm")  # Set the title of the web page

    # Initialize ANNMain in session state if not already present
    if "ann_main" not in st.session_state:
        st.session_state["ann_main"] = ANNMain()

    # Set the default page to 'home' if not set
    if "page" not in st.session_state:
        st.session_state["page"] = "home"

    # Render the home page
    if st.session_state["page"] == "home":
        home_page.home_page(st.session_state["ann_main"])

    # Render the dataset view page
    if st.session_state["page"] == "view_dataset":
        view_dataset_page.view_dataset_page(st.session_state["ann_main"])

    # Render the page for creating an ANN
    if st.session_state["page"] == "create_ann":
        create_ann_page.create_ann_page(st.session_state["ann_main"])

    # Render the page for showing the ANN structure
    if st.session_state["page"] == "show_ann":
        show_ann_page.show_ann_page(st.session_state["ann_main"])

    # Render the page for configuring PSO
    if st.session_state["page"] == "configure_pso":
        configure_pso_page.configure_pso_page(st.session_state["ann_main"])

    # Render the page for training the ANN
    if st.session_state["page"] == "train_ann":
        train_ann_page.train_ann_page(st.session_state["ann_main"])

    # Render the page for viewing the results
    if st.session_state["page"] == "view_results":
        view_results_page.view_results_page(st.session_state["ann_main"])


# Run the main function when the script is executed
if __name__ == "__main__":
    main()
