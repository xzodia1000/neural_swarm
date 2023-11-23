import streamlit as st
from home_page import home_page


def main():
    st.title("Neural Swarm")

    if "page" not in st.session_state:
        st.session_state["page"] = "home"

    if st.session_state["page"] == "home":
        home_page()


if __name__ == "__main__":
    main()
